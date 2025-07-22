#!/usr/bin/env python3

import jax
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree
import optax
import os
import multiprocessing
import math
import pandas as pd
import numpy as np 
import ast
from typing import List, Tuple

# 导入所需的模块
from crystalformer.src.utils import GLXYZAW_from_file, letter_to_number
from crystalformer.src.elements import element_dict, element_list
from crystalformer.src.transformer import make_transformer  
from crystalformer.src.sample import sample_crystal
from crystalformer.src.loss import make_loss_fn
import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.wyckoff import mult_table

class BatchSampler:
    def __init__(self, restore_path: str):
        """初始化批量采样器"""
        print("正在初始化批量采样器...")
        
        # 固定参数 (从原始命令中提取)
        self.args = self.create_args()
        
        # 初始化随机数种子
        self.key = jax.random.PRNGKey(42)
        
        # 启用x64精度
        jax.config.update("jax_enable_x64", True)
        
        print("正在创建transformer...")
        # 创建transformer (注意正确的参数顺序)
        self.params, self.transformer = make_transformer(
            self.key, self.args.Nf, self.args.Kx, self.args.Kl, self.args.n_max, 
            self.args.h0_size, 
            self.args.transformer_layers, self.args.num_heads, 
            self.args.key_size, self.args.model_size, self.args.embed_size, 
            self.args.atom_types, self.args.wyck_types,
            self.args.dropout_rate, self.args.attn_dropout,
            use_comp_feature=self.args.use_comp_feature,
            comp_feature_dim=self.args.comp_feature_dim,
            use_xrd_feature=self.args.use_xrd_feature,
            xrd_feature_dim=self.args.xrd_feature_dim
        )
        
        print("正在加载checkpoint...")
        # 加载checkpoint (覆盖初始参数)
        ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path)
        ckpt = checkpoint.load_data(ckpt_filename)
        self.params = ckpt["params"]
        
        # 创建损失函数 (注意正确的参数顺序)
        self.loss_fn, self.logp_fn = make_loss_fn(
            self.args.n_max, self.args.atom_types, self.args.wyck_types, self.args.Kx, self.args.Kl, 
            self.transformer, self.args.lamb_a, self.args.lamb_w, self.args.lamb_l, 
            self.args.use_comp_feature, self.args.use_xrd_feature
        )
        
        # 预计算一些mask (按照main.py的方式)
        # w_mask - 用于Wyckoff位置约束，我们不指定特定的Wyckoff位置，所以设为None
        self.w_mask = None
        
        # atom_mask - 用于元素类型约束
        radioactive_element = [43, 61, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
        noble_gas = [2, 10, 18, 36, 54, 86]
        
        if self.args.remove_radioactive:
            remove_list = radioactive_element + noble_gas
        else:
            remove_list = []
        
        # 创建atom_mask：每个原子位置都有一个mask (shape: n_max, atom_types)
        base_mask = jnp.ones(self.args.atom_types, dtype=int)
        if remove_list:
            base_mask = base_mask.at[remove_list].set(0)
        self.atom_mask = jnp.stack([base_mask] * self.args.n_max, axis=0)
        
        print("✓ 批量采样器初始化完成！")
    
    def create_args(self):
        """创建参数对象"""
        class Args:
            def __init__(self):
                # 模型参数
                self.atom_types = 119
                self.wyck_types = 28
                self.n_max = 21
                self.Nf = 5
                self.Kx = 16
                self.Kl = 16
                self.h0_size = 256
                self.transformer_layers = 16
                self.num_heads = 8
                self.key_size = 64
                self.model_size = 32
                self.embed_size = 32
                self.dropout_rate = 0.4
                self.attn_dropout = 0.4
                
                # 损失参数
                self.lamb_a = 1.0
                self.lamb_w = 1.0
                self.lamb_l = 1.0
                
                # 采样参数
                self.batchsize = 30  # 增加到30，每行只需1个批次
                self.temperature = 1.0
                self.top_p = 1.0
                self.remove_radioactive = False
                
                # 组成特征参数
                self.use_comp_feature = True
                self.comp_feature_dim = 256
                
                # XRD特征参数
                self.use_xrd_feature = False
                self.xrd_feature_dim = 1080
        
        return Args()
    
    def generate_composition_features(self, csv_path: str, data_index: int):
        """生成组成特征"""
        if csv_path is not None and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                if data_index >= len(df):
                    print(f"警告: data_index {data_index} >= 数据集大小 {len(df)}, 使用最后一行")
                    data_index = len(df) - 1
                
                # 从第10列开始读取256维组成特征
                comp_features = df.iloc[data_index, 9:9+self.args.comp_feature_dim].values
                comp_features = np.array(comp_features, dtype=np.float32)
                
                # 读取mp-id (第2列, 索引为1)
                mp_id = df.iloc[data_index, 1]
                
                return jnp.array(comp_features), str(mp_id)
                
            except Exception as e:
                print(f"加载组成特征时出错: {e}")
        
        # 回退到模拟数据
        return jnp.array(np.random.normal(0, 0.1, self.args.comp_feature_dim), dtype=jnp.float32), "mock-mp-id"
    
    def generate_xrd_features(self, csv_path: str, data_index: int):
        """生成XRD特征"""
        if csv_path is not None and os.path.exists(csv_path) and 'xrd' in csv_path:
            try:
                df = pd.read_csv(csv_path)
                
                if data_index >= len(df):
                    print(f"警告: data_index {data_index} >= 数据集大小 {len(df)}, 使用最后一行")
                    data_index = len(df) - 1
                
                if 'xrd_data' in df.columns:
                    # 读取XRD数据（逗号分隔的字符串）
                    xrd_str = df.iloc[data_index]['xrd_data']
                    xrd_values = [float(x) for x in xrd_str.split(',')]
                    xrd_features = np.array(xrd_values, dtype=np.float32)
                    
                    if len(xrd_features) == self.args.xrd_feature_dim:
                        return jnp.array(xrd_features)
                    else:
                        print(f"警告: XRD特征维度不匹配 ({len(xrd_features)} vs {self.args.xrd_feature_dim})")
                        
            except Exception as e:
                print(f"加载XRD特征时出错: {e}")
        
        # 回退到零向量（在生成新结构时，通常没有目标XRD）
        return jnp.zeros(self.args.xrd_feature_dim, dtype=jnp.float32)
    
    def sample_single_row(self, row_index: int, mp_id: str, elements: List[str], 
                         spacegroup: int, num_samples: int, csv_path: str):
        """为单行数据生成样本"""
        print(f"\n正在为第{row_index}行生成样本...")
        print(f"  mp_id: {mp_id}")
        print(f"  elements: {elements}")
        print(f"  spacegroup: {spacegroup}")
        print(f"  样本数: {num_samples}")
        
        # 生成组成特征
        composition_features, loaded_mp_id = self.generate_composition_features(csv_path, row_index)
        
        # 生成XRD特征
        xrd_features = self.generate_xrd_features(csv_path, row_index) if self.args.use_xrd_feature else None
        
        # 位置约束 (按照main.py的方式) - 这不是元素约束！
        constraints = jnp.arange(0, self.args.n_max, 1)
        
        # 元素约束通过修改atom_mask实现
        if elements:
            element_numbers = [letter_to_number(e) for e in elements]
            # 创建只允许特定元素的atom_mask
            base_constraints = jnp.zeros(self.args.atom_types, dtype=int)
            base_constraints = base_constraints.at[tuple(element_numbers)].set(1)
            atom_mask = jnp.stack([base_constraints] * self.args.n_max, axis=0)
        else:
            atom_mask = self.atom_mask
        
        # 批量生成样本
        all_samples = []
        num_batches = math.ceil(num_samples / self.args.batchsize)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.args.batchsize
            end_idx = min(start_idx + self.args.batchsize, num_samples)
            n_sample = end_idx - start_idx
            
            # 分割随机数种子
            self.key, subkey = jax.random.split(self.key)
            
            print(f"  批次 {batch_idx + 1}/{num_batches}: 生成 {n_sample} 个样本...")
            
            # 生成样本
            XYZ, A, W, M, L = sample_crystal(
                subkey, self.transformer, self.params, self.args.n_max, n_sample, 
                self.args.atom_types, self.args.wyck_types, self.args.Kx, self.args.Kl, 
                spacegroup, self.w_mask, atom_mask, self.args.top_p, self.args.temperature, 
                self.args.temperature, constraints, composition_features, xrd_features
            )
            
            # 准备数据
            batch_data = pd.DataFrame()
            batch_data['L'] = np.array(L).tolist()
            batch_data['X'] = np.array(XYZ).tolist()
            batch_data['A'] = np.array(A).tolist()
            batch_data['W'] = np.array(W).tolist()
            batch_data['M'] = np.array(M).tolist()
            batch_data['mp_id'] = [mp_id] * n_sample
            batch_data['source_row'] = [row_index] * n_sample
            batch_data['spacegroup'] = [spacegroup] * n_sample
            batch_data['elements'] = [str(elements)] * n_sample
            
            # 计算log概率
            num_atoms = jnp.sum(M, axis=1)
            length, angle = jnp.split(L, 2, axis=-1)
            length = length/num_atoms[:, None]**(1/3)
            angle = angle * (jnp.pi / 180)  # 转换为弧度
            L_normalized = jnp.concatenate([length, angle], axis=-1)
            
            G = spacegroup * jnp.ones((n_sample), dtype=int)
            
            # 根据特征配置调用不同的logp_fn
            if self.args.use_comp_feature and self.args.use_xrd_feature:
                # 两种特征都使用
                batch_comp_features = jnp.tile(composition_features[None, :], (n_sample, 1))
                batch_xrd_features = jnp.tile(xrd_features[None, :], (n_sample, 1))
                logp_w, logp_xyz, logp_a, logp_l = jax.jit(self.logp_fn, static_argnums=7)(
                    self.params, subkey, G, L_normalized, XYZ, A, W, False, batch_comp_features, batch_xrd_features
                )
            elif self.args.use_comp_feature:
                # 只使用composition特征
                batch_comp_features = jnp.tile(composition_features[None, :], (n_sample, 1))
                logp_w, logp_xyz, logp_a, logp_l = jax.jit(self.logp_fn, static_argnums=7)(
                    self.params, subkey, G, L_normalized, XYZ, A, W, False, batch_comp_features
                )
            else:
                # 不使用额外特征
                logp_w, logp_xyz, logp_a, logp_l = jax.jit(self.logp_fn, static_argnums=7)(
                    self.params, subkey, G, L_normalized, XYZ, A, W, False
                )
            
            batch_data['logp_w'] = np.array(logp_w).tolist()
            batch_data['logp_xyz'] = np.array(logp_xyz).tolist()
            batch_data['logp_a'] = np.array(logp_a).tolist()
            batch_data['logp_l'] = np.array(logp_l).tolist()
            batch_data['logp'] = np.array(logp_xyz + self.args.lamb_w*logp_w + 
                                         self.args.lamb_a*logp_a + self.args.lamb_l*logp_l).tolist()
            
            all_samples.append(batch_data)
        
        # 合并所有批次
        if all_samples:
            result_data = pd.concat(all_samples, ignore_index=True)
            result_data = result_data.sort_values(by='logp', ascending=False)
            print(f"  ✓ 成功生成 {len(result_data)} 个样本")
            return result_data
        else:
            print(f"  ✗ 生成样本失败")
            return None

def extract_csv_data(csv_path: str, num_rows: int = 10) -> List[Tuple[int, str, List[str], int]]:
    """从CSV文件中提取数据"""
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    results = []
    for i in range(min(num_rows, len(df))):
        row = df.iloc[i]
        
        # 获取mp_id (第2列，索引1)
        mp_id = row.iloc[1]
        
        # 获取elements (第7列，索引6)
        elements_str = row.iloc[6]
        try:
            elements_list = ast.literal_eval(elements_str)
        except:
            print(f"警告：无法解析第{i}行的elements: {elements_str}")
            continue
            
        # 获取spacegroup.number (第9列，索引8)
        spacegroup_number = int(row.iloc[8])
        
        results.append((i, mp_id, elements_list, spacegroup_number))
        print(f"第{i}行: mp_id={mp_id}, elements={elements_list}, spacegroup={spacegroup_number}")
    
    return results

def main():
    # 配置参数
    CSV_PATH = "data/test_comp_cleaned_xrd.csv"  # 使用包含XRD数据的文件
    NUM_ROWS = 1000
    SAMPLES_PER_ROW = 30
    RESTORE_PATH = "test_output/adam_bs_90_lr_0.0005_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_16_h0_256_l_16_H_8_k_64_m_32_e_32_drop_0.4_0.4"
    OUTPUT_FILE = "test_output/batch_samples_1000rows_fast_comp_only.csv"
    
    print("开始高速批量采样...")
    print(f"CSV文件: {CSV_PATH}")
    print(f"处理行数: {NUM_ROWS}")
    print(f"每行样本数: {SAMPLES_PER_ROW}")
    print(f"模型路径: {RESTORE_PATH}")
    print("=" * 80)
    
    # 1. 提取CSV数据
    csv_data = extract_csv_data(CSV_PATH, NUM_ROWS)
    
    if not csv_data:
        print("✗ 没有提取到有效的CSV数据")
        return
    
    print(f"\n成功提取 {len(csv_data)} 行数据")
    print("=" * 80)
    
    # 2. 初始化批量采样器 (只需要一次!)
    try:
        sampler = BatchSampler(RESTORE_PATH)
    except Exception as e:
        print(f"✗ 初始化采样器失败: {e}")
        return
    
    print("=" * 80)
    
    # 3. 循环处理每行数据
    all_results = []
    total_start_time = pd.Timestamp.now()
    
    for row_index, mp_id, elements, spacegroup in csv_data:
        start_time = pd.Timestamp.now()
        
        result_data = sampler.sample_single_row(
            row_index, mp_id, elements, spacegroup, SAMPLES_PER_ROW, CSV_PATH
        )
        
        if result_data is not None:
            all_results.append(result_data)
            
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        print(f"  行 {row_index} 耗时: {elapsed:.2f}秒")
    
    # 4. 合并所有结果
    if all_results:
        final_data = pd.concat(all_results, ignore_index=True)
        final_data.to_csv(OUTPUT_FILE, index=False)
        
        total_elapsed = (pd.Timestamp.now() - total_start_time).total_seconds()
        
        print("=" * 80)
        print(f"🎉 批量采样完成！")
        print(f"最终结果保存在: {OUTPUT_FILE}")
        print(f"总共生成了 {len(final_data)} 个样本")
        print(f"总耗时: {total_elapsed:.2f}秒")
        print(f"平均每行耗时: {total_elapsed/len(csv_data):.2f}秒")
        print(f"平均每个样本耗时: {total_elapsed/(len(csv_data)*SAMPLES_PER_ROW):.3f}秒")
    else:
        print("\n❌ 所有行的采样都失败了")

if __name__ == "__main__":
    main() 