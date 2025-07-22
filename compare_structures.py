#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较生成的晶体结构与ground truth结构的匹配率
从test_comp_cleaned.csv中读取对应material_id的ground truth结构进行比较
处理所有mp-id，按mp-id分组统计匹配情况
"""

import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter, CifParser
import os
import ast
import tempfile
import signal
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

class TimeoutError(Exception):
    """超时异常"""
    pass

def timeout_handler(signum, frame):
    """超时处理函数"""
    raise TimeoutError("操作超时")

def with_timeout(timeout_seconds):
    """超时装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 设置超时信号
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError:
                print(f"    操作超时 ({timeout_seconds}秒)")
                return None
            finally:
                # 清除超时设置
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

def load_and_group_structures(csv_path):
    """
    从CSV文件加载生成的结构，并按mp_id分组
    """
    print(f"正在加载生成的结构: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 按mp_id分组
    grouped_structures = defaultdict(list)
    
    for idx, row in df.iterrows():
        try:
            material_id = row['mp_id']
            
            # CSV中的第一列是'cif'，包含Structure的字典表示
            struct_str = str(row['cif'])  # 确保转换为字符串
            
            # 使用ast.literal_eval安全解析字典字符串
            struct_dict = ast.literal_eval(struct_str)
            
            # 从字典创建Structure对象
            structure = Structure.from_dict(struct_dict)
            
            grouped_structures[material_id].append(structure)
            
            if len(grouped_structures[material_id]) == 1:  # 第一个结构时打印信息
                print(f"开始处理 {material_id}: {structure.composition}")
                
        except Exception as e:
            print(f"加载第 {idx+1} 行结构时出错: {e}")
            continue
    
    print(f"总共找到 {len(grouped_structures)} 个不同的mp-id")
    for material_id, structures in grouped_structures.items():
        print(f"  {material_id}: {len(structures)} 个生成的结构")
    
    return grouped_structures

def create_reference_structure(material_id, test_comp_path="/root/autodl-tmp/CrystalFormer/data/test_comp_cleaned.csv"):
    """
    从test_comp_cleaned.csv中读取对应material_id的CIF结构作为参考结构
    """
    print(f"\n创建参考结构，material_id: {material_id}")
    
    # 读取test_comp_cleaned.csv
    test_df = pd.read_csv(test_comp_path)
    
    # 查找对应的material_id
    matching_row = test_df[test_df['material_id'] == material_id]
    
    if matching_row.empty:
        raise ValueError(f"在 {test_comp_path} 中找不到 material_id: {material_id}")
    
    if len(matching_row) > 1:
        print(f"警告: 找到 {len(matching_row)} 个匹配的 material_id，使用第一个")
    
    # 获取CIF结构数据
    cif_data = matching_row.iloc[0]['cif']
    
    try:
        # CIF数据是字符串格式，需要用CifParser解析
        # 创建临时文件来存储CIF数据
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as temp_file:
            temp_file.write(str(cif_data))
            temp_file_path = temp_file.name
        
        try:
            # 使用CifParser解析CIF文件
            parser = CifParser(temp_file_path)
            structure = parser.get_structures()[0]  # 取第一个结构
            
            print(f"参考结构:")
            print(f"  化学式: {structure.composition}")
            print(f"  空间群: {structure.get_space_group_info()}")
            print(f"  晶格参数: {structure.lattice.parameters}")
            
            return structure
            
        finally:
            # 清理临时文件
            os.unlink(temp_file_path)
        
    except Exception as e:
        raise ValueError(f"解析CIF数据时出错: {e}")

def compare_single_structure(matcher, gen_struct, reference_structure, structure_idx):
    """
    比较单个结构，带超时保护
    """
    try:
        # 使用超时保护的匹配
        @with_timeout(30)  # 30秒超时
        def match_with_timeout():
            return matcher.fit(gen_struct, reference_structure)
        
        is_match = match_with_timeout()
        
        if is_match is None:  # 超时
            return False, None, "超时"
        
        rms_dist = None
        if is_match:
            # RMS距离计算也可能很慢，同样加上超时
            @with_timeout(10)  # 10秒超时
            def rms_with_timeout():
                return matcher.get_rms_dist(gen_struct, reference_structure)
            
            rms_result = rms_with_timeout()
            if rms_result is not None:
                rms_dist = rms_result[0]
                return True, rms_dist, f"匹配,RMS={rms_dist:.4f}"
            else:
                return True, None, "匹配,RMS超时"
        else:
            return False, None, "不匹配"
        
    except Exception as e:
        return False, None, f"错误: {str(e)[:30]}"

def compare_structures(generated_structures, reference_structure, material_id):
    """
    使用StructureMatcher比较结构
    """
    print(f"\n开始结构匹配 {material_id}...")
    
    # 创建StructureMatcher对象 - 适中的匹配条件，避免过于宽松导致计算缓慢
    matcher = StructureMatcher(
        ltol=0.2,      # 晶格参数容差 (降低以提高速度)
        stol=0.3,      # 位点容差 (降低以提高速度)
        angle_tol=5,   # 角度容差 (降低以提高速度)
        primitive_cell=True,   # 使用原胞进行比较
        scale=True,            # 允许缩放
        attempt_supercell=False,  # 禁用超胞匹配以提高速度
        allow_subset=False,       # 禁用子集匹配以提高速度
        comparator=None
    )
    
    matches = []
    match_count = 0
    rms_distances = []
    
    print(f"参考结构信息:")
    print(f"  化学式: {reference_structure.composition}")
    print(f"  原子数: {len(reference_structure)}")
    print(f"  晶格: a={reference_structure.lattice.a:.3f}, b={reference_structure.lattice.b:.3f}, c={reference_structure.lattice.c:.3f}")
    
    total_structures = len(generated_structures)
    
    # 使用tqdm创建进度条
    with tqdm(total=total_structures, 
              desc=f"比较{material_id}的结构", 
              unit="结构",
              ncols=100) as pbar:
        
        for i, gen_struct in enumerate(generated_structures):
            is_match, rms_dist, status_msg = compare_single_structure(matcher, gen_struct, reference_structure, i)
            
            if is_match:
                match_count += 1
                if rms_dist is not None:
                    rms_distances.append(rms_dist)
            
            matches.append(is_match if is_match is not None else False)
            
            # 更新进度条状态
            pbar.set_postfix({
                '匹配': f"{match_count}/{i+1}", 
                '状态': status_msg[:20],
                '率': f"{match_count/(i+1)*100:.1f}%"
            })
            pbar.update(1)
    
    # 计算匹配率
    match_rate = match_count / len(generated_structures) if generated_structures else 0
    
    print(f"\n=== {material_id} 匹配结果 ===")
    print(f"生成结构数: {len(generated_structures)}")
    print(f"匹配数: {match_count}")
    print(f"匹配率: {match_rate:.2%}")
    if rms_distances:
        print(f"平均RMS距离: {np.mean(rms_distances):.4f}")
        print(f"最小RMS距离: {np.min(rms_distances):.4f}")
    
    return matches, match_rate, rms_distances

def save_reference_cif(reference_structure, material_id, output_dir="reference_structures"):
    """
    保存参考结构为CIF文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"reference_{material_id}.cif")
    writer = CifWriter(reference_structure)
    writer.write_file(output_path)
    print(f"参考结构已保存为: {output_path}")

def save_results_incrementally(all_results, timestamp):
    """
    实时保存结果到CSV文件
    """
    if not all_results:
        return
    
    # 保存汇总结果
    summary_df = pd.DataFrame(all_results)
    summary_filename = f'structure_comparison_summary_comp_only_{timestamp}.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    # 计算当前统计信息
    total_structures = sum(r['total_structures'] for r in all_results)
    total_matches = sum(r['matched_structures'] for r in all_results)
    overall_match_rate = total_matches / total_structures if total_structures > 0 else 0
    
    print(f"✓ 已保存 {len(all_results)} 个material_id的结果到: {summary_filename}")
    print(f"  当前总体匹配率: {overall_match_rate:.1%} ({total_matches}/{total_structures})")

def main():
    # 文件路径
    csv_path = "/root/autodl-tmp/CrystalFormer/no_feature_test_output/batch_structures.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    # 创建时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"开始处理，结果将保存为: structure_comparison_summary_comp_only_{timestamp}.csv")
    
    try:
        # 加载并按mp_id分组生成的结构
        grouped_structures = load_and_group_structures(csv_path)
        
        # 存储所有结果
        all_results = []
        
        print(f"\n开始处理 {len(grouped_structures)} 个material_id...")
        
        # 遍历每个material_id，使用总体进度条
        with tqdm(grouped_structures.items(), 
                  desc="总体进度", 
                  unit="material_id",
                  ncols=120) as material_pbar:
            
            for material_id, generated_structures in material_pbar:
                material_pbar.set_description(f"处理 {material_id}")
                
                try:
                    # 创建参考结构（从test_comp_cleaned.csv中读取对应的CIF）
                    reference_structure = create_reference_structure(material_id)
                    
                    # 保存参考结构为CIF文件
                    save_reference_cif(reference_structure, material_id)
                    
                    # 比较结构
                    matches, match_rate, rms_distances = compare_structures(
                        generated_structures, reference_structure, material_id
                    )
                    
                    # 保存单个material_id的详细结果
                    results_df = pd.DataFrame({
                        'material_id': [material_id] * len(matches),
                        'structure_id': range(1, len(matches) + 1),
                        'is_match': matches
                    })
                    # detailed_filename = f'detailed_results_{material_id}_{timestamp}.csv'
                    # results_df.to_csv(detailed_filename, index=False)
                    
                    # 收集汇总统计
                    all_results.append({
                        'material_id': material_id,
                        'total_structures': len(generated_structures),
                        'matched_structures': sum(matches),
                        'match_rate': match_rate,
                        'avg_rms_distance': np.mean(rms_distances) if rms_distances else None,
                        'min_rms_distance': np.min(rms_distances) if rms_distances else None,
                        'composition': str(reference_structure.composition)
                    })
                    
                    # 实时保存汇总结果
                    save_results_incrementally(all_results, timestamp)
                    
                    # 更新总体进度条状态
                    total_matched = sum(r['matched_structures'] for r in all_results)
                    total_structures = sum(r['total_structures'] for r in all_results)
                    overall_rate = total_matched / total_structures if total_structures > 0 else 0
                    
                    material_pbar.set_postfix({
                        '当前匹配率': f"{match_rate:.1%}",
                        '总体匹配率': f"{overall_rate:.1%}",
                        '总匹配': f"{total_matched}/{total_structures}"
                    })
                    
                except KeyboardInterrupt:
                    print(f"\n用户中断程序，已处理 {len(all_results)} 个material_id")
                    save_results_incrementally(all_results, timestamp)
                    print("结果已保存，程序退出")
                    return
                    
                except Exception as e:
                    print(f"\n处理 {material_id} 时出错: {e}")
                    all_results.append({
                        'material_id': material_id,
                        'total_structures': len(generated_structures),
                        'matched_structures': 0,
                        'match_rate': 0.0,
                        'avg_rms_distance': None,
                        'min_rms_distance': None,
                        'composition': 'Error'
                    })
                    
                    # 即使出错也保存结果
                    save_results_incrementally(all_results, timestamp)
                    continue
        
        # 最终保存汇总结果
        final_summary_df = pd.DataFrame(all_results)
        final_filename = f'structure_comparison_summary_comp_only_{timestamp}_final.csv'
        final_summary_df.to_csv(final_filename, index=False)
        
        # 打印总体统计
        print("\n" + "="*80)
        print("总体匹配情况统计:")
        print("="*80)
        
        for result in all_results:
            print(f"{result['material_id']:>15}: "
                  f"{result['matched_structures']:>2}/{result['total_structures']:>2} "
                  f"({result['match_rate']:>6.1%}) - {result['composition']}")
        
        # 计算总体统计
        total_structures = sum(r['total_structures'] for r in all_results)
        total_matches = sum(r['matched_structures'] for r in all_results)
        overall_match_rate = total_matches / total_structures if total_structures > 0 else 0
        
        print("-"*80)
        print(f"{'总计':>15}: {total_matches:>2}/{total_structures:>2} ({overall_match_rate:>6.1%})")
        print(f"平均每个mp-id的匹配率: {np.mean([r['match_rate'] for r in all_results]):.1%}")
        
        print(f"\n最终结果已保存为: {final_filename}")
        
    except KeyboardInterrupt:
        print(f"\n程序被中断，已处理的结果已保存")
        if 'all_results' in locals():
            save_results_incrementally(all_results, timestamp)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        if 'all_results' in locals() and all_results:
            print("尝试保存已处理的结果...")
            save_results_incrementally(all_results, timestamp)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 