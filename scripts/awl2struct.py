import sys
sys.path.append('./crystalformer/src/')

import pandas as pd
import numpy as np
from ast import literal_eval
import multiprocessing
import itertools
import argparse
import os
import glob

from pymatgen.core import Structure, Lattice
from wyckoff import wmax_table, mult_table, symops

symops = np.array(symops)
mult_table = np.array(mult_table)
wmax_table = np.array(wmax_table)


def symmetrize_atoms(g, w, x):
    '''
    symmetrize atoms via, apply all sg symmetry op, finding the generator, and lastly apply symops 
    we need to do that because the sampled atom might not be at the first WP
    Args:
       g: int 
       w: int
       x: (3,)
    Returns:
       xs: (m, 3) symmetrize atom positions
    '''

    # (1) apply all space group symmetry op to the x 
    w_max = wmax_table[g-1].item()
    m_max = mult_table[g-1, w_max].item()
    ops = symops[g-1, w_max, :m_max] # (m_max, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    coords = ops@affine_point # (m_max, 3) 
    coords -= np.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
    # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs  
    #https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
    def dist_to_op0x(coord):
        diff = np.dot(symops[g-1, w, 0], np.array([*coord, 1])) - coord
        diff -= np.rint(diff)
        return np.sum(diff**2) 
   #  loc = np.argmin(jax.vmap(dist_to_op0x)(coords))
    loc = np.argmin([dist_to_op0x(coord) for coord in coords])
    x = coords[loc].reshape(3,)

    # (3) lastly, apply the given symmetry op to x
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]   # (m, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    xs = ops@affine_point # (m, 3)
    xs -= np.floor(xs) # wrap back to 0-1 
    return xs

def get_struct_from_lawx(G, L, A, W, X):
    """
    Get the pymatgen.Structure object from the input data

    Args:
        G: space group number
        L: lattice parameters
        A: element number list
        W: wyckoff letter list
        X: fractional coordinates list
    
    Returns:
        struct: pymatgen.Structure object
    """
    A = A[np.nonzero(A)]
    X = X[np.nonzero(A)]
    W = W[np.nonzero(A)]

    lattice = Lattice.from_parameters(*L)
    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    as_list = [[A[idx] for _ in range(len(xs))] for idx, xs in enumerate(xs_list)]
    A_list = list(itertools.chain.from_iterable(as_list))
    X_list = list(itertools.chain.from_iterable(xs_list))
    struct = Structure(lattice, A_list, X_list)
    return struct.as_dict()


def find_batch_samples_file(output_path):
    """
    查找batch_samples开头的CSV文件
    
    Args:
        output_path: 搜索目录
    
    Returns:
        找到的文件路径，如果没找到返回None
    """
    pattern = os.path.join(output_path, "batch_samples*.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # 如果有多个文件，选择最新的
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def main(args):
    # 查找batch_samples开头的CSV文件
    input_path = find_batch_samples_file(args.output_path)
    
    if input_path is None:
        print(f"❌ 错误: 在 {args.output_path} 中找不到 batch_samples*.csv 文件")
        print("请先运行批量采样脚本生成样本数据")
        return
    
    # 输出文件名
    output_path = os.path.join(args.output_path, "batch_structures.csv")
    
    print(f"🔍 找到输入文件: {input_path}")
    print(f"📝 输出文件: {output_path}")
    print("=" * 60)
    
    # 读取CSV文件
    try:
        origin_data = pd.read_csv(input_path)
        print(f"✅ 成功读取 {len(origin_data)} 行数据")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 检查必要的列
    required_columns = ['L', 'X', 'A', 'W', 'spacegroup']
    missing_columns = [col for col in required_columns if col not in origin_data.columns]
    if missing_columns:
        print(f"❌ 错误: 缺少必要的列: {missing_columns}")
        return
    
    # 获取数据列
    L = origin_data['L'].apply(lambda x: literal_eval(x))
    X = origin_data['X'].apply(lambda x: literal_eval(x))
    A = origin_data['A'].apply(lambda x: literal_eval(x))
    W = origin_data['W'].apply(lambda x: literal_eval(x))
    
    # 从spacegroup列读取每行的spacegroup值
    G = origin_data['spacegroup'].values
    
    # 读取mp_id列（如果存在）
    mp_id_column = None
    if 'mp_id' in origin_data.columns:
        mp_id_column = origin_data['mp_id'].values
        print("✅ 找到 mp_id 列")
    else:
        print("⚠️  未找到 mp_id 列，将使用默认值")
        mp_id_column = [f"sample_{i}" for i in range(len(origin_data))]

    # 转换为numpy数组
    L = np.array(L.tolist())
    X = np.array(X.tolist())
    A = np.array(A.tolist())
    W = np.array(W.tolist())
    
    print(f"📊 数据形状: L{L.shape}, X{X.shape}, A{A.shape}, W{W.shape}")
    print(f"🔢 Spacegroup 范围: {G.min()} - {G.max()}")
    print(f"🔢 唯一的 Spacegroup: {sorted(set(G))}")
    
    print("\n🚀 开始生成结构...")
    
    # 使用多进程生成结构
    try:
        p = multiprocessing.Pool(args.num_io_process)
        structures = p.starmap_async(get_struct_from_lawx, zip(G, L, A, W, X)).get()
        p.close()
        p.join()
        print(f"✅ 成功生成 {len(structures)} 个结构")
    except Exception as e:
        print(f"❌ 生成结构时出错: {e}")
        return

    # 创建输出数据框
    output_data = pd.DataFrame()
    output_data['cif'] = structures
    output_data['mp_id'] = mp_id_column
    
    # 保存结果
    try:
        output_data.to_csv(output_path, index=False)
        print(f"🎉 成功保存结果到: {output_path}")
        print(f"📈 输出统计:")
        print(f"   - 总样本数: {len(output_data)}")
        print(f"   - 包含列: {list(output_data.columns)}")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将批量采样结果转换为晶体结构')
    parser.add_argument('--output_path', default='test_output/', 
                        help='搜索batch_samples*.csv文件的目录路径')
    parser.add_argument('--num_io_process', type=int, default=40, 
                        help='多进程处理的进程数')
    
    args = parser.parse_args()
    main(args)
