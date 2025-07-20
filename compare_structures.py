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
from collections import defaultdict

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

def compare_structures(generated_structures, reference_structure, material_id):
    """
    使用StructureMatcher比较结构
    """
    print(f"\n开始结构匹配 {material_id}...")
    
    # 创建StructureMatcher对象 - 非常宽松的匹配条件
    matcher = StructureMatcher(
        ltol=0.3,      # 晶格参数容差
        stol=0.5,      # 位点容差
        angle_tol=10,  # 角度容差
        primitive_cell=True,   # 使用原胞进行比较
        scale=True,            # 允许缩放
        attempt_supercell=True,  # 尝试超胞匹配
        allow_subset=True,       # 允许子集匹配
        comparator=None
    )
    
    matches = []
    match_count = 0
    rms_distances = []
    
    print(f"参考结构信息:")
    print(f"  化学式: {reference_structure.composition}")
    print(f"  原子数: {len(reference_structure)}")
    print(f"  晶格: a={reference_structure.lattice.a:.3f}, b={reference_structure.lattice.b:.3f}, c={reference_structure.lattice.c:.3f}")
    
    for i, gen_struct in enumerate(generated_structures):
        try:
            # 检查是否匹配
            is_match = matcher.fit(gen_struct, reference_structure)
            
            if is_match:
                match_count += 1
                # 获取匹配的详细信息
                rms_dist = matcher.get_rms_dist(gen_struct, reference_structure)
                if rms_dist is not None:
                    rms_distances.append(rms_dist[0])
                
            matches.append(is_match)
            
        except Exception as e:
            print(f"    结构 {i+1} 比较时出错: {e}")
            matches.append(False)
    
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

def main():
    # 文件路径
    csv_path = "/root/autodl-tmp/CrystalFormer/test_output/batch_structures.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    try:
        # 加载并按mp_id分组生成的结构
        grouped_structures = load_and_group_structures(csv_path)
        
        # 存储所有结果
        all_results = []
        
        # 遍历每个material_id
        for material_id, generated_structures in grouped_structures.items():
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
                results_filename = f'detailed_results_{material_id}.csv'
                # results_df.to_csv(results_filename, index=False)
                
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
                
            except Exception as e:
                print(f"处理 {material_id} 时出错: {e}")
                all_results.append({
                    'material_id': material_id,
                    'total_structures': len(generated_structures),
                    'matched_structures': 0,
                    'match_rate': 0.0,
                    'avg_rms_distance': None,
                    'min_rms_distance': None,
                    'composition': 'Error'
                })
                continue
        
        # 保存汇总结果
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv('structure_comparison_summary.csv', index=False)
        
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
        
        print(f"\n汇总结果已保存为: structure_comparison_summary.csv")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 