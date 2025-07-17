#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def prepare_comp_data_single(input_file, output_file, comp_path):
    """
    合并单个CSV文件和comp_data.csv，基于material_id交集
    """
    
    print(f"正在处理 {input_file} -> {output_file}")
    print("="*50)
    
    print("正在读取文件...")
    
    # 读取输入文件
    try:
        input_df = pd.read_csv(input_file)
        print(f"{input_file} 读取成功: {input_df.shape[0]} 行, {input_df.shape[1]} 列")
        print(f"{input_file} 列名: {list(input_df.columns)}")
    except Exception as e:
        print(f"读取 {input_file} 失败: {e}")
        return
    
    # 读取 comp_data.csv
    try:
        comp_df = pd.read_csv(comp_path)
        print(f"comp_data.csv 读取成功: {comp_df.shape[0]} 行, {comp_df.shape[1]} 列")
        print(f"comp_data.csv 列名: {list(comp_df.columns)}")
    except Exception as e:
        print(f"读取 comp_data.csv 失败: {e}")
        return
    
    # 检查 material_id 列是否存在
    if 'material_id' not in input_df.columns:
        print(f"错误: {input_file} 中没有找到 material_id 列")
        return
    
    # 获取 comp_data.csv 中的 material_id（第2列，索引1）
    comp_material_id_col = comp_df.columns[1]  # 第2列
    print(f"comp_data.csv 中的 material_id 列名: {comp_material_id_col}")
    
    # 获取 composition features（第8列开始，索引7开始）
    feature_cols = comp_df.columns[7:]  # 从第8列开始
    print(f"将提取 {len(feature_cols)} 个 composition features")
    print(f"Feature 列名示例: {list(feature_cols[:5])}...")
    
    # 找到 material_id 交集
    input_material_ids = set(input_df['material_id'])
    comp_material_ids = set(comp_df[comp_material_id_col])
    
    intersection = input_material_ids.intersection(comp_material_ids)
    
    print(f"{input_file} 中的 material_id 数量: {len(input_material_ids)}")
    print(f"comp_data.csv 中的 material_id 数量: {len(comp_material_ids)}")
    print(f"交集数量: {len(intersection)}")
    
    if len(intersection) == 0:
        print("错误: 没有找到共同的 material_id")
        return
    
    # 筛选出交集中的行
    input_filtered = input_df[input_df['material_id'].isin(list(intersection))].copy()
    comp_filtered = comp_df[comp_df[comp_material_id_col].isin(list(intersection))].copy()
    
    print(f"筛选后 {input_file}: {input_filtered.shape[0]} 行")
    print(f"筛选后 comp_data.csv: {comp_filtered.shape[0]} 行")
    
    # 重命名 comp_data 中的 material_id 列以便合并
    if comp_material_id_col != 'material_id':
        comp_filtered = comp_filtered.rename(columns={comp_material_id_col: 'material_id'})
    
    # 只保留需要的列 (material_id + features)
    comp_features = comp_filtered[['material_id'] + list(feature_cols)]
    
    # 基于 material_id 合并，保持输入文件的顺序
    print("正在合并数据...")
    merged_df = input_filtered.merge(comp_features, on='material_id', how='left')
    
    # 检查合并结果
    print(f"合并后数据: {merged_df.shape[0]} 行, {merged_df.shape[1]} 列")
    print(f"新增列数: {len(feature_cols)}")
    
    # 检查是否有缺失值
    missing_features = int(merged_df[feature_cols].isnull().any(axis=1).sum())
    if missing_features > 0:
        print(f"警告: 有 {missing_features} 行的 composition features 缺失")
    
    # 保存结果
    print(f"正在保存到 {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    print("处理完成!")
    print(f"最终文件包含 {merged_df.shape[0]} 行, {merged_df.shape[1]} 列")
    print(f"原始文件列数: {input_df.shape[1]}")
    print(f"新增 composition features 列数: {len(feature_cols)}")
    
    # 显示一些统计信息
    print("\n=== 数据统计 ===")
    print(f"保留的交集比例: {len(intersection)/len(input_material_ids)*100:.1f}%")
    
    # 显示前几行的列名
    print(f"\n=== 最终文件列名 ===")
    print(f"总列数: {len(merged_df.columns)}")
    print(f"前10列: {list(merged_df.columns[:10])}")
    print(f"后10列: {list(merged_df.columns[-10:])}")
    print("\n")

def prepare_comp_data():
    """
    批量处理 train.csv, val.csv, test.csv
    """
    
    # 文件路径配置
    base_path = "/root/autodl-tmp/CrystalFormer/data/"
    comp_path = os.path.join(base_path, "comp_data.csv")
    
    # 要处理的文件列表
    files_to_process = [
        ("train.csv", "train_comp.csv"),
        ("val.csv", "val_comp.csv"),
        ("test.csv", "test_comp.csv")
    ]
    
    for input_filename, output_filename in files_to_process:
        input_path = os.path.join(base_path, input_filename)
        output_path = os.path.join(base_path, output_filename)
        
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"跳过 {input_path}：文件不存在")
            continue
            
        # 处理单个文件
        prepare_comp_data_single(input_path, output_path, comp_path)

if __name__ == "__main__":
    prepare_comp_data()
