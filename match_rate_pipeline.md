# 记录如果运行sample脚本并评测match rate

第一步
sample.sh
其实是batch_sample_fast.py
    NUM_ROWS = 100
    SAMPLES_PER_ROW = 30
    改一下输出路径和这些参数
    注意use_comp_feature和use_xrd_feature参数
第二步
拿到输出的 batch_samples_100rows_fast.csv
然后运行
bash /root/autodl-tmp/CrystalFormer/transform_structure.sh

输出 batch_structures.csv

第三步
运行compare_structures.py
