#!/bin/bash

echo "开始批量结构转换 - 统一处理所有spacegroup"
echo "================================================"

# 配置参数
OUTPUT_PATH="test_output/"
NUM_IO_PROCESS=40

echo "输出路径: $OUTPUT_PATH"
echo "IO进程数: $NUM_IO_PROCESS"
echo ""

# 检查是否存在batch_samples开头的CSV文件
if ! ls ${OUTPUT_PATH}batch_samples*.csv 1> /dev/null 2>&1; then
    echo "❌ 错误：找不到 batch_samples*.csv 文件"
    echo "请先运行批量采样脚本生成样本数据"
    exit 1
fi

echo "🔍 找到批量采样文件："
ls ${OUTPUT_PATH}batch_samples*.csv
echo ""

echo "🚀 开始结构转换..."
echo "================================================"

# 运行新的awl2struct.py脚本
python ./scripts/awl2struct.py \
    --output_path "$OUTPUT_PATH" \
    --num_io_process "$NUM_IO_PROCESS"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "🎉 结构转换完成！"
    echo "输出文件: ${OUTPUT_PATH}batch_structures.csv"
    
    # 显示结果文件信息
    if [ -f "${OUTPUT_PATH}batch_structures.csv" ]; then
        echo "📊 文件大小: $(du -h ${OUTPUT_PATH}batch_structures.csv | cut -f1)"
        echo "📝 样本数量: $(tail -n +2 ${OUTPUT_PATH}batch_structures.csv | wc -l)"
        echo "🔍 文件预览:"
        head -3 "${OUTPUT_PATH}batch_structures.csv"
    fi
    echo "================================================"
else
    echo ""
    echo "❌ 结构转换失败！"
    exit 1
fi
