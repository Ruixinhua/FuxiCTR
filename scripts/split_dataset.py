#!/usr/bin/env python3
"""
数据集划分脚本
专门处理大文件，使用流式处理避免内存溢出
"""

import pandas as pd
import pyarrow.parquet as pq
import os
import sys
from pathlib import Path

def split_dataset_optimized(input_file, output_dir, num_parts=4):
    """
    使用pyarrow流式处理将parquet文件划分为指定数量的部分
    避免将整个文件加载到内存中
    
    Args:
        input_file (str): 输入文件路径
        output_dir (str): 输出目录
        num_parts (int): 划分部分数量，默认4
    """
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"开始处理文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"划分为 {num_parts} 个部分")
    
    # 使用pyarrow读取parquet文件信息
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    print(f"文件总行数: {total_rows}")
    
    # 计算每个部分的起始和结束位置
    rows_per_part = total_rows // num_parts
    remainder = total_rows % num_parts
    
    print(f"每个部分大约 {rows_per_part} 行")
    
    # 分块读取并写入
    start_row = 0
    
    for part in range(1, num_parts + 1):
        # 计算当前部分的结束行
        if part <= remainder:
            end_row = start_row + rows_per_part + 1
        else:
            end_row = start_row + rows_per_part
        
        print(f"正在处理 part{part:02d}: 行 {start_row} 到 {end_row-1}")
        
        # 使用pandas直接读取整个文件，然后切片
        # 这样可以避免复杂的row group处理
        df = pd.read_parquet(input_file)
        current_chunk = df.iloc[start_row:end_row]
        
        # 输出文件路径
        output_file = os.path.join(output_dir, f"part{part:02d}.parquet")
        
        # 保存为parquet文件
        current_chunk.to_parquet(output_file, engine='pyarrow', index=False)
        
        print(f"已保存 part{part:02d}: {len(current_chunk)} 行")
        
        start_row = end_row
    
    print("数据集划分完成！")
    print(f"输出文件:")
    for part in range(1, num_parts + 1):
        output_file = os.path.join(output_dir, f"part{part:02d}.parquet")
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"  - part{part:02d}.parquet ({file_size:.2f} MB)")

def split_dataset_chunked(input_file, output_dir, num_parts=4, chunk_size=10000000):
    """
    使用分块读取方式处理超大文件
    
    Args:
        input_file (str): 输入文件路径
        output_dir (str): 输出目录
        num_parts (int): 划分部分数量，默认4
        chunk_size (int): 每次读取的行数，默认50000
    """
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"开始分块处理文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"划分为 {num_parts} 个部分，每次读取 {chunk_size} 行")
    
    # 初始化输出文件列表
    output_files = []
    writers = []
    
    for part in range(1, num_parts + 1):
        output_file = os.path.join(output_dir, f"part{part:02d}.parquet")
        output_files.append(output_file)
        writers.append(None)  # 延迟初始化writer
    
    # 分块读取并写入
    current_chunk_num = 0
    total_processed = 0
    
    # 使用pyarrow进行流式读取
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(input_file)
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk = batch.to_pandas()
        current_chunk_num += 1
        chunk_rows = len(chunk)
        total_processed += chunk_rows
        
        print(f"处理第 {current_chunk_num} 个chunk: {chunk_rows} 行 (总计: {total_processed} 行)")
        
        # 计算这个chunk应该分配到哪些part中
        start_pos = total_processed - chunk_rows
        
        for part in range(1, num_parts + 1):
            # 计算每个part的行数范围
            rows_per_part = total_processed // num_parts
            remainder = total_processed % num_parts
            
            part_start = (part - 1) * rows_per_part + min(part - 1, remainder)
            part_end = part_start + rows_per_part + (1 if part <= remainder else 0)
            
            # 检查当前chunk是否与这个part有交集
            chunk_start = start_pos
            chunk_end = start_pos + chunk_rows
            
            if chunk_end > part_start and chunk_start < part_end:
                # 计算交集范围
                intersect_start = max(chunk_start, part_start) - chunk_start
                intersect_end = min(chunk_end, part_end) - chunk_start
                
                if intersect_end > intersect_start:
                    # 提取交集部分
                    part_chunk = chunk.iloc[intersect_start:intersect_end]
                    
                    # 写入对应的part文件
                    output_file = output_files[part - 1]
                    
                    if writers[part - 1] is None:
                        # 首次写入，创建新文件
                        part_chunk.to_parquet(output_file, engine='pyarrow', index=False)
                        writers[part - 1] = True
                    else:
                        # 追加到现有文件
                        existing_df = pd.read_parquet(output_file)
                        combined_df = pd.concat([existing_df, part_chunk], ignore_index=True)
                        combined_df.to_parquet(output_file, engine='pyarrow', index=False)
    
    print("数据集划分完成！")
    print(f"输出文件:")
    for part in range(1, num_parts + 1):
        output_file = output_files[part - 1]
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            df = pd.read_parquet(output_file)
            print(f"  - part{part:02d}.parquet ({len(df)} 行, {file_size:.2f} MB)")

def main():
    # 待分割文件的路径
    input_file = "/scratch/dliu2/FuxiCTR/data/mask_merge/taobaoad_x1_050_050_maskmerge/train.parquet"
    
    # 输出目录（在相同目录下创建split子目录）
    output_dir = "/scratch/dliu2/FuxiCTR/data/mask_merge/taobaoad_x1_050_050_maskmerge/train/"

    chunk_size = 10000000 # 每次读取1000万行
    num_parts = 8      # 划分为8个子数据集
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    try:
        print("尝试使用分块版本...")
        split_dataset_chunked(input_file, output_dir, num_parts=num_parts, chunk_size=chunk_size)
    except Exception as e2:
        print(f"分块版本也失败: {e2}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    test = pd.DataFrame()


