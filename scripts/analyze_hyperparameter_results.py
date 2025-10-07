#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析DTCN模型超参数实验结果的脚本
支持对不同超参数组合的性能进行统计分析和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HyperparameterAnalyzer:
    """超参数分析器"""
    
    def __init__(self, csv_path, output_dir=None):
        """
        初始化分析器
        
        Args:
            csv_path: CSV文件路径
            output_dir: 输出目录，默认为CSV文件所在目录
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        if output_dir is None:
            self.output_dir = Path(csv_path).parent / f"{Path(csv_path).stem}_analysis"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检测可用的列
        self.available_cols = self.df.columns.tolist()
        print(f"✓ 成功读取CSV文件，共 {len(self.df)} 行数据")
        print(f"✓ 可用列: {', '.join(self.available_cols)}")
        
    def filter_data(self, filter_conditions):
        """
        根据条件筛选数据
        
        Args:
            filter_conditions: 字典，{列名: 值} 的形式
        
        Returns:
            筛选后的DataFrame
        """
        filtered_df = self.df.copy()
        
        for col, value in filter_conditions.items():
            if col in filtered_df.columns:
                if isinstance(value, (list, tuple)):
                    # 如果是列表，使用isin
                    filtered_df = filtered_df[filtered_df[col].isin(value)]
                else:
                    # 否则使用等于
                    filtered_df = filtered_df[filtered_df[col] == value]
                print(f"  - 筛选条件: {col} = {value}, 剩余 {len(filtered_df)} 行")
            else:
                print(f"  ! 警告: 列 '{col}' 不存在，跳过此筛选条件")
        
        return filtered_df
    
    def analyze_parameter(self, 
                         target_param, 
                         filter_conditions=None,
                         metrics=None,
                         group_by_params=None):
        """
        分析目标参数对性能的影响
        
        Args:
            target_param: 要分析的目标参数名
            filter_conditions: 筛选条件字典
            metrics: 要分析的指标列表，默认为['test_auc', 'test_logloss']
            group_by_params: 额外的分组参数（比如training_mode）
        
        Returns:
            分析结果DataFrame
        """
        if metrics is None:
            metrics = ['test_auc', 'test_logloss']
        
        # 筛选数据
        if filter_conditions:
            print(f"\n应用筛选条件:")
            filtered_df = self.filter_data(filter_conditions)
        else:
            filtered_df = self.df.copy()
        
        if len(filtered_df) == 0:
            print("错误: 筛选后没有数据！")
            return None
        
        # 确定分组列
        group_cols = [target_param]
        if group_by_params:
            group_cols.extend(group_by_params)
        
        # 检查所有分组列是否存在
        missing_cols = [col for col in group_cols if col not in filtered_df.columns]
        if missing_cols:
            print(f"错误: 以下列不存在: {missing_cols}")
            return None
        
        # 检查指标列是否存在
        available_metrics = [m for m in metrics if m in filtered_df.columns]
        if not available_metrics:
            print(f"错误: 没有可用的指标列！")
            return None
        
        # 分组统计
        results = []
        
        for group_values, group_data in filtered_df.groupby(group_cols):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            
            result = {}
            for i, col in enumerate(group_cols):
                result[col] = group_values[i]
            
            result['count'] = len(group_data)
            
            # 计算每个指标的统计量
            for metric in available_metrics:
                result[f'{metric}_mean'] = group_data[metric].mean()
                result[f'{metric}_std'] = group_data[metric].std()
                result[f'{metric}_min'] = group_data[metric].min()
                result[f'{metric}_max'] = group_data[metric].max()
            
            results.append(result)
        
        result_df = pd.DataFrame(results)
        
        # 按目标参数排序
        result_df = result_df.sort_values(by=target_param)
        
        return result_df
    
    def plot_parameter_effect(self, 
                             result_df, 
                             target_param,
                             metrics=None,
                             group_by_param=None,
                             title_prefix=""):
        """
        绘制参数影响图
        
        Args:
            result_df: analyze_parameter返回的结果DataFrame
            target_param: 目标参数名
            metrics: 要绘制的指标列表
            group_by_param: 分组参数（用于绘制多条曲线）
            title_prefix: 图表标题前缀
        """
        if metrics is None:
            metrics = ['test_auc', 'test_logloss']
        
        # 检查可用指标
        available_metrics = [m for m in metrics 
                           if f'{m}_mean' in result_df.columns]
        
        if not available_metrics:
            print("错误: 没有可用的指标数据！")
            return
        
        # 创建子图
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            if group_by_param and group_by_param in result_df.columns:
                # 按分组参数绘制多条曲线
                for group_value in result_df[group_by_param].unique():
                    group_data = result_df[result_df[group_by_param] == group_value]
                    
                    ax.errorbar(
                        group_data[target_param],
                        group_data[f'{metric}_mean'],
                        yerr=group_data[f'{metric}_std'],
                        marker='o',
                        label=f'{group_by_param}={group_value}',
                        capsize=5,
                        linewidth=2,
                        markersize=8
                    )
            else:
                # 绘制单条曲线
                ax.errorbar(
                    result_df[target_param],
                    result_df[f'{metric}_mean'],
                    yerr=result_df[f'{metric}_std'],
                    marker='o',
                    capsize=5,
                    linewidth=2,
                    markersize=8,
                    color='#2E86AB'
                )
            
            ax.set_xlabel(target_param, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{title_prefix}{metric} vs {target_param}', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if group_by_param and group_by_param in result_df.columns:
                ax.legend()
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"{target_param}_effect"
        if group_by_param:
            filename += f"_by_{group_by_param}"
        filename += ".png"
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存至: {save_path}")
        
        plt.close()
    
    def save_analysis_report(self, result_df, analysis_name, filter_conditions=None):
        """
        保存分析报告
        
        Args:
            result_df: 分析结果DataFrame
            analysis_name: 分析名称
            filter_conditions: 筛选条件字典
        """
        # 保存CSV
        csv_path = self.output_dir / f"{analysis_name}_results.csv"
        result_df.to_csv(csv_path, index=False)
        print(f"✓ 结果已保存至: {csv_path}")
        
        # 保存文本报告
        txt_path = self.output_dir / f"{analysis_name}_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"超参数分析报告: {analysis_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"源文件: {self.csv_path}\n")
            f.write(f"分析时间: {pd.Timestamp.now()}\n\n")
            
            if filter_conditions:
                f.write("筛选条件:\n")
                for key, value in filter_conditions.items():
                    f.write(f"  - {key} = {value}\n")
                f.write("\n")
            
            f.write("分析结果:\n")
            f.write("-" * 80 + "\n")
            f.write(result_df.to_string(index=False))
            f.write("\n\n")
            
            # 找出最佳配置
            if 'test_auc_mean' in result_df.columns:
                best_idx = result_df['test_auc_mean'].idxmax()
                f.write("最佳配置 (按 test_auc 排序):\n")
                f.write("-" * 80 + "\n")
                for col in result_df.columns:
                    f.write(f"  {col}: {result_df.loc[best_idx, col]}\n")
        
        print(f"✓ 报告已保存至: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description='分析DTCN模型超参数实验结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 分析distance_loss_weight的影响（固定其他参数为0）:
   python analyze_hyperparameter_results.py \\
       --csv result.csv \\
       --target distance_loss_weight \\
       --filter knowledge_distillation_loss_weight=0 field_uniformity_loss_weight=0

2. 分析distance_loss_weight的影响，并按training_mode分组:
   python analyze_hyperparameter_results.py \\
       --csv result.csv \\
       --target distance_loss_weight \\
       --filter knowledge_distillation_loss_weight=0 field_uniformity_loss_weight=0 \\
       --group training_mode

3. 分析training_mode的影响:
   python analyze_hyperparameter_results.py \\
       --csv result.csv \\
       --target training_mode \\
       --filter distance_loss_weight=10.0

4. 自定义指标:
   python analyze_hyperparameter_results.py \\
       --csv result.csv \\
       --target distance_loss_weight \\
       --metrics test_auc test_logloss val_auc val_logloss
        """
    )
    
    parser.add_argument('--csv', required=True, help='CSV结果文件路径')
    parser.add_argument('--target', required=True, help='要分析的目标参数名')
    parser.add_argument('--filter', nargs='+', 
                       help='筛选条件，格式: param1=value1 param2=value2')
    parser.add_argument('--group', help='分组参数（如training_mode）')
    parser.add_argument('--metrics', nargs='+', 
                       default=['test_auc', 'test_logloss'],
                       help='要分析的指标列表')
    parser.add_argument('--output', help='输出目录路径')
    parser.add_argument('--title', default='', help='图表标题前缀')
    
    args = parser.parse_args()
    
    # 解析筛选条件
    filter_conditions = {}
    if args.filter:
        for condition in args.filter:
            if '=' not in condition:
                print(f"警告: 忽略无效的筛选条件 '{condition}'")
                continue
            
            key, value = condition.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # 尝试转换为数字
            try:
                value = float(value)
            except ValueError:
                pass  # 保持字符串
            
            filter_conditions[key] = value
    
    print("=" * 80)
    print("开始分析超参数实验结果")
    print("=" * 80)
    
    # 创建分析器
    analyzer = HyperparameterAnalyzer(args.csv, args.output)
    
    # 分析参数影响
    print(f"\n分析目标参数: {args.target}")
    
    group_by_params = [args.group] if args.group else None
    
    result_df = analyzer.analyze_parameter(
        target_param=args.target,
        filter_conditions=filter_conditions,
        metrics=args.metrics,
        group_by_params=group_by_params
    )
    
    if result_df is None or len(result_df) == 0:
        print("错误: 没有可分析的数据！")
        return
    
    print(f"\n分析结果摘要:")
    print("-" * 80)
    print(result_df.to_string(index=False))
    print()
    
    # 绘制图表
    analyzer.plot_parameter_effect(
        result_df=result_df,
        target_param=args.target,
        metrics=args.metrics,
        group_by_param=args.group,
        title_prefix=args.title
    )
    
    # 保存报告
    analysis_name = f"{args.target}_analysis"
    if args.group:
        analysis_name += f"_by_{args.group}"
    
    analyzer.save_analysis_report(
        result_df=result_df,
        analysis_name=analysis_name,
        filter_conditions=filter_conditions
    )
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print(f"所有结果已保存至: {analyzer.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

