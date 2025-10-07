#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制超参数性能变化折线图
用于展示不同超参数值下模型性能的变化趋势
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


# ========== 配置参数 ==========
# 图形尺寸和DPI（适合学术论文）
FIGURE_SIZE = (10, 6)  # 宽度, 高度（英寸）
DPI = 300  # 分辨率

# 字体大小（学术论文标准）
FONT_SIZES = {
    'title': 16,
    'label': 24,
    'tick': 18,
    'legend': 18
}

# 线条样式
LINE_STYLES = {
    'linewidth': 2.5,
    'markersize': 12,
    'alpha': 0.8  # 透明度
}

# 颜色方案（色盲友好）
COLORS = {
    'auc': '#2E86AB',      # 蓝色
    'logloss': '#A23B72'   # 紫红色
}

X_LABEL_MAP = {
    'distance_loss_weight': 'Distance Loss Weight',
}


def load_data(csv_file):
    """加载CSV数据"""
    df = pd.read_csv(csv_file)
    print(f"成功加载数据: {csv_file}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    return df


def setup_matplotlib_style():
    """设置matplotlib全局样式"""
    plt.rcParams['font.size'] = FONT_SIZES['tick']
    plt.rcParams['axes.labelsize'] = FONT_SIZES['label']
    plt.rcParams['axes.titlesize'] = FONT_SIZES['title']
    plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick']
    plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick']
    plt.rcParams['legend.fontsize'] = FONT_SIZES['legend']
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['font.family'] = 'sans-serif'


def plot_dual_axis(df, x_col, output_path=None, show_std=True):
    """
    绘制双y轴折线图（AUC和LogLoss）
    
    参数:
        df: DataFrame，包含数据
        x_col: str，x轴列名（如'distance_loss_weight'）
        output_path: str，输出文件路径（可选）
        show_std: bool，是否显示误差棒
    """
    setup_matplotlib_style()
    
    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    
    # 绘制AUC（左y轴）
    x = df[x_col]
    auc_mean = df['test_auc_mean']
    color_auc = COLORS['auc']
    
    line1 = ax1.plot(x, auc_mean, 
                     color=color_auc, 
                     marker='o', 
                     linewidth=LINE_STYLES['linewidth'],
                     markersize=LINE_STYLES['markersize'],
                     label='AUC',
                     alpha=LINE_STYLES['alpha'])
    
    if show_std and 'test_auc_std' in df.columns:
        auc_std = df['test_auc_std']
        ax1.fill_between(x, auc_mean - auc_std, auc_mean + auc_std, 
                         color=color_auc, alpha=0.2)
    
    ax1.set_xlabel(X_LABEL_MAP.get(x_col, x_col), fontsize=FONT_SIZES['label'], fontweight='bold')
    ax1.set_ylabel('AUC (%)', fontsize=FONT_SIZES['label'], fontweight='bold', color=color_auc)
    ax1.tick_params(axis='y', labelcolor=color_auc, labelsize=FONT_SIZES['tick'])
    ax1.tick_params(axis='x', labelsize=FONT_SIZES['tick'])
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')  # 只显示横向网格线
    
    # 绘制LogLoss（右y轴）
    ax2 = ax1.twinx()
    logloss_mean = df['test_logloss_mean']
    color_logloss = COLORS['logloss']
    
    line2 = ax2.plot(x, logloss_mean, 
                     color=color_logloss, 
                     marker='s', 
                     linewidth=LINE_STYLES['linewidth'],
                     markersize=LINE_STYLES['markersize'],
                     label='LogLoss',
                     alpha=LINE_STYLES['alpha'])
    
    if show_std and 'test_logloss_std' in df.columns:
        logloss_std = df['test_logloss_std']
        ax2.fill_between(x, logloss_mean - logloss_std, logloss_mean + logloss_std, 
                         color=color_logloss, alpha=0.2)
    
    ax2.set_ylabel('LogLoss', fontsize=FONT_SIZES['label'], fontweight='bold', color=color_logloss)
    ax2.tick_params(axis='y', labelcolor=color_logloss, labelsize=FONT_SIZES['tick'])
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=FONT_SIZES['legend'], framealpha=0.9)
    
    # plt.title(f'Performance vs {x_col}',
    #           fontsize=FONT_SIZES['title'],
    #           fontweight='bold',
    #           pad=20)

    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"图形已保存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_separate_subplots(df, x_col, output_path=None, show_std=True):
    """
    绘制分开的子图（上AUC，下LogLoss）
    
    参数:
        df: DataFrame，包含数据
        x_col: str，x轴列名（如'distance_loss_weight'）
        output_path: str，输出文件路径（可选）
        show_std: bool，是否显示误差棒
    """
    setup_matplotlib_style()
    
    # 创建子图，调整高度比例和间距
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)  # 减小子图间距
    
    x = df[x_col].astype('str')
    
    # 绘制AUC
    auc_mean = df['test_auc_mean']
    print(x)
    color_auc = COLORS['auc']
    
    ax1.plot(x, auc_mean, 
             color=color_auc, 
             marker='o', 
             linewidth=LINE_STYLES['linewidth'],
             markersize=LINE_STYLES['markersize'],
             label='AUC',
             alpha=LINE_STYLES['alpha'])
    
    if show_std and 'test_auc_std' in df.columns:
        auc_std = df['test_auc_std']
        ax1.fill_between(x, auc_mean - auc_std, auc_mean + auc_std, 
                         color=color_auc, alpha=0.2)
    
    ax1.set_ylabel('AUC (%)', fontsize=FONT_SIZES['label'], fontweight='bold', color=color_auc)
    ax1.tick_params(axis='y', labelcolor=color_auc, labelsize=FONT_SIZES['tick'])
    ax1.tick_params(axis='x', labelsize=FONT_SIZES['tick'])
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')  # 只显示横向网格线
    # 不显示图例（指标名称已在y轴标签中）
    # ax1.legend(fontsize=FONT_SIZES['legend'], framealpha=0.9, loc='best')
    
    # 绘制LogLoss
    logloss_mean = df['test_logloss_mean']
    color_logloss = COLORS['logloss']
    
    ax2.plot(x, logloss_mean, 
             color=color_logloss, 
             marker='s', 
             linewidth=LINE_STYLES['linewidth'],
             markersize=LINE_STYLES['markersize'],
             label='LogLoss',
             alpha=LINE_STYLES['alpha'])
    
    if show_std and 'test_logloss_std' in df.columns:
        logloss_std = df['test_logloss_std']
        ax2.fill_between(x, logloss_mean - logloss_std, logloss_mean + logloss_std, 
                         color=color_logloss, alpha=0.2)
    
    # 使用映射后的x轴标签
    x_label = X_LABEL_MAP.get(x_col, x_col)
    ax2.set_xlabel(x_label, fontsize=FONT_SIZES['label'], fontweight='bold')
    ax2.set_ylabel('LogLoss', fontsize=FONT_SIZES['label'], fontweight='bold', color=color_logloss)
    ax2.tick_params(axis='y', labelcolor=color_logloss, labelsize=FONT_SIZES['tick'])
    ax2.tick_params(axis='x', labelsize=FONT_SIZES['tick'])
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')  # 只显示横向网格线
    # 不显示图例（指标名称已在y轴标签中）
    # ax2.legend(fontsize=FONT_SIZES['legend'], framealpha=0.9, loc='best')
    
    # 隐藏上下子图相邻的边框
    ax1.spines['bottom'].set_visible(False)  # 关闭子图1底部边框
    ax2.spines['top'].set_visible(False)     # 关闭子图2顶部边框
    
    # 移除上下子图之间的x轴刻度
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    plt.tight_layout()
    
    # 添加断轴符号 "//" 连接两个子图
    d = 0.85  # 设置倾斜度
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle='none', color='k', mec='k', mew=2, clip_on=False)
    # 在子图1底部绘制断轴线
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    # 在子图2顶部绘制断轴线
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"图形已保存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='绘制超参数性能变化折线图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法（默认使用上下分开的子图）
  python plot_hyperparameter_performance.py data.csv
  
  # 指定输出文件
  python plot_hyperparameter_performance.py data.csv -o output.png
  
  # 使用双y轴模式
  python plot_hyperparameter_performance.py data.csv --dual-axis
  
  # 不显示误差棒
  python plot_hyperparameter_performance.py data.csv --no-std
  
  # 指定x轴列名
  python plot_hyperparameter_performance.py data.csv -x learning_rate
        """
    )
    
    parser.add_argument('input_csv', type=str, help='输入CSV文件路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出图形文件路径（支持.png, .pdf, .svg等）')
    parser.add_argument('-x', '--x-column', type=str, default='distance_loss_weight',
                        help='x轴列名（默认: distance_loss_weight）')
    parser.add_argument('--dual-axis', action='store_true',
                        help='使用双y轴模式而不是分开的子图（默认使用子图）')
    parser.add_argument('--no-std', action='store_true',
                        help='不显示标准差误差棒')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_data(args.input_csv)
    
    # 确定输出路径
    if args.output is None:
        input_path = Path(args.input_csv)
        output_path = input_path.parent / f"{input_path.stem}_plot.png"
    else:
        output_path = args.output
    
    # 绘制图形
    show_std = not args.no_std
    
    if args.dual_axis:
        plot_dual_axis(df, args.x_column, output_path, show_std)
    else:
        plot_separate_subplots(df, args.x_column, output_path, show_std)
    
    print("完成！")


if __name__ == '__main__':
    main()

