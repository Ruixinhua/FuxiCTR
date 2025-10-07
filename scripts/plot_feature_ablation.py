#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制特征消除实验结果柱状图
用于展示不同数量的特征丢弃后模型性能的变化
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
    'tick': 14,
    'legend': 18
}

# 柱状图样式
BAR_STYLES = {
    'width': 0.6,  # 柱子宽度
    'alpha': 0.8,   # 透明度
    'edgecolor': 'black',  # 边框颜色
    'linewidth': 1.5  # 边框粗细
}

# 颜色方案（色盲友好）
COLORS = {
    'auc': '#2E86AB',      # 蓝色
    'logloss': '#A23B72'   # 紫红色
}

# X轴标签映射
X_LABEL_MAP = {
    'drop_features': '# Dropped Personalized Features',
    '#drop_features': '# Dropped Personalized Features',
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


def save_figure_as_svg_and_pdf(fig, output_path):
    """
    保存图形为 SVG 矢量图并转换为 PDF
    
    参数:
        fig: matplotlib figure 对象
        output_path: str 或 Path，输出文件路径（不含扩展名或带.pdf/.svg扩展名）
    """
    output_path = Path(output_path)
    
    # 确定输出路径（去除扩展名）
    if output_path.suffix in ['.pdf', '.svg', '.png']:
        base_path = output_path.with_suffix('')
    else:
        base_path = output_path
    
    svg_path = base_path.with_suffix('.svg')
    pdf_path = base_path.with_suffix('.pdf')
    
    # 保存为 SVG 矢量图
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"SVG 矢量图已保存至: {svg_path}")
    
    # 使用 matplotlib 直接保存为 PDF（PDF 本身就是矢量格式）
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF 文件已保存至: {pdf_path}")
    
    return svg_path, pdf_path


def plot_feature_ablation_bars(df, x_col, output_path=None, show_std=True):
    """
    绘制特征消除实验结果的柱状图（上AUC，下LogLoss）
    
    参数:
        df: DataFrame，包含数据
        x_col: str，x轴列名（如'#drop_features'）
        output_path: str，输出文件路径（可选）
        show_std: bool，是否显示误差棒
    """
    setup_matplotlib_style()
    
    # 创建子图，调整高度比例和间距
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True)
    fig.subplots_adjust(hspace=0.15)  # 减小子图间距
    
    # 获取x轴数据
    x = df[x_col].values
    x_pos = np.arange(len(x))
    
    # 绘制AUC柱状图
    auc_mean = df['test_auc_mean'].values
    color_auc = COLORS['auc']
    
    bars1 = ax1.bar(x_pos, auc_mean, 
                    width=BAR_STYLES['width'],
                    color=color_auc,
                    alpha=BAR_STYLES['alpha'],
                    edgecolor=BAR_STYLES['edgecolor'],
                    linewidth=BAR_STYLES['linewidth'],
                    label='AUC')
    
    if show_std and 'test_auc_std' in df.columns:
        auc_std = df['test_auc_std'].values
        ax1.errorbar(x_pos, auc_mean, yerr=auc_std, 
                     fmt='none', ecolor='black', 
                     capsize=5, capthick=2, alpha=0.7)
    
    ax1.set_ylabel('AUC (%)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax1.tick_params(labelsize=FONT_SIZES['tick'])
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')  # 只显示横向网格线
    
    # 在柱子上方显示数值
    for i, (bar, val) in enumerate(zip(bars1, auc_mean)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=FONT_SIZES['tick']-2)
    
    # 绘制LogLoss柱状图
    logloss_mean = df['test_logloss_mean'].values
    color_logloss = COLORS['logloss']
    
    bars2 = ax2.bar(x_pos, logloss_mean, 
                    width=BAR_STYLES['width'],
                    color=color_logloss,
                    alpha=BAR_STYLES['alpha'],
                    edgecolor=BAR_STYLES['edgecolor'],
                    linewidth=BAR_STYLES['linewidth'],
                    label='LogLoss')
    
    if show_std and 'test_logloss_std' in df.columns:
        logloss_std = df['test_logloss_std'].values
        ax2.errorbar(x_pos, logloss_mean, yerr=logloss_std, 
                     fmt='none', ecolor='black', 
                     capsize=5, capthick=2, alpha=0.7)
    
    # 在柱子上方显示数值
    for i, (bar, val) in enumerate(zip(bars2, logloss_mean)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=FONT_SIZES['tick']-2)
    
    # 设置x轴
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x, fontsize=FONT_SIZES['tick'])
    
    # 使用映射后的x轴标签
    x_label = X_LABEL_MAP.get(x_col, x_col)
    ax2.set_xlabel(x_label, fontsize=FONT_SIZES['label'], fontweight='bold')
    ax2.set_ylabel('LogLoss', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax2.tick_params(labelsize=FONT_SIZES['tick'])
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')  # 只显示横向网格线
    
    # 隐藏上下子图相邻的边框
    ax1.spines['bottom'].set_visible(False)  # 关闭子图1底部边框
    ax2.spines['top'].set_visible(False)     # 关闭子图2顶部边框
    
    # 移除上子图的x轴刻度
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
        save_figure_as_svg_and_pdf(fig, output_path)
    else:
        plt.show()
    
    plt.close()


def plot_feature_ablation_single(df, x_col, output_path=None, show_std=True):
    """
    绘制特征消除实验结果的单图（双y轴，AUC和LogLoss在同一图中）
    
    参数:
        df: DataFrame，包含数据
        x_col: str，x轴列名（如'#drop_features'）
        output_path: str，输出文件路径（可选）
        show_std: bool，是否显示误差棒
    """
    setup_matplotlib_style()
    
    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    
    # 获取x轴数据
    x = df[x_col].values
    x_pos = np.arange(len(x))
    width = 0.35
    
    # 获取数据
    auc_mean = df['test_auc_mean'].values
    logloss_mean = df['test_logloss_mean'].values
    
    # 绘制AUC柱状图（左y轴）
    bars1 = ax1.bar(x_pos - width/2, auc_mean, width,
                   label='AUC (%)', color=COLORS['auc'],
                   alpha=BAR_STYLES['alpha'],
                   edgecolor=BAR_STYLES['edgecolor'],
                   linewidth=BAR_STYLES['linewidth'])
    
    # 添加AUC误差棒
    if show_std and 'test_auc_std' in df.columns:
        auc_std = df['test_auc_std'].values
        ax1.errorbar(x_pos - width/2, auc_mean, yerr=auc_std, 
                     fmt='none', ecolor='black', 
                     capsize=5, capthick=2, alpha=0.7)
    
    # 在柱子上方显示AUC数值
    for i, (bar, val) in enumerate(zip(bars1, auc_mean)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (auc_std[i] if show_std and 'test_auc_std' in df.columns else 0),
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=FONT_SIZES['tick']-2, fontweight='bold')
    
    # 设置左y轴（AUC）
    ax1.set_ylabel('AUC (%)', fontsize=FONT_SIZES['label'], fontweight='bold', color=COLORS['auc'])
    ax1.tick_params(axis='y', labelcolor=COLORS['auc'], labelsize=FONT_SIZES['tick'])
    
    # 自动调整AUC的y轴范围，聚焦在数据变化范围
    auc_min, auc_max = auc_mean.min(), auc_mean.max()
    auc_range = auc_max - auc_min
    auc_margin = max(auc_range * 0.3, 0.1)  # 至少留10%边距
    ax1.set_ylim(auc_min - auc_margin, auc_max + auc_margin * 1.5)  # 上方多留空间给数值标签
    
    # 创建第二个y轴用于LogLoss
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + width/2, logloss_mean, width,
                    label='LogLoss', color=COLORS['logloss'],
                    alpha=BAR_STYLES['alpha'],
                    edgecolor=BAR_STYLES['edgecolor'],
                    linewidth=BAR_STYLES['linewidth'])
    
    # 添加LogLoss误差棒
    if show_std and 'test_logloss_std' in df.columns:
        logloss_std = df['test_logloss_std'].values
        ax2.errorbar(x_pos + width/2, logloss_mean, yerr=logloss_std, 
                     fmt='none', ecolor='black', 
                     capsize=5, capthick=2, alpha=0.7)
    
    # 在柱子上方显示LogLoss数值
    for i, (bar, val) in enumerate(zip(bars2, logloss_mean)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (logloss_std[i] if show_std and 'test_logloss_std' in df.columns else 0),
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=FONT_SIZES['tick']-2, fontweight='bold')
    
    # 设置右y轴（LogLoss）
    ax2.set_ylabel('LogLoss', fontsize=FONT_SIZES['label'], fontweight='bold', color=COLORS['logloss'])
    ax2.tick_params(axis='y', labelcolor=COLORS['logloss'], labelsize=FONT_SIZES['tick'])
    
    # 自动调整LogLoss的y轴范围，聚焦在数据变化范围
    logloss_min, logloss_max = logloss_mean.min(), logloss_mean.max()
    logloss_range = logloss_max - logloss_min
    logloss_margin = max(logloss_range * 0.3, 0.0002)  # 至少留0.02%边距
    ax2.set_ylim(logloss_min - logloss_margin, logloss_max + logloss_margin * 1.5)  # 上方多留空间给数值标签
    
    # 设置x轴
    x_label = X_LABEL_MAP.get(x_col, x_col)
    ax1.set_xlabel(x_label, fontsize=FONT_SIZES['label'], fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x, fontsize=FONT_SIZES['tick'])
    ax1.tick_params(axis='x', labelsize=FONT_SIZES['tick'])
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=FONT_SIZES['legend'], framealpha=0.9)
    
    # 只在左y轴显示网格线
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        save_figure_as_svg_and_pdf(fig, output_path)
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='绘制特征消除实验结果柱状图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法（默认使用单图双y轴，y轴范围自动调整）
  python plot_feature_ablation.py data.csv
  
  # 指定输出文件
  python plot_feature_ablation.py data.csv -o output.png
  
  # 使用上下分开的子图
  python plot_feature_ablation.py data.csv --subplot
  
  # 不显示误差棒
  python plot_feature_ablation.py data.csv --no-std
  
  # 指定x轴列名
  python plot_feature_ablation.py data.csv -x drop_features
        """
    )
    
    parser.add_argument('input_csv', type=str, help='输入CSV文件路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出图形文件路径（支持.png, .pdf, .svg等）')
    parser.add_argument('-x', '--x-column', type=str, default='#drop_features',
                        help='x轴列名（默认: #drop_features）')
    parser.add_argument('--subplot', action='store_true',
                        help='使用上下分开的子图而不是单图双y轴（默认使用单图）')
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
    
    if args.subplot:
        plot_feature_ablation_bars(df, args.x_column, output_path, show_std)
    else:
        plot_feature_ablation_single(df, args.x_column, output_path, show_std)
    
    print("完成！")


if __name__ == '__main__':
    main()

