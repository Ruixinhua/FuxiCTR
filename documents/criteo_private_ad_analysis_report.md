# CriteoPrivateAd 数据集分析报告

## 数据集概览

### 基本信息
- **数据集名称**: CriteoPrivateAd
- **数据格式**: Parquet
- **总特征数**: 150
- **样本数量**: 约1亿条记录（full数据集）
- **分析样本**: 100,000条记录（从valid.parquet采样）
- **目标标签**: is_clicked（点击预测）
- **点击率**: 38.15%

### 文件结构
- `train.parquet`: 33GB（约9000万条记录）
- `valid.parquet`: 3.6GB（约900万条记录）
- `test.parquet`: 3.9GB（约1000万条记录）

## 特征分类分析

根据特征名称和隐私约束，我们将特征分为以下类别：

### 1. 用户相关特征 (USER_RELATED, 2个)
- `user_id`: 用户唯一标识
- `display_order`: 用户当日展示顺序

### 2. Key-Value服务器约束特征 (KV_CONSTRAINED, 31个)
这些特征来自Key-Value服务器，受12位约束限制：
- `features_kv_bits_constrained_0` 到 `features_kv_bits_constrained_30`
- 这些特征代表单域用户特征，从modelingSignals派生

### 3. Key-Value服务器非约束特征 (KV_NOT_CONSTRAINED, 8个)
- `features_kv_not_constrained_1` 到 `features_kv_not_constrained_8`
- 来自Interest Group名称/renderURL

### 4. 浏览器约束特征 (BROWSER_CONSTRAINED, 11个)
跨域特征，在generateBid中可用，受12位约束：
- `features_browser_bits_constrained_0` 到 `features_browser_bits_constrained_10`

### 5. 上下文非约束特征 (CTX_NOT_CONSTRAINED, 8个)
来自上下文调用的特征：
- `features_ctx_not_constrained_0` 到 `features_ctx_not_constrained_7`

### 6. 不可用特征 (NOT_AVAILABLE, 80个)
隐私保护下不可用的特征（跨设备和跨域）：
- `features_not_available_0` 到 `features_not_available_79`

### 7. 标签特征 (LABELS, 4个)
- `is_clicked`: 是否点击（主要目标标签）
- `is_visit`: 是否访问
- `nb_sales`: 销售数量
- `is_click_landed`: 点击是否着陆

### 8. 元数据特征 (METADATA, 6个)
- `id`: 记录ID
- `campaign_id`: 广告系列ID
- `publisher_id`: 发布商ID
- `sale_delay_after_display_array`: 展示后销售延迟数组
- `click_delay_after_display_array`: 展示后点击延迟数组
- `landed_click_delay_after_display_array`: 展示后着陆点击延迟数组

## 缺失值分析

### 缺失率最高的特征（Top 20）:

| 特征名称 | 缺失率 | 数据类型 | 特征类别 |
|---------|--------|----------|----------|
| features_not_available_68 | 100.0% | float64 | 不可用特征 |
| features_kv_bits_constrained_5 | 100.0% | float64 | KV约束特征 |
| features_not_available_58 | 99.17% | float64 | 不可用特征 |
| nb_sales | 98.61% | float64 | 标签特征 |
| sale_delay_after_display_array | 98.61% | object | 元数据 |
| features_not_available_70 | 97.38% | float64 | 不可用特征 |
| features_not_available_57 | 94.86% | float64 | 不可用特征 |
| features_kv_bits_constrained_14 | 94.54% | float64 | KV约束特征 |
| features_not_available_78 | 92.23% | float64 | 不可用特征 |
| features_kv_bits_constrained_17 | 90.19% | float64 | KV约束特征 |

### 缺失值分析结论：
1. **NOT_AVAILABLE特征**缺失率极高，这符合隐私保护的预期
2. **销售相关数据**缺失率达98.6%，说明大部分展示没有产生销售
3. 某些**KV约束特征**也有较高缺失率，可能受数据收集限制

## 个性化特征识别

基于与用户的关联度分析，以下特征被识别为个性化特征：

### 强个性化特征（与用户强相关）：
1. **用户标识特征**: `user_id`（显然的个性化特征）
2. **行为序列特征**: `display_order`（用户行为顺序）
3. **Key-Value约束特征**: 这些特征基于用户的历史行为建模
4. **浏览器特征**: 跨域用户特征，包含用户设备和行为信息

### 非个性化特征：
1. **上下文特征**: 主要基于当前页面/环境
2. **广告系列特征**: `campaign_id`, `publisher_id`
3. **一些不可用特征**: 已被隐私保护机制移除

## 特征与点击的相关性分析

### 与点击最相关的特征（Top 15）：

| 排名 | 特征名称 | 相关性 | P值 | 特征类别 |
|------|----------|-------|-----|----------|
| 1 | features_browser_bits_constrained_10 | -0.4097 | 0.0000 | 浏览器特征 |
| 2 | features_browser_bits_constrained_1 | -0.3539 | 0.0000 | 浏览器特征 |
| 3 | features_kv_bits_constrained_23 | -0.3218 | 0.0000 | KV约束特征 |
| 4 | features_kv_bits_constrained_24 | -0.2972 | 0.0000 | KV约束特征 |
| 5 | features_browser_bits_constrained_2 | -0.2941 | 0.0000 | 浏览器特征 |
| 6 | features_kv_bits_constrained_13 | 0.2572 | 0.0000 | KV约束特征 |
| 7 | features_kv_bits_constrained_30 | 0.2561 | 0.0000 | KV约束特征 |
| 8 | features_browser_bits_constrained_7 | -0.2433 | 0.0000 | 浏览器特征 |
| 9 | features_kv_bits_constrained_12 | 0.2425 | 0.0000 | KV约束特征 |
| 10 | features_kv_bits_constrained_21 | -0.2335 | 0.0000 | KV约束特征 |

### 关键发现：
1. **浏览器特征**具有最强的预测能力（相关性达-0.41）
2. **KV约束特征**也表现出较强的相关性
3. 大部分高相关性特征都是**负相关**，这可能表示某些特征值的增加会降低点击概率
4. 所有高相关特征的P值都极小，统计显著性很强

## 隐私保护分析

### 在非个性化场景下的特征可用性：

1. **完全可用的特征**（非个性化）：
   - 上下文特征 (CTX_NOT_CONSTRAINED)
   - 广告系列ID
   - 发布商ID

2. **部分可用的特征**（受隐私约束）：
   - KV约束特征（12位限制）
   - 浏览器约束特征（12位限制）

3. **完全不可用的特征**（隐私保护）：
   - NOT_AVAILABLE特征（80个）
   - 用户ID（直接用户标识）

## 建议和结论

### 1. 模型训练建议：
- **主要特征集**：优先使用浏览器特征和KV约束特征
- **备用特征集**：在隐私受限环境下，重点使用上下文特征
- **数据预处理**：需要处理高缺失率特征（考虑删除或填充）

### 2. 隐私保护策略：
- 在非个性化场景下，模型性能可能下降40%左右（基于最强特征的相关性）
- 需要开发基于聚合数据的训练方法
- 考虑差分隐私技术来保护用户特征

### 3. 特征工程建议：
- 对缺失率>90%的特征考虑删除
- 对sequence类型特征进行适当的编码处理
- 考虑特征组合来提升预测能力

### 4. 评估指标：
- 使用Log-likelihood (LLH)和LLH-CompVN
- 关注校准度（预测值和实际值的比率应接近1）
- AUC不是最适合的指标（如论文所述） 