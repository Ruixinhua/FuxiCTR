# Model Performance Comparison

The table below shows the best performance of each model on different datasets. For each model, we conducted extensive hyperparameter tuning experiments and selected the configuration with the best performance. Bold values indicate the best results for each metric across all models. Arrows (↑/↓) indicate whether higher or lower values are better.

<table>
  <tr>
    <th>Model</th>
    <th colspan="2" align="center">criteo_x1</th>
    <th colspan="2" align="center">criteo_x4</th>
    <th colspan="2" align="center">avazu_x1</th>
    <th colspan="2" align="center">avazu_x4</th>
    <th colspan="3" align="center">taobaoad_x1</th>
  </tr>
  <tr>
    <th>Metrics</th>
    <th>AUC↑</th>
    <th>logloss↓</th>
    <th>AUC↑</th>
    <th>logloss↓</th>
    <th>AUC↑</th>
    <th>logloss↓</th>
    <th>AUC↑</th>
    <th>logloss↓</th>
    <th>AUC↑</th>
    <th>logloss↓</th>
    <th>gAUC↑</th>
  </tr>
  <tr>
    <td>DCNv2</td>
    <td>0.814065</td>
    <td>0.437973</td>
    <td>0.814648</td>
    <td>0.437595</td>
    <td>0.764602</td>
    <td>0.366846</td>
    <td>0.793271</td>
    <td>0.372105</td>
    <td><b>0.649314</b></td>
    <td>0.192884</td>
    <td>0.574243</td>
  </tr>
  <tr>
    <td>DCNv3</td>
    <td><b>0.815935</b></td>
    <td><b>0.436042</b></td>
    <td><b>0.816022</b></td>
    <td><b>0.436022</b></td>
    <td>0.763836</td>
    <td>0.366750</td>
    <td><b>0.797461</b></td>
    <td><b>0.369204</b></td>
    <td>0.648585</td>
    <td>0.194355</td>
    <td>0.572273</td>
  </tr>
  <tr>
    <td>FinalMLP</td>
    <td>0.814950</td>
    <td>0.436936</td>
    <td>0.814678</td>
    <td>0.437285</td>
    <td>0.766319</td>
    <td>0.365867</td>
    <td>0.792807</td>
    <td>0.372292</td>
    <td>0.645906</td>
    <td>0.194208</td>
    <td><b>0.574752</b></td>
  </tr>
  <tr>
    <td>FinalNet</td>
    <td>0.815243</td>
    <td>0.436716</td>
    <td>0.815397</td>
    <td>0.436639</td>
    <td><b>0.766980</b></td>
    <td><b>0.365594</b></td>
    <td>0.793642</td>
    <td>0.371790</td>
    <td>0.648157</td>
    <td>0.193585</td>
    <td>0.572010</td>
  </tr>
  <tr>
    <td>MaskNet</td>
    <td>0.814009</td>
    <td>0.437861</td>
    <td>0.813405</td>
    <td>0.442204</td>
    <td>0.762942</td>
    <td>0.368110</td>
    <td>0.794315</td>
    <td>0.371351</td>
    <td>0.646183</td>
    <td><b>0.192420</b></td>
    <td>0.569544</td>
  </tr>
</table>

## Dataset Information

### Criteo_x1

The Criteo dataset is a widely-used benchmark dataset for CTR prediction, containing about one week of click-through data for display advertising. It has:
- 13 numerical feature fields and 26 categorical feature fields
- 45,840,617 total samples split into 33,003,326 training, 8,250,124 validation, and 4,587,167 test samples (7:2:1 split)
- Source: Kaggle Criteo Display Advertising Challenge

### Criteo_x4

Same as Criteo_x1 but with a different data split approach:
- 45,840,617 total samples split into 36,672,493 training, 4,584,062 validation, and 4,584,062 test samples (8:1:1 split)
- Preprocessing follows the winner's solution of the Criteo challenge to discretize each integer value
- Infrequent categorical features are replaced with a default <OOV> token
- Used by AutoInt and other CTR prediction models

### Avazu_x1

This dataset contains about 10 days of labeled click-through data on mobile advertisements:
- 22 feature fields including user features and advertisement attributes
- 40,428,967 total samples split into 28,300,276 training, 4,042,897 validation, and 8,085,794 test samples (7:1:2 split)
- Source: Kaggle Avazu CTR Prediction Challenge

### Avazu_x4

Same dataset as Avazu_x1 but with a different splitting strategy:
- 40,428,967 total samples split into 32,343,172 training, 4,042,897 validation, and 4,042,898 test samples (8:1:1 split)
- Preprocessing includes removing the 'id' field and transforming timestamp into hour, weekday, and is_weekend fields
- Used in the AutoInt and FGCNN research papers

### TaobaoAd_x1

Taobao Ad dataset provided by Alibaba:
- Contains 8 days of ad click-through data (26 million records) randomly sampled from 1,140,000 users
- First 7 days (20170506-20170512) used as training and the last day (20170513) as test data
- Also includes shopping behavior of all users in the recent 22 days (700 million records)
- Features user behavior sequences with maximal length of 50
- 25,029,426 total samples split into 21,929,911 training and 3,099,515 test samples
- Unique features include group AUC (gAUC) evaluation for multi-behavior prediction

All datasets are widely used benchmarks in CTR prediction research.
