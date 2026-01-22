# FuxiCTR: A Comprehensive Library for Recommender System Models

FuxiCTR is a comprehensive and scalable library for developing and evaluating recommender system models, with a focus on Click-Through Rate (CTR) prediction. It provides a flexible framework for implementing a wide range of state-of-the-art models and running experiments on various datasets.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FuxiCTR.git
   cd FuxiCTR
   ```

2. Install the required dependencies. It is recommended to use a virtual environment (e.g., conda or venv).
   ```bash
   pip install -r requirements.txt 
   ```
   *Note: Please make sure you have `torch`, `pandas`, `pyyaml`, `scikit-learn`, and `gdown` installed.*

## Data Preparation

The processed datasets are available for download from Google Drive.

1. **Install `gdown` to download files from Google Drive:**
   ```bash
   pip install gdown
   ```

2. **Download the datasets:**
   You can download the datasets using the following commands. The datasets will be saved as zip files.

   *   **TaobaoAd:** ([Original Link](https://drive.google.com/file/d/1qstjIM8VzyTl7cNseb0udFs32754PK66/view?usp=sharing))
       ```bash
       gdown '1qstjIM8VzyTl7cNseb0udFs32754PK66' -O data/TaobaoAd.zip
       ```
   *   **Avazu:** ([Original Link](https://drive.google.com/file/d/1y1ZdZgobNRPIj6HosWZ-gnllbNKMEkN_/view?usp=sharing))
       ```bash
       gdown '1y1ZdZgobNRPIj6HosWZ-gnllbNKMEkN_' -O data/Avazu.zip
       ```
   *   **CriteoPrivateAd:** ([Original Link](https://drive.google.com/file/d/1wGhz0Vg6RVuIKoBp7cEJ3YvSluMGv5wB/view?usp=sharing))
       ```bash
       gdown '1wGhz0Vg6RVuIKoBp7cEJ3YvSluMGv5wB' -O data/CriteoPrivateAd.zip
       ```

3. **Unzip the datasets:**
   After downloading, unzip the files into the `data/` directory.

   ```bash
   unzip data/TaobaoAd.zip -d data/
   unzip data/Avazu.zip -d data/
   unzip data/CriteoPrivateAd.zip -d data/
   ```

   After unzipping, you should have the following directory structure:
   ```
   data/
   ├── TaobaoAd/
   ├── Avazu/
   └── CriteoPrivateAd/
   ```

## Running Experiments

Experiments are managed and run through the `experiment/run_expid.py` script. Each experiment is defined by a configuration file located in the `experiment/config` directory.

To run an experiment, you need to specify the experiment ID (`expid`) and the GPU device ID.

**Command template:**
```bash
python experiment/run_expid.py --expid <experiment_id> --gpu <gpu_id>
```

**Arguments:**
*   `--expid`: The ID of the experiment to run (e.g., `DeepFM_test`). This corresponds to a config file in `experiment/config/`.
*   `--gpu`: The GPU device index to use. Use -1 for CPU.
*   `--config` (optional): The directory containing your experiment config files. Defaults to `experiment/config/`.

**Example:**
To run the `DeepFM_test` experiment on GPU 0:
```bash
python experiment/run_expid.py --expid DeepFM_test --gpu 0
```
The script will load the corresponding configuration, preprocess the data if necessary, train the model, and evaluate its performance on the validation and test sets. Results will be logged to the console and saved to a CSV file.

## Guidelines for Reproduction

To facilitate the reproducibility of DTCN, in addition to providing the FuxiCTR-based code implementation, we hereby provide more details on our experimental setup. We run all our experiments on one NVIDIA Tesla A100 GPU to mitigate the impact from different hardware. The hyper-parameters reported within our paper are the optimal results obtained via an extensive grid search process. As the final step, we also fix the random seeds used for the training/validation/test of each dataset.

In this research, we experiment on CTR datasets including Avazu and TaobaoAd, and CVR datasets including CriteoPrivateAd and a private industry dataset. Intuitively, a key step to initiate our exploration on the prediction performance of personalized and non-personalized data is to distinguish personalized and non-personalized features. The industry dataset contains a binary variable named *is_personalization*. All features with a null value under is_personalization = 0 are recognized as personalized features, accounting for 30 out of 60 feature fields in the dataset. For the other 3 datasets, we list the feature fields used for experimentation in the table below. Fields in **bold** are marked as personalized features.

### Feature Fields for Public Datasets

| Dataset | Feature Fields                                                                                                                                                                                                                                                                      |
| :--- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Avazu | [**device_ip**, **device_id**, **device_model**, **device_type**, **device_conn_type**, banner_pos, site_id, site_domain, site_category, app_id, app_domain, app_category, hour, weekday, weekend, C1, C14, C15, C16, C17, C18, C19, C20, C21]                                      |
| TaobaoAd | [**cms_segid**, **cms_group_id**, **userid**, **final_gender_code**, **age_level**, **pvalue_level**, **shopping_level**, **occupation**, **new_user_class_level**, **cate_his**, **brand_his**, **btag_his**, adgroup_id, cate_id, campaign_id, customer, brand, pid, btag, price] |
| CriteoPrivateAd | [**user_id**, **features_kv_bits_constrained_0~24**, **features_browser_bits_constrained_0~6**, campaign_id, display_order, publisher_id, features_kv_not_constrained_1\~7, features_ctx_not_constrained_0\~7]                                                                      |

To ensure reproducibility, we also provide the detailed hyperparameter configurations for DTCN across different datasets in the table below. The configurations include the backbone model architecture, distance loss weight ($\beta$), embedding regularizer, and model-specific parameters for both personalized and non-personalized towers. These parameters were determined through extensive grid search and represent the optimal configurations that achieve the best performance on each dataset.

### Hyperparameter Configurations

| Dataset | Backbone | $\beta$ | Emb. Reg. | Model-Specific Parameters |
| :--- | :--- | :--- | :--- | :--- |
| Avazu | FINAL | 40 | 1e-05 | embedding_dim=32, <br> **$FI_A$-PER-PER: FINAL**, block_type=2B, block1_hidden_units=[800], block1_dropout=0.2, block2_hidden_units=[800,800], block2_dropout=0.3 <br> **$FI_A$-Non PER: FINAL**, block_type=1B, block1_hidden_units=[800], block1_dropout=0.2 |
| TaobaoAd | PNN | 150 | 0.05 | embedding_dim=16, <br> **$FI_A$-PER-PER: PNN**, hidden_units=[512, 256] <br> **$FI_A$-Non PER: PNN**, hidden_units=[512, 256] |
| TaobaoAd | FCN | 150 | 0.05 | embedding_dim=32, <br> **$FI_A$-PER-PER: FCN**, num_heads=1, num_deep_cross_layers=4, num_shallow_cross_layers=4 <br> **$FI_A$-Non PER-Non PER: FCN**, num_heads=1, num_deep_cross_layers=4, num_shallow_cross_layers=4 |
| CriteoPrivateAd | PNN | 80 | 1e-05 | embedding_dim=16, <br> **$FI_A$-PER-PER: PNN**, hidden_units=[1000, 1000] <br> **$FI_A$-Non PER-Non PER: PNN**, hidden_units=[1000, 1000] |
