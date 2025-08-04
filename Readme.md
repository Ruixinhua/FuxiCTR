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
```
