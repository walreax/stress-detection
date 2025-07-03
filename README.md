# Stress Detection with WESAD Dataset

This project provides tools and scripts for stress detection using the WESAD physiological dataset. It includes data extraction, exploratory data analysis (EDA), feature engineering, and a simple neural network for binary stress classification.

## Project Structure

- `train_stress_nn.py` — Train a neural network to classify stress/non-stress from WESAD signals.
- `eda_wesad.py` — Generate an EDA summary table of the dataset.
- `get_dataset.py` — Download the WESAD dataset from Kaggle.
- `stress.py` — Inspect and print the structure of WESAD data files.
- `test.py` — Test Python and ML library imports.
- `requirements.txt` / `environment.yml` — List of dependencies.
- `WESAD/` — (Ignored) Directory containing the dataset files (not tracked in git).

## Setup

1. **Install dependencies**
   - Using pip:
     ```sh
     pip install -r requirements.txt
     ```
   - Or with conda:
     ```sh
     conda env create -f environment.yml
     conda activate <your_env_name>
     ```

2. **Download the dataset**
   - Run:
     ```sh
     python get_dataset.py
     ```
   - Place the `WESAD/` folder in the project root if not done automatically.

## Usage

- **Train the neural network:**
  ```sh
  python train_stress_nn.py
  ```
- **Run EDA and generate summary:**
  ```sh
  python eda_wesad.py
  ```
- **Inspect dataset structure:**
  ```sh
  python stress.py
  ```

## Notes
- The `WESAD/` dataset directory is ignored by git for privacy and size reasons.
- Output files like `wesad_eda_table.md` and `wesad_eda_table.csv` are also ignored.

## References
- [WESAD Dataset on UCI](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
- [Original Paper](https://ieeexplore.ieee.org/document/8269806)

---

Feel free to contribute or open issues for improvements!
