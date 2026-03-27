# IoT Cooling Load Predictor

This repository contains a structured, modular machine learning pipeline designed to predict building cooling loads based on IoT meter readings and weather conditions. This codebase processes time-series building energy dataset, models the load using baseline regressors, linear models, XGBoost, and Random Forest, performs Cross-Validation (CV), and subsequently generates tables, metrics, and visualization artifacts for paper publication.

## Repository Structure

- `config.py`: Centralized configuration constants including static seed generation, xgboost hyper-parameters, and standardized data load directories.
- `data_processing.py`: Preprocessing utility logic to load CSV files correctly, engineer complex lag/time cyclic features reliably, and split train-test datasets.
- `utils.py`: Metric evaluation scripts implementing proper predictions conversion techniques globally and generating DataFrames reliably.
- `train_evaluate.py`: The fundamental definitions for training predictive variables (Baseline, Linear Regression, XGBoost, Random Forest, Time-Series Models). It contains unified cross-validation utilities to validate dataset stability properly.
- `main.py`: The central execution entry point linking modules sequentially. It yields logging, handles evaluation integrity verification, and saves validation graphics smoothly.
- `dataset/`: Target dataset directory (expects `train.csv`, `weather_train.csv`, `building_metadata.csv`).
- `paper_outputs/`: Directory where structured `.csv`, dynamically styled `.html`, and graphical `.png` reports are saved.

## Installation and Requirements

1. **Python 3.7+** is recommended.
2. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost jinja2
   ```

## Dataset Configuration

Place the following files inside a `dataset/` directory:
- `train.csv`
- `weather_train.csv`
- `building_metadata.csv`

*(Note: The `config.py` allows fallback hardcoded directories, but adopting relative local `dataset/` folder formats promotes optimal execution.)*

## Usage

Simply run `main.py` using Python 3:

```bash
python main.py
```

### Outputs Execution
- Console metric reports displaying performance across various statistical models smoothly.
- Time-series naive assessments efficiently processed.
- Results and graphical elements are permanently committed to `paper_outputs/` natively.
