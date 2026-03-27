import os
import random
import numpy as np
from pathlib import Path

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# XGBoost estimator settings: final model vs CV speed-tradeoff
XGB_FINAL_ESTIMATORS = 1000
XGB_CV_ESTIMATORS = 700

# Data directories
DEFAULT_DATA_DIR = Path.cwd() / "dataset"
FALLBACK_DATA_DIR = Path("/Users/thilak/PythonFiles/Sem 8/IOT/PAPER/dataset")
DATA_DIR = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else FALLBACK_DATA_DIR
