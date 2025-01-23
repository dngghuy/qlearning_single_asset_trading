import os
from pathlib import Path

ROOT_FILE = Path(__file__).parent.parent.parent.resolve()
DATA_PATH = os.path.join(ROOT_FILE, 'data')
MODEL_PATH = os.path.join(ROOT_FILE, 'models')
RESULT_PATH = os.path.join(ROOT_FILE, 'results')


# Paths
TRAIN_RF_PATH = os.path.join(DATA_PATH, 'GOOGL_RF.csv')
TRAIN_RL_PATH = os.path.join(DATA_PATH, 'VNM_train.csv')
TEST_RL_PATH = os.path.join(DATA_PATH, 'VNM_test.csv')
RF_MODEL_PATH = os.path.join(MODEL_PATH, 'trendRF.pkl')
FEATURE_ENGINEER_PATH = os.path.join(MODEL_PATH, 'feature_engineer.pkl')
RL_MODEL_PATH = os.path.join(MODEL_PATH, 'trendRL.pkl')

# General Parameters
TRAIN_TEST_RATIO = 0.8
RL_DATA_RATIO = 0.7
WINDOW_LENGTH = 10
DATE_COLUMN = 'Date'
INITIAL_WEALTH = 5000
REPLAY_MEMORY_SIZE = 100000
LR = 1e-3
TRAINING_FREQUENCY = 100

# Training Parameters
EPOCHS = 50
BATCH_SIZE = 512

# Random Forest Parameters
RATIO = 0.8
RF_GRID_PARAMS = {
    'n_estimators': [250, 500, 1000],
    'max_features': ['sqrt'],
    'max_depth': [2, 4, 8],
    'criterion': ['gini', 'entropy']
}
N_SPLIT = 4

# More columns
SMA_COL = 'sma'
CLOSE_COL = 'close'
HIGH_COL = 'high'
LOW_COL = 'low'
TICKER_COL = 'ticker'
MACD_COL = 'macd'
MACD_SIGNAL_COL = 'macd_signal'
MACD_HIST_COL = 'macd_hist'
RSI_COL = 'rsi'
MOM_COL = 'mom'
ROC_COL = 'roc'
WILLR_COL = 'willr'
VOLUME_COL = 'volume'
DELTA_VOLUME_COL = 'delta_volume'
MFI_COL = 'mfi'
DATE = 'date'
TICKER = 'ticker'
# Ensure directories exist
# os.makedirs(BASE_PATH, exist_ok=True)