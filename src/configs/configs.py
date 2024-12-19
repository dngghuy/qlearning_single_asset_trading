import os

# Paths
BASE_PATH = '/home/huy/Desktop/Thesis-code/FULL'
DATA_PATH = os.path.join(BASE_PATH, 'GOOGL.csv')
TRAIN_RF_PATH = os.path.join(BASE_PATH, 'GOOGL_RF.csv')
TRAIN_RL_PATH = os.path.join(BASE_PATH, 'GOOGL_RL.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'trendRF.pkl')

# General Parameters
TRAIN_TEST_RATIO = 0.8       # Split ratio for train-test datasets
RL_DATA_RATIO = 0.7          # Ratio for splitting RL-specific data
SEQ_LEN = 10                 # Sequence length for labeling
DATE_COLUMN = 'Date'         # Name of the Date column in the dataset
INITIAL_WEALTH = 5000        # Initial wealth for reinforcement learning

# Training Parameters
EPOCHS = 10                  # Number of epochs for model training
BATCH_SIZE = 32              # Batch size for RL agent training

# Random Forest Parameters
RF_RATIO = 0.8               # Ratio of data used for training RandomForest
RF_GRID_PARAMS = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_features': ['sqrt'],
    'max_depth': [2, 4, 8, 10],
    'criterion': ['gini', 'entropy']
}
N_SPLIT = 4                  # Number of splits for TimeSeries cross-validation

# Ensure directories exist
os.makedirs(BASE_PATH, exist_ok=True)

# Debugging Tip: Print paths to verify setup
if __name__ == "__main__":
    print("Base Path:", BASE_PATH)
    print("Data Path:", DATA_PATH)
    print("Train RF Path:", TRAIN_RF_PATH)
    print("Train RL Path:", TRAIN_RL_PATH)
    print("Model Path:", MODEL_PATH)