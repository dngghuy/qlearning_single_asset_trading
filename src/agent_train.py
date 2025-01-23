import os

import torch

import src.q_agent.utils as us
from src.configs.configs import (
    CLOSE_COL,
    TRAIN_RL_PATH,
    RF_MODEL_PATH,
    DATE,
    TICKER,
    LR,
    REPLAY_MEMORY_SIZE,
    MODEL_PATH
)
from src.q_agent.agent import TradingAgent, QNetwork
from src.q_agent.train_agent import ReplayMemory, EnvironmentSingleStock, Trainer
from src.regime_detection.data_processing import (
    WrapMOM,
    WrapROC,
    WrapWILLR,
    WrapMFI,
    WrapDELTA,
    WrapMACD,
    WrapRSI, DataFeatureEngineer
)
from src.regime_detection.rf_classifier import load_trend_detector


def train():
    data = us.get_stock(TRAIN_RL_PATH)
    rf_clf = load_trend_detector(RF_MODEL_PATH)

    # preprocess data
    data.columns = [dat.lower() for dat in data.columns]
    data.set_index(data[DATE], inplace=True)
    data = data.sort_index()
    data_ticker = data[TICKER]
    data.drop([DATE, TICKER], axis=1, inplace=True)

    transformation_list = [
        WrapMACD(fastperiod=12, slowperiod=26, signalperiod=9),
        WrapRSI(timeperiod=10),
        WrapMOM(timeperiod=12),
        WrapROC(timeperiod=10),
        WrapWILLR(timeperiod=14),
        WrapMFI(timeperiod=14),
        WrapDELTA(timeperiod=10),
    ]

    feature_engineer = DataFeatureEngineer(transformation_list)
    data = feature_engineer.process(data)
    data[TICKER] = data_ticker
    data = data.dropna()

    data = data[feature_engineer.feature_list + [CLOSE_COL, TICKER]]

    # Init Q net
    model = QNetwork(input_dim=10, action_dim=3)
    trading_agent = TradingAgent(model=model, train=True)

    optimizer = torch.optim.Adam(trading_agent.model.parameters(), lr=LR)
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    environment = EnvironmentSingleStock(
        data=data,
        feature_list=feature_engineer.feature_list,
        rf_clf=rf_clf,
        initial_wealth=5000
    )
    trading_agent.model = model
    RL_MODEL_PATH = os.path.join(MODEL_PATH, 'trendRL7.pkl')
    trainer = Trainer(
        agent=trading_agent,
        environment=environment,
        replay_memory=replay_memory,
        optimizer=optimizer,
        model_path=RL_MODEL_PATH,
    )

    trainer.train(num_episodes=500)
    us.plot_metrics(trainer, filename=os.path.join(MODEL_PATH, 'train_results.png'))


if __name__ == "__main__":
    train()