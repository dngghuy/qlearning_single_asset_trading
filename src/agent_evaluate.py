import os

import matplotlib.pyplot as plt
import torch

import src.q_agent.utils as us
from src.configs.configs import RF_MODEL_PATH, DATE, TICKER, \
    MODEL_PATH, TEST_RL_PATH
from src.q_agent.agent import TradingAgent, QNetwork
from src.q_agent.train_agent import EnvironmentSingleStock
from src.regime_detection.data_processing import *
from src.regime_detection.rf_classifier import load_trend_detector


class Evaluator:
    """
    Handles the evaluation loop for the reinforcement learning agent.
    """

    def __init__(self, agent, environment):
        """
        Initializes the evaluator.

        Args:
            agent: The trading agent.
            environment: The trading environment.
        """
        self.agent = agent
        self.environment = environment
        self.action_counts = {
            "hold": [],
            "buy": [],
            "sell": [],
            "failed_buy": [],
            "failed_sell": [],
        }
        self.logged_actions = []  # Store actions for visualization
        self.profit_history = []

        # Load the trained model
        self.agent.model.eval()

    def act(self, num_episodes=1):
        """
        Evaluates the agent for a specified number of episodes.

        Args:
            num_episodes: The number of episodes to evaluate for.
        """
        for e in range(num_episodes):
            state = self.environment.reset()
            done = False
            total_reward = 0
            action_counts = {
                "hold": 0,
                "buy": 0,
                "sell": 0,
                "failed_buy": 0,
                "failed_sell": 0,
            }

            while not done:
                state_tensor = torch.tensor(
                    np.array(state), dtype=torch.float32
                ).unsqueeze(
                    0
                )  # Add batch dimension
                action = (
                    self.agent.make_action(state)
                )  # epsilon=0 means no exploration
                next_state, reward, done, info = self.environment.step(action)

                self.logged_actions.append(
                    {
                        "episode": e,
                        "step": self.environment.current_step,
                        "action": action,
                        "info": info,
                    }
                )

                state = next_state
                total_reward += reward

                if action == 0:
                    action_counts["hold"] += 1
                elif action == 1:
                    if "action_successful" in info and not info["action_successful"]:
                        action_counts["failed_buy"] += 1
                    else:
                        action_counts["buy"] += 1
                elif action == 2:
                    if "action_successful" in info and not info["action_successful"]:
                        action_counts["failed_sell"] += 1
                    else:
                        action_counts["sell"] += 1

            for action_type, count in action_counts.items():
                self.action_counts[action_type].append(count)

            if "final_wealth" in info:
                profit = info.get("final_wealth", 0) - self.environment.initial_wealth
            else:
                profit = 0
            self.profit_history.append(profit)
            print(
                f"Episode {e}: Total Reward: {total_reward:.2f}, Profit: {profit:.2f}"
            )
            print(
                f"  Action Counts - Hold: {action_counts['hold']}, Buy: {action_counts['buy']}, Sell: {action_counts['sell']}, Failed Buy: {action_counts['failed_buy']}, Failed Sell: {action_counts['failed_sell']}"
            )

        print(
            f"Total profit across {num_episodes} episode(s): {sum(self.profit_history)}"
        )

    def visualize_trades(self, episode_to_plot=0):
        """
        Visualizes the buy and sell actions on the price chart for a given episode.

        Args:
        episode_to_plot: The episode number to plot.
        """

        # Extract prices from data
        prices = self.environment.data[CLOSE_COL].values

        # Filter actions for the specified episode
        episode_actions = [
            action
            for action in self.logged_actions
            if action["episode"] == episode_to_plot
        ]

        # Get buy and sell points
        buy_points = [
            (action["step"], prices[action["step"] - 1])
            for action in episode_actions
            if (action["action"] == 1 and 'action_successful' not in action['info'])
        ]
        sell_points = [
            (action["step"], prices[action["step"] - 1])
            for action in episode_actions
            if (action["action"] == 2 and 'action_successful' not in action['info'])
        ]

        plt.figure(figsize=(15, 8))
        plt.plot(prices, label="Price", color="blue")

        if buy_points:
            buy_x, buy_y = zip(*buy_points)
            plt.scatter(buy_x, buy_y, label="Buy", color="green", marker="^", s=100)

        if sell_points:
            sell_x, sell_y = zip(*sell_points)
            plt.scatter(sell_x, sell_y, label="Sell", color="red", marker="v", s=100)

        plt.xlabel("Step")
        plt.ylabel("Price")
        plt.title(f"Trades in Episode {episode_to_plot}")
        plt.legend()
        plt.savefig(os.path.join(MODEL_PATH, 'evaluate_results.png'))
        # plt.show()


def main():
    data = us.get_stock(TEST_RL_PATH)
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
    data = data.loc[data.index > '2015-08-04']

    model = QNetwork(input_dim=10, action_dim=3)
    state_dict = torch.load(os.path.join(MODEL_PATH, 'trendRL6.pkl'))
    state_dict = us.remove_prefix(state_dict, 'model')
    model.model.load_state_dict(state_dict)

    environment = EnvironmentSingleStock(
        data=data,
        feature_list=feature_engineer.feature_list,
        rf_clf=rf_clf,
        initial_wealth=5000
    )
    trading_agent = TradingAgent(model=model, train=False)
    evaluator = Evaluator(agent=trading_agent, environment=environment)
    evaluator.act()
    evaluator.visualize_trades()


if __name__ == '__main__':
    main()