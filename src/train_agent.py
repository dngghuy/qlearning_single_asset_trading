from agent import TradingAgent
import fin_utils
import clf_utils
import config
import pickle as pkl
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt


def train_epoch(agent, data, clf, epoch, length):
    """
    Train the agent for one epoch and return metrics.
    """
    wealth = config.INITIAL_WEALTH
    current_wealth = wealth
    portfolio, tickers = fin_utils.initialize_portfolio(data)
    ticker = tickers[0]  # Assume single stock for now

    state = fin_utils.get_current_state(data, wealth, portfolio, clf, 37)
    total_profit = 0
    reward = 0
    penalty = 6
    penalty_same_action = 10
    action_past = None

    agent.inventory = []

    for t in range(37, length):
        # Select action
        action = agent.act(state)
        if action_past is None:
            action = 1

        # Buy Action
        if action == 1 and action != action_past:
            if wealth >= data.iloc[t].close:
                num_shares = int(wealth / data.iloc[t].close)
                portfolio[ticker] += num_shares
                wealth -= num_shares * data.iloc[t].close
                agent.inventory.append(data.iloc[t])
                reward -= penalty
                print(f"BUY {ticker} at {data.iloc[t].close}, wealth: {wealth}")
                action_past = action

        # Sell Action
        elif action == 2 and action != action_past:
            if len(agent.inventory) > 0:
                past_data = agent.inventory.pop(0)
                profit = (data.iloc[t].close - past_data.close) * portfolio[ticker]
                reward += profit - penalty
                total_profit += profit
                wealth += data.iloc[t].close * portfolio[ticker]
                portfolio[ticker] = 0
                print(f"SELL {ticker} at {data.iloc[t].close}, profit: {profit}, wealth: {wealth}")
                action_past = action

        # Hold Action
        elif action == 0 and action == action_past:
            reward -= penalty_same_action * 0.9
        else:
            reward -= penalty_same_action
            action_past = action

        # Update wealth and state
        current_wealth = wealth + portfolio[ticker] * data.iloc[t].close
        done = t == (length - 1)
        next_state = fin_utils.get_current_state(data, current_wealth, portfolio, clf, t + 1)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        # Finalize at end of epoch
        if done:
            final_wealth = portfolio[ticker] * data.iloc[t].close + wealth
            print(f"Epoch {epoch} completed. Final wealth: {final_wealth}, Profit: {total_profit}")
            return total_profit, reward


def main():
    num_epochs = 30

    # Load or create models
    rf_clf = clf_utils.preprocess_and_create_clf()
    agent = TradingAgent(input_dim=8, action_dim=3, hidden_layers=[64, 32], train=True)

    # Load and prepare data
    data = fin_utils.get_single_stock(config.TRAIN_RL_PATH)
    length = len(data) - 1

    # Training loop
    total_profit_per_epoch = []
    total_reward_per_epoch = []

    for epoch in range(1, num_epochs + 1):
        print(f"Starting Epoch {epoch}")
        profit, reward = train_epoch(agent, data, rf_clf, epoch, length)
        total_profit_per_epoch.append(profit)
        total_reward_per_epoch.append(reward)

        # Train agent on replay buffer
        agent.replay(batch_size=32)

        # Save progress
        if epoch % 3 == 0:
            torch.save(agent.model.state_dict(), f"models/model_epoch_{epoch}.pt")
            with open(f"results/profit_epoch_{epoch}.pkl", 'wb') as f:
                pkl.dump(total_profit_per_epoch, f)
            with open(f"results/reward_epoch_{epoch}.pkl", 'wb') as f:
                pkl.dump(total_reward_per_epoch, f)

    # Save final metrics
    with open('results/final_profit.pkl', 'wb') as f:
        pkl.dump(total_profit_per_epoch, f)
    with open('results/final_reward.pkl', 'wb') as f:
        pkl.dump(total_reward_per_epoch, f)

    print("Training completed.")


if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    main()
