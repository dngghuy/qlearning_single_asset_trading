import sys
import time

import torch

import src.q_agent.utils as us  # Assuming this module exists
from agent import TradingAgent, QNetwork
from src.configs.configs import TICKER_COL, CLOSE_COL, TRAIN_RL_PATH, RL_MODEL_PATH
from src.regime_detection.rf_classifier import load_trend_detector

if __name__ == '__main__':

    # filename, epochs = sys.argv[1], int(sys.argv[2])
    # path = '/home/huy/Desktop/Thesis-code/trend-predict/data/' + filename  # Update path if needed
    model = QNetwork(input_dim=18, action_dim=3)
    trading_agent = TradingAgent(model=model, train=True)

    data = us.get_stock(TRAIN_RL_PATH)
    length = len(data) - 1
    batch_size = 32
    rf_clf = load_trend_detector()
    epochs = 10

    for e in range(epochs + 1):
        wealth = 5000.0
        portfolio, tickers = us.initialize_portfolio(data, wealth)
        state = us.get_current_state(data, wealth, portfolio, 0)
        print(f'Initial state: we have {wealth} in VND and our port is {portfolio}')

        total_profit = 0
        trading_agent.inventory = []
        total_val = [wealth]

        for t in range(length):
            # if t % 100 == 0:
            #     print('------------------------------------------------------')
            #     print(f'At time step {t} in epoch {e}:\n', reward)
            #     # print(f'Current wealth {wealth + portfolio[tickers] * close}')
            #     print('------------------------------------------------------')
            #     print('\n')
            #     time.sleep(0.9)

            action = trading_agent.make_action(state)
            reward = 0
            temp = data.iloc[t]
            ticker = tickers[0]
            if action == 1:
                if portfolio['cash'][t] >= data.iloc[t][CLOSE_COL]:
                    trading_agent.inventory.append(temp)
                    num_shares = int(portfolio['cash'][t] / temp[CLOSE_COL])
                    portfolio['cash'].append(portfolio['cash'][t] - num_shares * temp[CLOSE_COL])
                    portfolio[ticker].append(portfolio.get(ticker)[t] + num_shares)
                    # print(f'All-in {temp[TICKER_COL]} at {temp[CLOSE_COL]}:\n')
                    time.sleep(0.9)
                else:
                    # print('wrong buy move')
                    reward -= portfolio['cash'][t] * 0.5
                    portfolio['cash'].append(portfolio['cash'][t])
                    portfolio[ticker].append(portfolio[ticker][t])
            elif action == 2:
                if len(trading_agent.inventory) > 0:
                    data_past = trading_agent.inventory.pop(0)
                    reward = temp[CLOSE_COL] - data_past[CLOSE_COL]
                    profit = reward * portfolio[ticker][t]
                    current_cash = portfolio['cash'][t] + temp[CLOSE_COL] * portfolio[ticker][t]
                    portfolio['cash'].append(current_cash)
                    portfolio[ticker].append(0)

                    # print(f'All-out {temp[TICKER_COL]} at {temp[CLOSE_COL]}:\n')
                    time.sleep(0.9)
                else:
                    # print('wrong sell move')
                    reward -= 0.5 * temp[CLOSE_COL] * portfolio[ticker][t]
                    portfolio['cash'].append(portfolio['cash'][t])
                    portfolio[ticker].append(portfolio[ticker][t])
            else:
                portfolio['cash'].append(portfolio['cash'][t])
                portfolio[ticker].append(portfolio[ticker][t])

            next_step = t + 1
            data_state = data[[i for i in data.columns if i != TICKER_COL]].iloc[[t + 1]]
            next_state = us.get_current_state_single_stock(data_state, ticker, portfolio, rf_clf, t=t + 1)

            done = t == (length - 1)
            trading_agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                final_wealth = portfolio[tickers] * data.iloc[t].CLOSE + wealth
                print(f'Final wealth: {final_wealth}')
                print('reward: ', reward)

            if len(trading_agent.memory) > batch_size:
                print('Learning..., current reward: ', reward)
                trading_agent.learn(batch_size)
        if e % 2 == 0:
            torch.save(trading_agent.model.state_dict(), f"{RL_MODEL_PATH}")  # Save PyTorch model

    print(trading_agent.history)  # Print the loss history
