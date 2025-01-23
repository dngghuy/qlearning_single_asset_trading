import numpy as np
from collections import deque
import random
import torch

import src.q_agent.utils as us
from src.configs.configs import CLOSE_COL, BATCH_SIZE, REPLAY_MEMORY_SIZE, \
    TRAINING_FREQUENCY
import random
from collections import deque

import numpy as np
import torch

import src.q_agent.utils as us
from src.configs.configs import CLOSE_COL, BATCH_SIZE, REPLAY_MEMORY_SIZE, \
    TRAINING_FREQUENCY


class ReplayMemory:
    """
    A cyclic buffer to store experience tuples for training a reinforcement learning agent.
    """
    def __init__(self, capacity=REPLAY_MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        """
        Randomly samples a batch of experiences from the memory.

        Args:
            batch_size: The size of the batch to sample.

        Returns:
            A list of tuples, where each tuple represents an experience (state, action, reward, next_state, done).
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the current size of the memory.
        """
        return len(self.memory)


class EnvironmentSingleStock:
    """
    Represents the trading environment, managing the portfolio and generating states.
    """

    def __init__(self, data, feature_list, rf_clf, initial_wealth=5000.0):
        self.data = data
        self.feature_list = feature_list
        self.rf_clf = rf_clf  # Assuming this is a trained classifier needed for state representation
        self.initial_wealth = initial_wealth
        self.portfolio = None
        self.tickers = None
        self.current_step = 0
        self.length = len(data)
        self.current_buy = 0

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.portfolio, self.tickers = us.initialize_portfolio(self.data, self.initial_wealth)
        self.current_step = 0
        state = self.get_state()
        self.current_buy = 0
        return state

    def get_state(self):
        """
        Generates the current state based on market data, portfolio, and potentially other factors.
        """
        ticker = self.tickers[0]
        data_state = self.data[self.feature_list].iloc[[self.current_step]]
        state = us.get_current_state_single_stock(data_state, self.rf_clf)
        return state

    def step(self, action):
        """
        Takes a step in the environment based on the given action.

        Args:
            action: The action to take (0: hold, 1: buy, 2: sell).

        Returns:
            A tuple containing the next state, reward, done flag, and additional information (optional).
        """
        ticker = self.tickers[0]
        current_price = self.data[CLOSE_COL].iloc[self.current_step]
        reward = 0
        info = {}  # Can be used to store additional information about the step

        if action == 1:  # Buy
            if self.portfolio['cash'][self.current_step] >= current_price:
                num_shares = int(self.portfolio['cash'][self.current_step] / current_price)
                self.portfolio['cash'].append(self.portfolio['cash'][self.current_step] - num_shares * current_price)
                self.portfolio[ticker].append(self.portfolio[ticker][self.current_step] + num_shares)
                info['buy_price'] = num_shares * current_price  # Store buy price for later profit calculation
                reward -= 0.001 * num_shares * current_price / self.initial_wealth
                self.current_buy = self.current_step
            else:
                reward -= 0.0005  # Penalty for incorrect sell
                reward -= 0.001 * self.portfolio['cash'][self.current_step] / self.initial_wealth
                self.portfolio['cash'].append(self.portfolio['cash'][self.current_step])
                self.portfolio[ticker].append(self.portfolio[ticker][self.current_step])
                info['action_successful'] = False

        elif action == 2:  # Sell
            if self.portfolio[ticker][self.current_step] > 0 and self.current_step >= self.current_buy + 20:
                profit = current_price * (self.portfolio[ticker][self.current_step] - info.get('buy_price', 0))
                info['profit'] = profit
                reward += profit / ((current_price / self.data[CLOSE_COL].iloc[0]) * self.initial_wealth) - 1
                reward -= 0.001 * np.abs(profit) / self.initial_wealth  # Transaction Fee
                self.portfolio['cash'].append(
                    self.portfolio['cash'][self.current_step] + current_price * self.portfolio[ticker][
                        self.current_step])
                self.portfolio[ticker].append(0)
                info['buy_price'] = 0
                self.current_buy = 0
            else:
                reward -= 0.0005  # Penalty for incorrect sell
                reward -= 0.001 * self.portfolio['cash'][self.current_step] / self.initial_wealth
                self.portfolio['cash'].append(self.portfolio['cash'][self.current_step])
                self.portfolio[ticker].append(self.portfolio[ticker][self.current_step])
                info['action_successful'] = False

        else:  # Hold
            reward -= 0.001 * self.portfolio['cash'][self.current_step] / self.initial_wealth
            self.portfolio['cash'].append(self.portfolio['cash'][self.current_step])
            self.portfolio[ticker].append(self.portfolio[ticker][self.current_step])


        self.current_step += 1
        done = self.current_step == self.length

        if done:
            final_wealth = self.portfolio['cash'][-1] + self.portfolio[ticker][-1] * current_price
            additional_reward = final_wealth / ((self.data[CLOSE_COL].iloc[self.current_step - 1] /
                                                 self.data[CLOSE_COL].iloc[0]) * self.initial_wealth) - 1

            final_reward = final_wealth / self.initial_wealth - 1
            reward += (additional_reward)
            info['final_wealth'] = final_wealth
        else:
            next_state = self.get_state()

        return next_state if not done else None, reward, done, info


class Trainer:
    """
    Handles the training loop for the reinforcement learning agent.
    """

    def __init__(self, agent, environment, replay_memory, optimizer, model_path):
        """
        Initializes the trainer.

        Args:
            agent: The trading agent.
            environment: The trading environment.
            replay_memory: The replay memory.
            optimizer: The optimizer for training the agent's model.
        """
        self.agent = agent
        self.environment = environment
        self.replay_memory = replay_memory
        self.optimizer = optimizer
        self.model_path = model_path
        self.rewards_per_episode = []
        self.loss_history = []
        self.action_counts = {'hold': [], 'buy': [], 'sell': [], 'failed_buy': [], 'failed_sell': []}
        self.logged_actions = []
        self.profit_history = []

    def train(self, num_episodes=200):
        """
        Trains the agent for a specified number of episodes.

        Args:
            num_episodes: The number of episodes to train for.
        """
        for e in range(num_episodes):
            state = self.environment.reset()
            done = False
            total_reward = 0
            episode_loss = []
            action_counts = {'hold': 0, 'buy': 0, 'sell': 0, 'failed_buy': 0, 'failed_sell': 0}
            v = 0
            while not done:

                action = self.agent.make_action(state)
                next_state, reward, done, info = self.environment.step(action)
                self.replay_memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if action == 0:
                    action_counts['hold'] += 1
                elif action == 1:
                    if 'action_successful' in info and not info['action_successful']:
                        action_counts['failed_buy'] += 1
                    else:
                        action_counts['buy'] += 1
                elif action == 2:
                    if 'action_successful' in info and not info['action_successful']:
                        action_counts['failed_sell'] += 1
                    else:
                        action_counts['sell'] += 1

                if self.environment.current_step % TRAINING_FREQUENCY == 0 and len(self.replay_memory) > BATCH_SIZE:
                    loss = self.train_step()
                    episode_loss.append(loss)

                self.agent.update_epsilon()

            self.rewards_per_episode.append(total_reward)
            if episode_loss:
                self.loss_history.append(np.mean(episode_loss))
            else:
                self.loss_history.append(0)

            for action_type, count in action_counts.items():
                self.action_counts[action_type].append(count)

            if 'final_wealth' in info:
                profit = info.get('final_wealth', 0) - self.environment.initial_wealth
            else:
                profit = 0
            self.profit_history.append(profit)
            print(
                f"Episode {e}: Total Reward: {total_reward:.2f}, Avg Loss: {self.loss_history[-1]:.4f}, Profit: {profit:.2f}")
            print(
                f"  Action Counts - Hold: {action_counts['hold']}, Buy: {action_counts['buy']}, Sell: {action_counts['sell']}, Failed Buy: {action_counts['failed_buy']}, Failed Sell: {action_counts['failed_sell']}")
            if e % 2 == 0:
                torch.save(self.agent.model.state_dict(), self.model_path)

    def train_step(self):
        """
        Performs a single training step using a batch from the replay memory.

        Returns:
            The loss value.
        """
        batch = self.replay_memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        states = states.squeeze(1)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
        non_final_next_states = torch.stack(
            [torch.tensor(s, dtype=torch.float32) for s in next_states if s is not None]
        )

        state_action_values = self.agent.model(states).squeeze(1).gather(1, actions)
        next_state_values = torch.zeros(BATCH_SIZE, dtype=torch.float32)
        if non_final_next_states.size(0) > 0:
            next_state_values[non_final_mask] = self.agent.model(non_final_next_states).squeeze(1).max(1)[0].detach()

        target_q_values = rewards + (self.agent.gamma * next_state_values * (~dones).float())
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, target_q_values.unsqueeze(1))


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
