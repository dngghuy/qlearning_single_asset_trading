from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        """
        Flexible Q-network that allows varying hidden layer structures.
Ø
        Args:
            input_dim (int): Dimension of input features.
            action_dim (int): Number of possible actions.
            hidden_layers (list): List of neurons in each hidden layer.
        """
        super(QNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class TradingAgent:
    def __init__(self, model, train=True):
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.train = train

        self.gamma = 0.95
        self.epsilon = 0.4
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.model = model
        self.history = []
        self.action_history = []

    def make_action(self, state):
        """
        Chooses an action based on the current state using an epsilon-greedy strategy.
        """
        if self.train and np.random.binomial(1, self.epsilon):
            return np.random.randint(0, self.action_size)

        state = torch.FloatTensor(state)
        options = self.model(state)
        action = torch.argmax(options).item()

        self.action_history.append(action)
        return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
