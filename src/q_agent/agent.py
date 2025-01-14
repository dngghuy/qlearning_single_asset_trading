import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        """
        Flexible Q-network that allows varying hidden layer structures.

        Args:
            input_dim (int): Dimension of input features.
            action_dim (int): Number of possible actions.
            hidden_layers (list): List of neurons in each hidden layer.
        """
        super(QNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class TradingAgent:
    def __init__(self, model, train=True):
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        # self.model_name = model_name
        self.train = train

        self.gamma = 0.8
        self.epsilon = 0.4
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.2

        self.model = model
        self.history = []
        self.action_history = []

    def make_action(self, state):
        """
        Chooses an action based on the current state using an epsilon-greedy strategy.
        """
        if self.train and np.random.binomial(1, self.epsilon):
            return np.random.randint(0, self.action_size)  # Explore

        state = torch.FloatTensor(state)  # Convert state to PyTorch tensor
        options = self.model(state)
        action = torch.argmax(options).item()

        self.action_history.append(action)
        return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, batch_size):
        """
        Updates the model using experience replay and a PyTorch training loop.
        """
        mini_batch = list(self.memory)[-batch_size:]
        criterion = torch.nn.MSELoss()  # Example loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for state, action, reward, next_state, done in mini_batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = torch.FloatTensor([reward])

            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state))).reshape(-1)

            # Forward pass
            output = self.model(state)
            output_action = output[0][action].reshape(-1)
            loss = criterion(output_action, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.history.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
