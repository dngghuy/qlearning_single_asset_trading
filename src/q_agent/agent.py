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
            # nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
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
    # def act(self, state):
    #     """
    #     Select an action based on the current state and epsilon-greedy strategy.
    #     """
    #     if self.train and random.random() < self.epsilon:
    #         return random.randint(0, self.action_dim - 1)
    #
    #     state = torch.FloatTensor(state).to(self.device)
    #     with torch.no_grad():
    #         q_values = self.model(state)
    #     return torch.argmax(q_values).item()
    #
    # def remember(self, state, action, reward, next_state, done):
    #     """
    #     Store experience in memory.
    #     """
    #     self.memory.append((state, action, reward, next_state, done))

    # def replay(self, batch_size):
    #     """
    #     Train the model using experiences sampled from memory.
    #     """
    #     if len(self.memory) < batch_size:
    #         return
    #
    #     mini_batch = random.sample(self.memory, batch_size)
    #
    #     for state, action, reward, next_state, done in mini_batch:
    #         state = torch.FloatTensor(state).to(self.device)
    #         next_state = torch.FloatTensor(next_state).to(self.device)
    #         target = reward
    #
    #         if not done:
    #             target += self.gamma * torch.max(self.model(next_state)).item()
    #
    #         q_values = self.model(state)
    #         target_q = q_values.clone()
    #         target_q[0][action] = target
    #
    #         self.optimizer.zero_grad()
    #         loss = self.loss_fn(q_values, target_q.detach())
    #         loss.backward()
    #         self.optimizer.step()
    #
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    #
    # def save_model(self, path):
    #     """
    #     Save the model to a file.
    #     """
    #     torch.save(self.model, path)
