import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layers):
        """
        Flexible Q-network that allows varying hidden layer structures.

        Args:
            input_dim (int): Dimension of input features.
            action_dim (int): Number of possible actions.
            hidden_layers (list): List of neurons in each hidden layer.
        """
        super(QNetwork, self).__init__()
        layers = []

        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden Layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        # Output Layer
        layers.append(nn.Linear(hidden_layers[-1], action_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TradingAgent:
    def __init__(self, input_dim, action_dim, hidden_layers, model_path=None, train=True):
        """
        Reinforcement Learning Trading Agent.

        Args:
            input_dim (int): Number of features in the state.
            action_dim (int): Number of possible actions.
            hidden_layers (list): List of neurons per layer for Q-network.
            model_path (str): Path to load a pretrained model.
            train (bool): Whether the agent is training.
        """
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.train = train
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.8
        self.memory = deque(maxlen=100000)
        self.inventory = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize or load model
        if model_path and not train:
            self.model = torch.load(model_path).to(self.device)
        else:
            self.model = QNetwork(input_dim, action_dim, hidden_layers).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        """
        Select an action based on the current state and epsilon-greedy strategy.
        """
        if self.train and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Train the model using experiences sampled from memory.
        """
        if len(self.memory) < batch_size:
            return

        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward

            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()

            q_values = self.model(state)
            target_q = q_values.clone()
            target_q[0][action] = target

            self.optimizer.zero_grad()
            loss = self.loss_fn(q_values, target_q.detach())
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        """
        Save the model to a file.
        """
        torch.save(self.model, path)
