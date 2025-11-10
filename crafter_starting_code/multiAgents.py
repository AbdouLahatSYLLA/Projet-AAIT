# import the necessary libraries.
"""
    Authors: Agnes Maryline Sasha Esméralda KAMENI & ABDOU LAHAT SYLLA
"""
import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ReplayBuffer:
    """
    Replay buffer to store transitions (state, action, reward, next_state, done).
    """

    # Initializes a 'deque', an optimized list with a maximum size.
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    #  Adds a transition (an experience) to the buffer.
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Samples and returns a random batch of transitions from the buffer.
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    # returns the current buffer size.
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural network to approximate the Q-function (action-value).
    """

    # Class constructor, inherits from PyTorch's nn.Module.
    def __init__(self, num_actions, history_length=4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(history_length, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.relu = nn.ReLU()

    # Defines the forward pass of data through the network.
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

class DQNAgentBase:
    """
    BASE CLASS containing all shared logic for DQN and DoubleDQN.
    """

    def __init__(self, action_num, history_length=4, buffer_size=100000, batch_size=32, gamma=0.99, learning_rate=1e-5):
        self.action_num = action_num
        self.batch_size = batch_size
        self.gamma = gamma

        self.q_network = QNetwork(action_num, history_length)
        self.target_network = QNetwork(action_num, history_length)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()

    def act(self, observation, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_num - 1)
        else:
            obs_tensor = observation.unsqueeze(0).float()
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            return q_values.argmax().item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # [FR] 2. Convertir les données en tenseurs PyTorch.
        # [EN] 2. Convert data to PyTorch tensors.
        #states = torch.from_numpy(np.array(states, dtype=np.float32))
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        #next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        current_q_values = self.q_network(states).gather(1, actions)


        # [FR] On appelle une méthode que les sous-classes vont définir.
        # [EN] We call a method that the subclasses will define.
        with torch.no_grad():
            next_q_values = self._get_next_q_values(next_states)

            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Copies the weights from the main network to the target network.
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _get_next_q_values(self, next_states):
        """
        "Abstract" method that subclasses MUST implement.
        """
        raise NotImplementedError("Cette méthode doit être implémentée par la sous-classe / this method must be implemented by the sub-class")


class DQNAgent(DQNAgentBase):
    """
    Standard DQN implementation.
    """

    # Standard DQN logic: max(Q_target)
    def _get_next_q_values(self, next_states):
        return self.target_network(next_states).max(1, keepdim=True)[0]


class DoubleDQNAgent(DQNAgentBase):
    """
    Double DQN implementation.
    """

    # Double DQN logic: Q_target(argmax(Q_online))
    def _get_next_q_values(self, next_states):
        best_next_actions = self.q_network(next_states).argmax(1, keepdim=True)
        return self.target_network(next_states).gather(1, best_next_actions)