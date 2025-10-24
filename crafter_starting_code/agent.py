# agent.py

# [FR] On importe les bibliothèques nécessaires.
# [EN] We import the necessary libraries.
import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ReplayBuffer:
    """
    [FR] Buffer de relecture pour stocker les transitions (état, action, récompense, état_suivant, terminé).
    [EN] Replay buffer to store transitions (state, action, reward, next_state, done).
    """
    def __init__(self, capacity):
        # [FR] Initialise un 'deque', une liste optimisée avec une taille maximale.
        # [EN] Initializes a 'deque', an optimized list with a maximum size.
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # [FR] Ajoute une transition (une expérience) au buffer.
        # [EN] Adds a transition (an experience) to the buffer.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # [FR] Échantillonne et retourne un lot aléatoire de transitions depuis le buffer.
        # [EN] Samples and returns a random batch of transitions from the buffer.
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        # [FR] Permet d'obtenir la taille actuelle du buffer avec len(replay_buffer).
        # [EN] Allows getting the current buffer size using len(replay_buffer).
        return len(self.buffer)



class QNetwork(nn.Module):
    """
    [FR] Réseau de neurones pour approximer la fonction Q (action-valeur).
    [EN] Neural network to approximate the Q-function (action-value).
    """

    def __init__(self, num_actions, history_length=4):
        # [FR] Constructeur de la classe, hérite de nn.Module de PyTorch.
        # [EN] Class constructor, inherits from PyTorch's nn.Module.
        super(QNetwork, self).__init__()

        # [FR] Définition des couches convolutives pour traiter les images du jeu.
        # [EN] Definition of the convolutional layers to process the game images.
        self.conv1 = nn.Conv2d(history_length, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # [FR] Définition des couches entièrement connectées pour produire les Q-valeurs.
        # [EN] Definition of the fully connected layers to produce the Q-values.
        self.fc1 = nn.Linear(64 * 7 * 7,
                             512)  # [FR] La taille d'entrée dépend de la sortie des convs. / [EN] The input size depends on the convs' output.
        self.fc2 = nn.Linear(512, num_actions)

        # [FR] Définition de la fonction d'activation ReLU.
        # [EN] Definition of the ReLU activation function.
        self.relu = nn.ReLU()

    def forward(self, x):
        # [FR] Définit le passage avant (forward pass) des données à travers le réseau.
        # [EN] Defines the forward pass of data through the network.
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # [FR] Aplatit la sortie des couches convolutives pour la passer aux couches denses.
        # [EN] Flattens the output of the convolutional layers to pass it to the dense layers.
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))

        # [FR] La sortie finale correspond aux Q-valeurs pour chaque action.
        # [EN] The final output corresponds to the Q-values for each action.
        q_values = self.fc2(x)
        return q_values


class DQNAgent:
    """
    [FR] Agent qui implémente l'algorithme Deep Q-Learning.
    [EN] Agent that implements the Deep Q-Learning algorithm.
    """

    def __init__(self, action_num, history_length=4, buffer_size=100000, batch_size=32, gamma=0.99, learning_rate=1e-5):
        self.action_num = action_num  # [FR] Nombre d'actions possibles. / [EN] Number of possible actions.
        self.batch_size = batch_size  # [FR] Taille des lots pour l'apprentissage. / [EN] Batch size for learning.
        self.gamma = gamma  # [FR] Facteur d'actualisation (discount factor). / [EN] Discount factor.

        # [FR] Le réseau principal, mis à jour à chaque étape d'apprentissage.
        # [EN] The main network, updated at each learning step.
        self.q_network = QNetwork(action_num, history_length)

        # [FR] Le réseau cible, utilisé pour stabiliser l'apprentissage. Ses poids sont gelés pendant un certain temps.
        # [EN] The target network, used to stabilize learning. Its weights are frozen for a period.
        self.target_network = QNetwork(action_num, history_length)
        self.update_target_network()  # [FR] On copie les poids initiaux. / [EN] We copy the initial weights.

        # [FR] Optimiseur Adam pour mettre à jour les poids du q_network.
        # [EN] Adam optimizer to update the q_network's weights.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # [FR] Instance du Replay Buffer.
        # [EN] Instance of the Replay Buffer.
        self.replay_buffer = ReplayBuffer(buffer_size)

        # [FR] Fonction de perte (Erreur Quadratique Moyenne).
        # [EN] Loss function (Mean Squared Error).
        self.loss_fn = nn.MSELoss()

    def act(self, observation, epsilon=0.1):
        # [FR] Stratégie Epsilon-Greedy pour l'exploration.
        # [EN] Epsilon-Greedy strategy for exploration.
        if random.random() < epsilon:
            # [FR] Action aléatoire (exploration).
            # [EN] Random action (exploration).
            return random.randint(0, self.action_num - 1)
        else:
            # [FR] Action basée sur la plus grande Q-valeur (exploitation).
            # [EN] Action based on the highest Q-value (exploitation).
            #obs_tensor = torch.from_numpy(observation).unsqueeze(0).float()
            # [FR] L'observation est déjà un tenseur, on ajoute juste la dimension du lot avec unsqueeze(0).
            # [EN] The observation is already a tensor, we just add the batch dimension with unsqueeze(0).
            obs_tensor = observation.unsqueeze(0).float()
            with torch.no_grad():  # [FR] Pas de calcul de gradient ici. / [EN] No gradient calculation here.
                q_values = self.q_network(obs_tensor)
            return q_values.argmax().item()

    def learn(self):
        # [FR] Étape d'apprentissage de l'agent.
        # [EN] Learning step for the agent.
        if len(self.replay_buffer) < self.batch_size:
            return  # [FR] On attend que le buffer soit assez rempli. / [EN] We wait for the buffer to be filled enough.

        # [FR] 1. Échantillonner un lot de transitions.
        # [EN] 1. Sample a batch of transitions.
        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # [FR] 2. Convertir les données en tenseurs PyTorch.
        # [EN] 2. Convert data to PyTorch tensors.
        states = torch.from_numpy(np.array(states, dtype=np.float32))
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # [FR] 3. Calculer les Q-valeurs actuelles pour les paires (état, action) du lot.
        # [EN] 3. Compute the current Q-values for the (state, action) pairs in the batch.
        current_q_values = self.q_network(states).gather(1, actions)

        # [FR] 4. Calculer les valeurs cibles en utilisant l'équation de Bellman.
        # [EN] 4. Compute the target values using the Bellman equation.
        with torch.no_grad():
            # [FR] On utilise le target_network pour prédire la valeur de l'état suivant.
            # [EN] We use the target_network to predict the value of the next state.
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            # [FR] Si un état est terminal, sa valeur future est 0.
            # [EN] If a state is terminal, its future value is 0.
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # [FR] 5. Calculer la perte entre les valeurs actuelles et les cibles.
        # [EN] 5. Compute the loss between the current values and the targets.
        loss = self.loss_fn(current_q_values, targets)

        # [FR] 6. Mettre à jour les poids du q_network via la rétropropagation.
        # [EN] 6. Update the q_network's weights via backpropagation.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # [FR] Copie les poids du réseau principal vers le réseau cible.
        # [EN] Copies the weights from the main network to the target network.
        self.target_network.load_state_dict(self.q_network.state_dict())