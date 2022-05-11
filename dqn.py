import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self,
     state_size,
     action_size,
     seed,
     buffer_size = int(1e5),
     batch_size = 128,
     gamma = 0.99,
     tau = 1e-3,
     lr = 5e-4,
     update_every = 20,
     ):
        """ DQN agent

        This class instantiates a DQN agent with TODO

        Args:
            state_size (int): shape of the state encoding.
            action_size (int): number of actions in the environment.
            seed (int): seed for random number generators.
            buffer_size (int): "memory size", how many experiences can the agent store.
            batch_size (int): number of experiences in a batch.
            gamma (float): reward decay, rewards in the future are worse than
                           imidiate rewards. Between 0 and 1. 
            tau (float): factor with which we control the size of the update
                         of the target network. Between 0 and 1.
            lr (float): learning rate for the agent. Between 0 and 1.
            update_every (int): number of episodes after which the agent weights 
                                are updated.

        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

        # Q-Network
        self.qnetwork_local = \
            QNetwork(state_size, action_size, seed,).to(device)
        self.qnetwork_target = \
            QNetwork(state_size, action_size, seed,).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay memory
        self.memory = \
            ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """ Agent takes a "step"

        This function updates the internal state of the agent, adding the 
        experience to the buffer and triggering learning if the proper 
        conditions are met

        Args:
            state (): Old (or current) state of the environment
            action (): Action taken
            reward (): Reward obtained by executing `action` in `state`
            next_state (): Current (or next) state of the environment
            done (bool): indicates whether the episode is finished

        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)


        # Learn every update_every time steps.
        self.t_step += 1
        if self.t_step >= self.update_every:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                self.t_step = 0
                # Sample 
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma) 

    def act(self, state, eps):
        """ Policy

        Epsilon greedy policy using TODO

        Args:
            state: current state of the agent
            eps (float): current epsilon value. Likelyhood of the agent 
                         selecting an action at random.

        Returns:
            action (int): action the agent will take

        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval() # setting to eval mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # setting to train mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """ title

        does

        Args:

        Returns:

        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)