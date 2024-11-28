"""
This file contains the implementation of the DQSN model.
"""

import random
from collections import namedtuple

import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

from .base.Neuron import NonSpikingLIFNode


class DQSN(nn.Module):
    """
    This class represents the DQSN model.
    It inherits from the nn.Module class and implements the forward pass of the model.

    Methods
    -------
    __init__(input_size, hidden_size, output_size, T=16)
        Initializes the DQSN model with the given arguments.
    forward(x)
        Performs the forward pass of the model.
    """

    def __init__(self, input_size, hidden_size, output_size, T=16):
        """
        Initializes the DQSN model with the given arguments.

        Parameters
        ----------
        input_size : int
            The size of the input tensor.
        hidden_size : int
            The size of the hidden layer.
        output_size : int
            The size of the output tensor.
        T : int, optional
            The number of time steps to simulate the spiking neurons. The default value is 16.
        """
        super().__init__()

        self.fc = nn.Sequential(
            layer.Linear(input_size, hidden_size),
            neuron.IFNode(),  # Relu
            layer.Linear(hidden_size, output_size),
            NonSpikingLIFNode(tau=2.0),
        )

        self.T = T

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        for t in range(self.T):
            self.fc(x)

        return self.fc[-1].v


class ReplayMemory(object):
    """
    A class to store and manage replay memory for experience replay in reinforcement learning.

    Attributes
    -----------
    capacity : int
        The maximum number of transitions that the replay memory can hold.
    memory : list
        A list to store the transitions.
    position : int
        The current position in the memory to insert the next transition.
    transition : namedtuple
        A named tuple representing a transition with fields: state, action, next_state, and reward.

    Methods
    --------
    __init__(self, capacity):
        Initializes the replay memory with a given capacity.
    push(self, *args):
        Adds a transition to the replay memory. If the memory is full, it overwrites the oldest transition.
    sample(self, batch_size):
        Samples a batch of transitions from the replay memory.
    __len__(self):
        Returns the current size of the replay memory.
    """

    def __init__(self, capacity):
        """
        Initializes the replay memory.

        Args:
            capacity (int): The maximum number of transitions that the memory can hold.

        Attributes:
            capacity (int): The maximum number of transitions that the memory can hold.
            memory (list): The list to store transitions.
            position (int): The current position in the memory to insert the next transition.
            transition (namedtuple): A named tuple representing a single transition in the environment.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )

    def push(self, *args):
        """
        Adds a transition to the replay memory.

        Args:
            *args: The components of the transition to be stored in memory.
                   Typically, this includes state, action, reward, next_state, and done.

        Notes:
            - If the memory is not yet at full capacity, a new slot is appended.
            - The transition is stored at the current position in the memory.
            - The position is updated in a circular manner, wrapping around to the beginning when the capacity is reached.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly samples a batch of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list: A list of randomly sampled experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
