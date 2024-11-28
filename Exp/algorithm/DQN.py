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
            neuron.IFNode(), # Relu
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
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
