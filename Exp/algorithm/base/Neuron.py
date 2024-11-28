"""
This file contains the implementation of the NonSpikingLIFNode class, which is a non-spiking version of the LIFNode class.
"""

import torch
from spikingjelly.activation_based import neuron


class NonSpikingLIFNode(neuron.LIFNode):
    """
    This class represents a non-spiking Leaky Integrate-and-Fire (LIF) neuron model.
    It inherits from the LIFNode class and overrides the behavior to simulate a
    non-spiking neuron.

    Methods
    -------
    __init__(*args, **kwargs)
        Initializes the NonSpikingLIFNode with the given arguments.
    single_step_forward(x: torch.Tensor)
        Performs a single step forward pass for the neuron. Depending on the training
        mode and the presence of a reset voltage, it updates the membrane potential
        `v` using different neuronal charge functions.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the NonSpikingLIFNode with the given arguments.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def single_step_forward(self, x: torch.Tensor):
        """
        Performs a single step forward pass for the neuron.
        Depending on the training mode and the presence of a reset voltage,
        it updates the membrane potential `v` using different neuronal charge functions.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        """
        self.v_float_to_tensor(x)

        if self.training:
            self.neuronal_charge(x)
        else:
            if self.v_reset is None:
                if self.decay_input:
                    self.v = self.neuronal_charge_decay_input_reset0(
                        x, self.v, self.tau
                    )
                else:
                    self.v = self.neuronal_charge_no_decay_input_reset0(
                        x, self.v, self.tau
                    )
            else:
                if self.decay_input:
                    self.v = self.neuronal_charge_decay_input(
                        x, self.v, self.v_reset, self.tau
                    )
                else:
                    self.v = self.neuronal_charge_no_decay_input(
                        x, self.v, self.v_reset, self.tau
                    )
