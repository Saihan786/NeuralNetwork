"""Defines the neuron class."""

from __future__ import annotations
from typing import Dict, List, Optional


class Neuron:
    """
    Each neuron is defined by its bias, activation value, and weights for all
    of its forward-connections.

    Weights and biases of neurons in the previous layer determine this neuron's
    activation value.

    Forward-connections are connections between this neuron and all the neurons
    in the next layer.

    The activation value cannot be modified manually, but is determined by the
    bias of this neuron and the activation values of the neurons in the
    previous layer.

    Attributes:
        - Weights (Dict[Neuron, int]): All of the weights of the
        forward-connections of this neuron.
        - Bias (int): A single value associated with this neuron. It affects
        the activation value of this neuron.
        - Activation value (int): This is used to determine activation of
        neurons in the next layer, until the final layer neurons output a
        response to a query. This is recalculated for every input to the
        neural network.

    Instance Methods:
        - set_activation
        - get_activation
        - set_bias
        - get_bias
        - set_weights
        - get_weights
        - set_weight
    """

    def __init__(
        self,
        bias=0,
        weights: Optional[Dict[Neuron, float]] = None,
    ) -> None:
        self.bias = bias
        self.__weights = weights if weights else {}
        self.activation = 0

    @property
    def weights(self) -> Dict[Neuron, int]:
        return self.__weights

    @weights.setter
    def weights(self, weights: Dict[Neuron, float]) -> None:
        self.__weights = weights
