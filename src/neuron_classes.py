"""Defines classes for neuron, neuron layers, and the entire neural network."""

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
        weights: Dict[Neuron, int] = {},
    ) -> None:
        self.bias = bias
        self.weights = weights
        self.activation = 0

    def set_activation(self, activation: int) -> None:
        self.activation = activation

    def get_activation(self) -> int:
        return self.activation

    def set_bias(self, bias: int) -> None:
        self.bias = bias

    def get_bias(self) -> int:
        return self.bias

    def set_weights(self, weights: Dict[Neuron, int]) -> None:
        self.weights = weights

    def get_weights(self) -> Dict[Neuron, int]:
        return self.weights

    def set_weight(self, next_neuron: Neuron, weight: int) -> None:
        """Selects a weight to modify based on the corresponding
        forward-Neuron."""

        self.weights[next_neuron] = weight


class NeuronLayer:
    """
    Each layer is made up a group of neurons. Neurons in one layer determine
    the activation values for the neurons in the next layer, and have their
    own activation values determined by neurons in the previous layer.

    If `initial_layer`, activation values for the neurons are directly equated
    to the input data.

    If `neurons` and `size`, `size` is ignored.

    Attributes:
        - neurons (List[Neuron]): The neurons that make up this layer.

    Instance Methods:
        - get_neurons
        - get_biases
        - activate_next_layer
        - activate_initial_layer
    """

    next_layer: Optional[NeuronLayer] = None

    def __init__(
        self,
        size: int,
        initial_layer: bool = False,
        neurons: List[Neuron] = [],
        next_layer: Optional[NeuronLayer] = None,
    ) -> None:
        self.initial_layer = initial_layer
        self.neurons: List[Neuron] = (
            neurons if neurons else self.__initialise_neurons(size)
        )
        self.next_layer = next_layer

    def __initialise_neurons(self, size) -> List[Neuron]:
        if self.initial_layer:
            return [Neuron(initial_layer=True) for i in range(size)]
        else:
            return [Neuron() for i in range(size)]

    def get_neurons(self) -> List[Neuron]:
        return self.neurons

    def get_next_layer(self) -> Optional[NeuronLayer]:
        return self.next_layer

    def get_biases(self) -> List[int]:
        return [neuron.get_bias() for neuron in self.neurons]

    def activate_initial_layer(self, input_data: List[int]) -> str:
        """If this is the initial layer of the network, the neurons of this
        layer have their activation values set. Otherwise nothing happens."""

        if len(input_data) > len(self.neurons):
            return "Too much input data"
        elif len(input_data) < len(self.neurons):
            return "Too little input data"
        elif not self.initial_layer:
            return "Not initial layer"
        else:
            for i in range(len(self.neurons)):
                self.neurons[i].set_activation(input_data[i])
            return "Success"

    def activate_next_layer(self):
        """
        Finds the activation values for each neuron in the next layer.

        Uses:
            - The activation values of the neurons of this layer
            - The weight of every forward-connection.
        """

        if self.get_next_layer():
            for forward_neuron in self.get_next_layer().get_neurons():
                activation = forward_neuron.get_bias()
                for neuron in self.neurons:
                    activation += (
                        neuron.activation * neuron.get_weights()[forward_neuron]
                    )
                forward_neuron.set_activation(activation)


class Network:
    """
    This is an entire Neural Network. It can input some data and calculate
    the activation values for every neuron in the network, and output what the
    network thinks the data represents.

    If input data is specified as training data, the network will change its
    weights of connections and biases of neurons to output a more accurate
    result to the input data.

    The initial layer has one neuron for every value in the input data, and
    its activation is directly tied to its corresponding input data value.

    Each neuron in the output layer corresponds to a digit.
    """

    def __init__(self, layers: List[NeuronLayer] = []) -> None:
        self.layers = layers
        self.initial_layer: NeuronLayer = NeuronLayer(size=10)
        self.output_layer: NeuronLayer = NeuronLayer(size=10)

    def think(self, input_data: List[int]):
        """Return what the network thinks the input data represents, based on
        the current weights and biases."""

        self.initial_layer.activate_next_layer()

    def train(self, input_data: List[int]):
        """
        Trains the network to generate a result more accurate to the input
        data.

        NOTE: Requires backpropagation and the cost function to be set up.
        """

        pass
