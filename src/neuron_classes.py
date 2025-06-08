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
        weights: Optional[Dict[Neuron, int]] = None,
    ) -> None:
        self.bias = bias
        self.weights = weights if weights else {}
        self.activation = 0

    def set_activation(self, activation: int) -> None:
        self.activation = activation

    def get_activation(self) -> int:
        return self.activation

    def set_bias(self, bias: int) -> None:
        self.bias = bias

    def get_bias(self) -> int:
        return self.bias

    def set_weight(self, next_neuron: Neuron, weight: int) -> None:
        """Modifies a weighted connection between this Neuron and the given
        Neuron."""

        self.weights[next_neuron] = weight

    def set_weights(self, weights: Dict[Neuron, int]) -> None:
        """Sets all weighted connections between this Neuron and other
        Neurons."""

        self.weights = weights

    def get_weights(self) -> Dict[Neuron, int]:
        return self.weights


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
        neurons: Optional[List[Neuron]] = None,
        next_layer: Optional[NeuronLayer] = None,
    ) -> None:
        self.initial_layer = initial_layer
        self.neurons: List[Neuron] = (
            neurons if neurons else self.__initialise_neurons(size)
        )
        self.next_layer = next_layer

    def __initialise_neurons(self, size) -> List[Neuron]:
        return [Neuron() for i in range(size)]

    def get_neurons(self) -> List[Neuron]:
        return self.neurons

    def get_next_layer(self) -> Optional[NeuronLayer]:
        return self.next_layer

    def get_biases(self) -> List[int]:
        return [neuron.get_bias() for neuron in self.neurons]

    def get_activations(self) -> List[int]:
        return [neuron.get_activation() for neuron in self.neurons]

    def activate_initial_layer(self, input_data: List[int]):
        """
        If this is the initial layer of the network, the neurons of this
        layer have their activation values set to the `input_data`. Otherwise
        nothing happens.

        The size of `input_data` must match the number of neurons, or nothing
        will happen.
        """

        if self.initial_layer and len(input_data) == len(self.neurons):
            for i in range(len(self.neurons)):
                self.neurons[i].set_activation(input_data[i])

    def activate_next_layer(self):
        """
        TODO

        Sets the activation values for each neuron in the next layer.

        Uses:
            - The activation values of the neurons of this layer.
            - The weight of every forward-connection.
            - The bias of each neuron in the next layer.
        """

        if self.get_next_layer():
            for forward_neuron in self.get_next_layer().get_neurons():
                activation: int = forward_neuron.get_bias()
                for neuron in self.neurons:
                    activation += (
                        neuron.activation * neuron.get_weights()[forward_neuron]
                    )
                forward_neuron.set_activation(activation)


class Network:
    """
    This is an entire Neural Network. It can input some data and calculate
    the activation values for every neuron in the network, and output what
    the network thinks the data represents.

    If input data is specified as training data, the network will change its
    weights of connections and biases of neurons to output a more accurate
    result to the input data.

    The initial layer has one neuron for every value in the input data, and
    its activation is directly tied to its corresponding input data value.

    Each neuron in the output layer corresponds to a digit.

    Instance methods:
        - get_initial_layer
        - get_output_layers
        - get_layers
        - think
        - train
    """

    def __init__(self, layers: Optional[List[NeuronLayer]] = None) -> None:
        if layers:
            self.layers = layers
            self.initial_layer = layers[0]
            self.output_layer = layers[-1]
        else:
            self.layers = []
            self.initial_layer: NeuronLayer = NeuronLayer(size=10)
            self.output_layer: NeuronLayer = NeuronLayer(size=10)

    def get_initial_layer(self) -> NeuronLayer:
        return self.initial_layer

    def get_output_layer(self) -> NeuronLayer:
        return self.output_layer

    def get_layers(self) -> List[NeuronLayer]:
        return self.layers

    def think(self, input_data: List[int]):
        """Return what the network thinks the input data represents, based on
        the current weights and biases."""

        self.initial_layer.activate_next_layer()

    def randomise(self):
        """Sets the biases and weights of every Neuron to a random number."""

    def set_all_activation_values(self, input_data: List[int]):
        """
        This requires all biases and weights to be set.

        `input_data` is used to set activation values for the initial layer,
        then each following layer has their activation values set by the
        directly preceding layer.

        The weights from the preceding layer and the
        activation values of the Neurons in the preceding layer are used to
        determine the activation value of a Neuron of a layer.

        NOTE: Requires backpropagation and the cost function to be set up.
        """

        self.initial_layer.activate_initial_layer(input_data=input_data)
        for layer in self.layers[:-1]:
            layer.activate_next_layer()

    def train_one_example(
        self, input_data: List[int], desired_output: int, minimise_cost: bool
    ):
        """
        TODO
        Trains the network based on a single training example.

        The cost of input data is found from one training example. If
        `minimise_cost` is True, then the cost will be minimised to find a
        gradient vector with backpropagation, which is then applied to the
        network.

        The network is then considered trained on this one example.

        Args:
            - input_data (): Data that activates certain Neurons in the
            initial layer.
            - desired_output (): The output that the network should produce
            after receiving the `input_data`. Cost is measured against this.

        Returns:
            - cost (float): Cost of the network for this training example.
        """

    def train(self, all_input_data: List[List[int]]):
        """
        TODO
        Trains the network based on a list of input_data.

        The average cost over all training examples is found, and is passed to
        the cost minimisation functions.

        Cost minimisation is used alongside backpropagation to find the
        gradient vector, which is applied to each neuron to adjust all
        activation values.

        The network is then considered trained.
        """
