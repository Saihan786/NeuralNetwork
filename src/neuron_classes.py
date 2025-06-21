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
        self.__weights = weights if weights else {}
        self.activation = 0

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights: Dict[Neuron, int]) -> None:
        self.__weights = weights


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

    @property
    def biases(self) -> List[int]:
        return [neuron.bias for neuron in self.neurons]

    @biases.setter
    def biases(self, biases: List[int]):
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron.bias = biases[i]

    @property
    def weights(self) -> List[Dict[Neuron, int]]:
        return [neuron.weights for neuron in self.neurons]

    @weights.setter
    def weights(self, all_weights: List[List[int]]):
        fneurons: List[Neuron] = self.next_layer.neurons

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron_weights = all_weights[i]

            neuron.weights = {
                fneurons[j]: neuron_weights[j] for j in range(len(fneurons))
            }

    @property
    def activations(self) -> List[int]:
        return [neuron.activation for neuron in self.neurons]

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
                self.neurons[i].activation = input_data[i]

    def activate_next_layer(self):
        """
        Sets the activation values for each neuron in the next layer.

        Uses:
            - The activation values of the neurons of this layer.
            - The weight of every forward-connection.
            - The bias of each neuron in the next layer.
        """

        if self.next_layer:
            for forward_neuron in self.next_layer.neurons:
                activation: int = forward_neuron.bias
                for neuron in self.neurons:
                    if neuron.weights:
                        activation += neuron.activation * neuron.weights[forward_neuron]
                forward_neuron.activation = activation


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

    def think(self, input_data: List[int]):
        """Return what the network thinks the input data represents, based on
        the current weights and biases."""

        self.initial_layer.activate_next_layer()

    def randomise(self):
        """Sets the biases and weights of every Neuron to a random number."""

    def activate_layers(self, input_data: List[int]) -> List[int]:
        """
        This activates every layer in the network. This means every neuron in
        the network will have its activation value set, based on the weights
        and biases already in the network.

        Args:
            - input_data (List[int]): Sets activation values for the initial
            layer.
        Returns:
            - A list of activation values of the output layer.
        """

        self.initial_layer.activate_initial_layer(input_data=input_data)
        for non_output_layer in self.layers[:-1]:
            non_output_layer.activate_next_layer()

        return self.output_layer.activations

    def cost_function(self, desired_output: List[int]) -> int:
        """
        Overall:
            - This determines the "cost" of the current set of weights and
            biases for the given training example.

        Cost:
            - Cost is the sum of all squared differences.
            - Each squared difference is between the actual output activation
            value and its corresponding desired activation value.

        Example:
            - Training data expects an activation value of 100 for the output
            neuron indicating 3, and 0 for all the other output neurons.
            - Cost is therefore found by seeing how far all the actual
            activation values are from 100 and 0.

        Args:
            - desired_output (List[int]): A list of expected output activation
            values corresponding to each output neuron.

        Returns:
            - -1 if `desired_output` is not in the correct format.
            - cost (int): Summed sqr differences between expected and actual
            activation values.
        """

        cost: int = 0
        output_neurons: List[Neuron] = self.output_layer.neurons
        if len(output_neurons) != len(desired_output):
            return -1

        for i in range(len(output_neurons)):
            actual_activation = output_neurons[i].activation
            desired_activation = desired_output[i]

            sqr_diff = actual_activation - desired_activation
            sqr_diff *= sqr_diff
            cost += sqr_diff

        return cost

    def train_one_example(
        self, input_data: List[int], desired_output: int, minimise_cost: bool
    ):
        """
        TODO
        NOTE: Requires cost_function and backpropagation to be set up.

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

    def train(self, all_input_data: List[List[int]], all_desired_outputs: List[int]):
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
