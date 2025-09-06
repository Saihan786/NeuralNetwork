"""Defines classes for neuron, neuron layers, and the entire neural network."""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

class IncorrectInputError(Exception):
    pass


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

    def __init__(
        self,
        size: int,
        initial_layer: bool = False,
        output_layer: bool = False,
        neurons: Optional[List[Neuron]] = None,
        next_layer: Optional[NeuronLayer] = None,
        previous_layer: Optional[NeuronLayer] = None,
    ) -> None:
        self.initial_layer = initial_layer
        self.neurons: List[Neuron] = neurons if neurons else self.__initialise_neurons(size)
        if not output_layer:
            self.next_layer = next_layer if next_layer else None

        if not initial_layer:
            self.previous_layer = previous_layer if previous_layer else None

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
    def neurons_with_connections(self) -> List[Dict[Neuron, float]]:
        return {neuron: neuron.weights for neuron in self.neurons}

    @property
    def weights(self) -> List[Dict[Neuron, float]]:
        return [neuron.weights for neuron in self.neurons]

    @weights.setter
    def weights(self, all_weights: List[List[float]]):
        if not self.next_layer:
            return

        fneurons: List[Neuron] = self.next_layer.neurons

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron_weights = all_weights[i]

            neuron.weights = {fneurons[j]: neuron_weights[j] for j in range(len(fneurons))}

    @property
    def weights_as_list(self) -> List[List[float]]:
        return [list(neuron.weights.values()) for neuron in self.neurons]

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

    def proportional_changes(self, costs: List[float]) -> List[float]:
        """
        This method returns a list of desired changes to the biases of this layer's neurons and to the weights going
        into the neurons of this layer.
        
        Requires `self.previous_layer` to be set.

        This method returns a mapping between:
            {
                Each current neuron (c_neuron)
                :
                The c_neuron's list of proportional changes that it wants to make to neurons in the previous
                layer (p_neurons)
            }

        - A change is the overall impact that the p_neuron has on the c_neuron. It is defined by the p_neuron's
        activation value and its connection weight to the c_neuron (TODO - incorporate bias)

        This is how it works:
            - Consider one c_neuron and one p_neuron:
                - A `change` value is generated for the p_neuron.
                    - This is repeated for every p_neuron
                    
                    - Then, these `change` values are grouped into a list and mapped to the c_neuron.

        Args:
            - costs: A cost for each neuron in this layer which indicates 
        
        Returns:
            - Mapping of every neuron in this layer to a list of changes.
                - Each change is a float corresponding to one neuron in the previous layer.

                - Each change represents what amount this neuron wants to change a neuron in the previous layer by.
        """

        if not self.previous_layer:
            return {}

        if not len(costs) == len(self.neurons):
            raise ValueError(
                f"The length of the costs list must be the same as the number of neurons in the layer (which is {len(self.neurons)})."
            )

        changes: Dict[Neuron, List[float]] = {}

        for i in range(len(self.neurons)):
            c_neuron = self.neurons[i]
            overall_change = costs[i]
            print(f"overall_change={overall_change}")

            proportions: List[float] = [
                p_neuron.weights[c_neuron] * p_neuron.activation for p_neuron in self.previous_layer.neurons
            ]
            total = sum(proportions)
            print("\n\nabout to do proportions with these values:") 
            for p_neuron in self.previous_layer.neurons:
                print("weights =", p_neuron.weights[c_neuron])
                print("activation =", p_neuron.activation)
                print()
            print(f"proportions={proportions}")
            print(f"total={total}")


            changes[c_neuron] = [0 if total == 0 else (p / total) * overall_change for p in proportions]
        return changes


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

    def print(self):
        print("\n\n\nprinting network...")
        for layer in self.layers:
            print("weights")
            for weight in layer.weights:
                print(list(weight.values()))
            print("activations")
            print(layer.activations)
            print("biases")
            print(layer.biases)
            print()
        print("printed network!\n\n\n")

    def think(self, input_data: List[int]):
        """Return what the network thinks the input data represents, based on
        the current weights and biases."""

        self.initial_layer.activate_next_layer()

    def randomise(self):
        """Sets the biases and weights of every Neuron to a random number."""

    def activate_layers(self, input_data: List[float]) -> List[float]:
        """
        This activates every layer in the network. This means every neuron in
        the network will have its activation value set, based on the weights
        and biases already in the network.

        `input_data` must be the same size as the number of neurons in each layer.

        This assumes every layer in the network has the same number of neurons.

        Args:
            - input_data (List[int]): Sets activation values for the initial
            layer.
        Returns:
            - A list of activation values of the output layer.
        """

        if len(self.initial_layer.neurons) != len(input_data):
            raise IncorrectInputError(f"`input_data` should have been length {len(self.initial_layer.neurons)} but was length {len(input_data)}")

        self.initial_layer.activate_initial_layer(input_data=input_data)
        for non_output_layer in self.layers[:-1]:
            non_output_layer.activate_next_layer()

        return self.output_layer.activations

    def cost_function(self, desired_activation_values: List[float], input_data: Optional[List[float]] = None) -> List[float]:
        """
        Overall:
            - This determines the "cost" of the current set of weights and
            biases for the given training example.

            - If `input_data` is provided, then output activation values are
            recalculated, then cost is calculated based on these.
                - Else, the existing output activation values are used.

        Cost:
            - Cost is the list of all squared differences.
            - Each squared difference is between the actual output activation
            value and its corresponding desired activation value.

        Example:
            - Training data expects an activation value of 100 for the output
            neuron indicating 3, and 0 for all the other output neurons.
            - Cost is therefore found by seeing how far all the actual
            activation values are from 100 and 0.

        Args:
            - desired_activation_values (List[int]): A list of expected output activation
            values corresponding to each output neuron.

        Returns:
            - Empty list if `desired_activation_values` does not contain a value corresponding to each output neuron.
                - i.e., len(desired_activation_values) MUST EQUAL len(output_neurons)

            - cost (List[int]): List of summed sqr differences between expected and actual
            activation values.
        """

        if input_data:
            self.activate_layers(input_data=input_data)

        sqr_diffs: List[float] = []
        output_neurons: List[Neuron] = self.output_layer.neurons
        if len(output_neurons) != len(desired_activation_values):
            raise IncorrectInputError(f"`desired_activation_values` should have been length {len(output_neurons)} but was length {len(desired_activation_values)}")

        for i in range(len(output_neurons)):
            actual_activation = output_neurons[i].activation
            desired_activation = desired_activation_values[i]

            sqr_diff = desired_activation - actual_activation
            sqr_diff *= sqr_diff

            sqr_diffs.append(sqr_diff)

        return sqr_diffs

    def backpropagate(self, desired_outputs: List[float]):
        """
        Generates a list of changes to each weight and bias in the network, then applies them.

        For each layer of weights, to find the changes to weight, we need:
            - The weight itself

            - The activation value of the neuron the weight is coming from

            - The derivative of the total cost with respect to the activation value of the neuron the weight is connecting to
                - This is generated at each layer and propagated backwards for ease.

                - For the output layer, this is just the squared difference for an output neuron.                
        """

        effect_of_actval_on_cost: Dict[Neuron, float] = {}
        output_neuron_to_desired_outputs: Dict[Neuron, float] = {
            self.output_layer.neurons[i]: desired_outputs[i] for i in range(len(self.output_layer.neurons))
        }

        neuron_to_weight_changes: Dict[Neuron, Dict[Neuron, float]] = {}
        neurons_with_connections: dict = self.output_layer.previous_layer.neurons_with_connections

        for p_neuron, connection in neurons_with_connections.items():
            weight_changes = {}
            for o_neuron, weight in connection.items():
                desired_output: float = output_neuron_to_desired_outputs[o_neuron]

                effect_of_actval_on_cost[o_neuron] = 2 * (desired_output - o_neuron.activation)

                weight_changes[o_neuron] = effect_of_actval_on_cost[o_neuron] * p_neuron.activation

            neuron_to_weight_changes[p_neuron] = weight_changes


        neurons_with_connections: dict = self.output_layer.previous_layer.previous_layer.neurons_with_connections

        for q_neuron, connection in neurons_with_connections.items():
            weight_changes = {}
            
            for p_neuron, weight_to_p in connection.items():
                effect = 0.0
                for o_neuron, weight_to_o in p_neuron.weights.items():
                    effect += effect_of_actval_on_cost[o_neuron] * weight_to_p

                effect_of_actval_on_cost[p_neuron] = effect
                weight_changes[p_neuron] = effect_of_actval_on_cost[p_neuron] * q_neuron.activation
            
            neuron_to_weight_changes[q_neuron] = weight_changes
        
        for neuron, weight_changes in neuron_to_weight_changes.items():
            for target_neuron, change in weight_changes.items():
                neuron.weights[target_neuron] += change * 0.01  # Learning rate