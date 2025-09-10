"""Defines the neural network class."""

from __future__ import annotations
from typing import Dict, List, Optional
from neural_network import neuron_layer_classes


class IncorrectInputError(Exception):
    pass


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

    def __init__(self, layers: Optional[List[neuron_layer_classes.BaseNeuronLayer]] = None) -> None:
        if layers:
            self.layers = layers
            self.initial_layer = layers[0]
            self.output_layer = layers[-1]
        else:
            self.layers = []
            self.initial_layer: neuron_layer_classes.BaseNeuronLayer = neuron_layer_classes.BaseNeuronLayer(size=10)
            self.output_layer: neuron_layer_classes.BaseNeuronLayer = neuron_layer_classes.BaseNeuronLayer(size=10)

    def print(self):
        print("\n\n\nprinting network...")
        for idx, layer in enumerate(self.layers):
            print(f"printing layer {idx + 1}")
            if not isinstance(layer, neuron_layer_classes.OutputNeuronLayer):
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
        Generates a list of changes to each weight in the network, then applies them.

        For each layer of weights, to find the changes to weight that we want to apply, we need:
            - The weight itself.

            - The activation value of the neuron the weight is coming from.

            - The derivative of the total cost with respect to the activation value of the neuron the weight is connecting to.
                - This is generated at each layer and propagated backwards for ease.

                - For the output layer, this is just the squared difference for an output neuron.
        
        These values are then used to calculate the weight changes for the weights in a neuron layer. We then track the
        derivative values we need for the previous layer and repeat the process.
        """

        neuron_to_weight_changes: Dict[Neuron, Dict[Neuron, float]] = {}
        neuron_to_bias_changes: Dict[Neuron, float] = {}
        effect_of_actval_on_cost: neuron_layer_classes.BaseNeuronLayer = {}

        layer_before_weights = self.output_layer.previous_layer
        while layer_before_weights:
            layer_before_weights.weight_changes_for_layer(
                neuron_to_bias_changes=neuron_to_bias_changes,
                neuron_to_weight_changes=neuron_to_weight_changes,
                effect_of_actval_on_cost=effect_of_actval_on_cost,
                desired_outputs=desired_outputs
            )
            layer_before_weights = layer_before_weights.previous_layer

        for neuron, weight_changes in neuron_to_weight_changes.items():
            for target_neuron, change in weight_changes.items():
                original = neuron.weights[target_neuron]
                neuron.weights[target_neuron] += change * 0.05  # Learning rate

        for neuron, bias_change in neuron_to_bias_changes.items():
            neuron.bias += bias_change * 0.025  # Learning rate
