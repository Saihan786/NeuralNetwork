"""Defines classes and subclasses for neuron layers."""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


def sigmoid_function(z):
    """The `item` function is used to change the resulting `np.float` datatype to the normal `float`."""
    
    return 1/(1 + np.exp(-z).item())


def sigmoid_derivative(z):
    return sigmoid_function(z) * (1.0 - sigmoid_function(z))


class CostsNotProvidedInBackPropagationError(Exception):
    pass


class BaseNeuronLayer:
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
        neurons: Optional[List[Neuron]] = None,
        next_layer: Optional[BaseNeuronLayer] = None,
        previous_layer: Optional[BaseNeuronLayer] = None,
    ) -> None:
        self.neurons: List[Neuron] = neurons if neurons else self.__initialise_neurons(size)
        self.previous_layer = previous_layer if previous_layer else None
        self.next_layer = next_layer if next_layer else None

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
    def activations(self) -> List[int]:
        return [neuron.activation for neuron in self.neurons]

    def print(self):
        print("\n\n\nprinting layer...")
        print("weights")
        for weight in self.weights:
            print(list(weight.values()))
        print("activations")
        print(self.activations)
        print("biases")
        print(self.biases)
        print()


class NonOutputNeuronLayer(BaseNeuronLayer):
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
                forward_neuron.activation = sigmoid_function(activation)

    def weight_changes_for_layer(self, neuron_to_weight_changes: Dict[Neuron, Dict[Neuron, float]], effect_of_actval_on_cost: Dict[Neuron, float], desired_outputs: List[float] = None):
        """
        This function updates `neuron_to_weight_changes` with the changes for the weights that come from this layer.

        If the next layer is an output layer, a special path will be taken.

        Args:
            - effect_of_actval_on_cost: A mapping between each neuron in the layer after the weights and the
            (indirect) effect of that neuron's activation value on the cost.
        """

        if isinstance(self.next_layer, OutputNeuronLayer) and not desired_outputs:
            raise CostsNotProvidedInBackPropagationError()

        elif isinstance(self.next_layer, OutputNeuronLayer):
            output_neuron_to_desired_outputs: Dict[Neuron, float] = {
                self.next_layer.neurons[i]: desired_outputs[i] for i in range(len(self.next_layer.neurons))
            }
            neurons_with_connections = self.neurons_with_connections

            for neuron_before_weight, connection in neurons_with_connections.items():
                weight_changes = {}
                for output_neuron, weight in connection.items():
                    desired_output: float = output_neuron_to_desired_outputs[output_neuron]

                    effect_of_actval_on_cost[output_neuron] = 2 * (desired_output - output_neuron.activation)

                    weight_changes[output_neuron] = effect_of_actval_on_cost[output_neuron] * neuron_before_weight.activation * sigmoid_derivative(output_neuron.activation)

                neuron_to_weight_changes[neuron_before_weight] = weight_changes

        else:
            neurons_with_connections = self.neurons_with_connections

            for neuron_before_weight, connection in neurons_with_connections.items():
                weight_changes = {}
                
                for neuron_after_weight, weight_to_adjust in connection.items():
                    effect = 0.0
                    for indirectly_affected_neuron, _ in neuron_after_weight.weights.items():
                        effect += effect_of_actval_on_cost[indirectly_affected_neuron] * \
                                  sigmoid_derivative(indirectly_affected_neuron.activation) * \
                                  neuron_after_weight.weights[indirectly_affected_neuron]

                    effect_of_actval_on_cost[neuron_after_weight] = effect

                    weight_changes[neuron_after_weight] = effect_of_actval_on_cost[neuron_after_weight] * sigmoid_derivative(neuron_after_weight.activation) * neuron_before_weight.activation
                
                neuron_to_weight_changes[neuron_before_weight] = weight_changes


class InitialNeuronLayer(NonOutputNeuronLayer):
    def __init__(
        self,
        size: int,
        neurons: Optional[List[Neuron]] = None,
        next_layer: Optional[BaseNeuronLayer] = None,
    ) -> None:
        super().__init__(
            size=size,
            neurons=neurons,
            next_layer=next_layer,
            previous_layer=None,
        )

    def activate_initial_layer(self, input_data: List[int]):
        """
        The neurons of this layer have their activation values set to the `input_data`. Otherwise nothing happens.

        The size of `input_data` must match the number of neurons, or nothing will happen.
        """

        if len(input_data) == len(self.neurons):
            for i in range(len(self.neurons)):
                self.neurons[i].activation = input_data[i]


class InternalNeuronLayer(NonOutputNeuronLayer):
    def __init__(
        self,
        size: int,
        neurons: Optional[List[Neuron]] = None,
        next_layer: Optional[BaseNeuronLayer] = None,
        previous_layer: Optional[BaseNeuronLayer] = None,
    ) -> None:
        super().__init__(
            size=size,
            neurons=neurons,
            next_layer=next_layer,
            previous_layer=previous_layer,
        )


class OutputNeuronLayer(BaseNeuronLayer):
    def __init__(
        self,
        size: int,
        neurons: Optional[List[Neuron]] = None,
        previous_layer: Optional[BaseNeuronLayer] = None,
    ) -> None:
        super().__init__(
            size=size,
            neurons=neurons,
            next_layer=None,
            previous_layer=previous_layer,
        )
