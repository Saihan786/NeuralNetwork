"""Defines classes and subclasses for neuron layers."""

from __future__ import annotations
from neural_network import neuron_class
from typing import Dict, List, Optional
import numpy as np


def sigmoid_function(z) -> float:
    """
    Used to flatten activation values and weights to values from 0 to 1.

    The `item` function is used to change the resulting `np.float` datatype to
    the normal `float`.
    """

    return 1 / (1 + np.exp(-z).item())


def sigmoid_derivative(z) -> float:
    """Derivative of the sigmoid function. Used in backpropagation."""

    return sigmoid_function(z) * (1.0 - sigmoid_function(z))


class CostsNotProvidedInBackPropagationError(Exception):
    """Costs must be provided to find weight, bias changes for the output
    layer."""

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
        - neurons (List[neuron_class.Neuron]): The neurons that make up this layer.

    Instance Methods:
        - get_neurons

        - get_biases

        - activate_next_layer

        - activate_initial_layer

    Subclasses:
        - NonOutputNeuronLayer
            - neurons_with_connections(self) -> List[Dict[neuron_class.Neuron,float]]

            - weights(self) -> List[Dict[neuron_class.Neuron,float]]

            - weights(self, all_weights: List[List[float]])

            - weights_as_list(self) -> List[List[float]]

            - activate_next_layer(self)

            - calculate_changes_for_layer

        - InitialNeuronLayer
            - activate_initial_layer(self, input_data: List[int])

        - InternalNeuronLayer

        - OutputNeuronLayer

    """

    def __init__(
        self,
        size: int,
        neurons: Optional[List[neuron_class.Neuron]] = None,
        next_layer: Optional[BaseNeuronLayer] = None,
        previous_layer: Optional[BaseNeuronLayer] = None,
    ) -> None:
        self.neurons: List[neuron_class.Neuron] = (
            neurons if neurons else self.__initialise_neurons(size)
        )
        self.previous_layer = previous_layer if previous_layer else None
        self.next_layer = next_layer if next_layer else None

    def __initialise_neurons(self, size) -> List[neuron_class.Neuron]:
        return [neuron_class.Neuron() for i in range(size)]

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
    def neurons_with_connections(
        self,
    ) -> Dict[neuron_class.Neuron, Dict[neuron_class.Neuron, float]]:
        """Each Neuron in this layer is mapped to its weights and returned in
        a dictionary."""

        return {neuron: neuron.weights for neuron in self.neurons}

    @property
    def weights(self) -> List[Dict[neuron_class.Neuron, float]]:
        return [neuron.weights for neuron in self.neurons]

    @weights.setter
    def weights(self, all_weights: List[List[float]]):
        """
        `all_weights` is structured as the following:
        [
            [ # weights for neuron_1 in this layer
                1.0,
                2.0,
                ...
            ]
        ]
        """

        if not self.next_layer:
            return

        fneurons: List[neuron_class.Neuron] = self.next_layer.neurons

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron_weights = all_weights[i]

            neuron.weights = {
                fneurons[j]: neuron_weights[j] for j in range(len(fneurons))
            }

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

            - The sigmoid function to flatten activation values.
        """

        if self.next_layer:
            for forward_neuron in self.next_layer.neurons:
                activation: int = forward_neuron.bias
                for neuron in self.neurons:
                    if neuron.weights:
                        activation += neuron.activation * neuron.weights[forward_neuron]
                forward_neuron.activation = sigmoid_function(activation)

    def changes_for_penultimate_layer(
        self,
        neuron_to_bias_changes: Dict[neuron_class.Neuron, float],
        neuron_to_weight_changes: Dict[
            neuron_class.Neuron, Dict[neuron_class.Neuron, float]
        ],
        effect_of_actval_on_cost: Dict[neuron_class.Neuron, float],
        desired_outputs: List[float],
    ):
        """
        This is a special case of `calculate_changes_for_layer`.

        For a group of weights that precede a layer, part of the calculation
        of one weight change involves finding the indirect effect of the
        weight on activation values of even later layers, which is done by
        applying derivatives to `effect_of_actval_on_cost` on those future
        neurons.

        This doesn't apply in the penultimate layer, as there is only one more
        later layer, thus no entries in `effect_of_actval_on_cost` for neurons
        in later layers. However, the cost provided to
        `calculate_changes_for_layer` replaces this, and this method
        represents this special case implementation.

        The same logic applies to bias changes.
        """

        output_neuron_to_desired_outputs: Dict[neuron_class.Neuron, float] = {
            self.next_layer.neurons[i]: desired_outputs[i]
            for i in range(len(self.next_layer.neurons))
        }
        neurons_with_connections = self.neurons_with_connections

        for neuron_before_weight, connection in neurons_with_connections.items():
            weight_changes = {}
            for output_neuron, weight in connection.items():
                desired_output: float = output_neuron_to_desired_outputs[output_neuron]

                effect_of_actval_on_cost[output_neuron] = 2 * (
                    desired_output - output_neuron.activation
                )

                common_change = effect_of_actval_on_cost[
                    output_neuron
                ] * sigmoid_derivative(output_neuron.activation)
                bias_change = common_change
                weight_changes[output_neuron] = (
                    common_change * neuron_before_weight.activation
                )

                neuron_to_bias_changes[output_neuron] = bias_change
            neuron_to_weight_changes[neuron_before_weight] = weight_changes

    def bias_changes_for_initial_layer(
        self,
        neuron_before_weight: neuron_class.Neuron,
        connection: Dict[neuron_class.Neuron, float],
        effect_of_actval_on_cost: Dict[neuron_class.Neuron, float],
        neuron_to_bias_changes: Dict[neuron_class.Neuron, float],
    ):
        """
        This is a special case of `calculate_changes_for_layer`.

        Normally, `calculate_changes_for_layer` finds changes for the weights
        coming out from this layer and biases for neurons of the next layer.
        This means that the method does not cover the input layer at all.

        To remedy this, this helper method is used while iterating through
        neurons of this layer. It handles the calculations for biases of this
        layer while the surrounding function handles biases of the next layer.
        """

        effect = 0.0
        input_neuron = neuron_before_weight

        for neuron_after_weight, _ in connection.items():
            effect += (
                effect_of_actval_on_cost[neuron_after_weight]
                * sigmoid_derivative(neuron_after_weight.activation)
                * input_neuron.weights[neuron_after_weight]
            )

        bias_change = effect * sigmoid_derivative(neuron_after_weight.activation)
        neuron_to_bias_changes[input_neuron] = bias_change

    def calculate_changes_for_layer(
        self,
        neuron_to_bias_changes: Dict[neuron_class.Neuron, float],
        neuron_to_weight_changes: Dict[
            neuron_class.Neuron, Dict[neuron_class.Neuron, float]
        ],
        effect_of_actval_on_cost: Dict[neuron_class.Neuron, float],
        desired_outputs: List[float] = None,
    ):
        """
        This function updates `neuron_to_weight_changes` with the changes for
        the weights that come from this layer, and `neuron_to_bias_changes`
        likewise.

        The general idea for finding a weight change is to find that weight's
        indirect effect on the final OVERALL cost of the network for some
        input data. This means you have to find how that weight affected the
        activation of the neuron it leads to, the activations of the neurons
        in the following layer, and so on... and you do this for every single
        weight in the network!

        To make this much easier, we start at the output layer and find its
        neurons' weight changes. Then, the previous layer's neurons' weight
        changes will use the output layer changes to figure out the indirect
        effect of the previous weight on the output neuron activations.

        This is then repeated for every layer, propagating changes backwards
        for every new layer. The repetition is handled by the calling method
        of the network class - this method only handles updating propagation
        values for this layer.

        The process is only slightly different for finding bias changes (one
        value in the calculation for a weight change is the activation value
        of a previous neuron, but that value is replaced by 1 when finding
        the bias change).

        Args:
            - neuron_to_bias_changes

            - neuron_to_weight_changes

            - effect_of_actval_on_cost: A mapping between each neuron in the
              layer after the weights and the (indirect) effect of that
              neuron's activation value on the cost.
        """

        if isinstance(self.next_layer, OutputNeuronLayer) and not desired_outputs:
            raise CostsNotProvidedInBackPropagationError()

        elif isinstance(self.next_layer, OutputNeuronLayer):
            self.changes_for_penultimate_layer(
                neuron_to_bias_changes,
                neuron_to_weight_changes,
                effect_of_actval_on_cost,
                desired_outputs,
            )
            return

        neurons_with_connections = self.neurons_with_connections

        for neuron_before_weight, connection in neurons_with_connections.items():
            weight_changes = {}

            for neuron_after_weight, weight_to_adjust in connection.items():
                effect = 0.0
                for (
                    indirectly_affected_neuron,
                    _,
                ) in neuron_after_weight.weights.items():
                    effect += (
                        effect_of_actval_on_cost[indirectly_affected_neuron]
                        * sigmoid_derivative(indirectly_affected_neuron.activation)
                        * neuron_after_weight.weights[indirectly_affected_neuron]
                    )

                effect_of_actval_on_cost[neuron_after_weight] = effect

                common_change = effect_of_actval_on_cost[
                    neuron_after_weight
                ] * sigmoid_derivative(neuron_after_weight.activation)
                bias_change = common_change
                weight_changes[neuron_after_weight] = (
                    common_change * neuron_before_weight.activation
                )

                neuron_to_bias_changes[neuron_after_weight] = bias_change
            neuron_to_weight_changes[neuron_before_weight] = weight_changes

            if isinstance(self, InitialNeuronLayer):
                # repeat the above for the input layer, as it isn't covered for weight backpropagation

                self.bias_changes_for_initial_layer(
                    neuron_before_weight,
                    connection,
                    effect_of_actval_on_cost,
                    neuron_to_bias_changes,
                )


class InitialNeuronLayer(NonOutputNeuronLayer):
    def __init__(
        self,
        size: int,
        neurons: Optional[List[neuron_class.Neuron]] = None,
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
        The neurons of this layer have their activation values set to the `input_data`.

        The size of `input_data` must match the number of neurons, or nothing will happen.
        """

        if len(input_data) == len(self.neurons):
            for i in range(len(self.neurons)):
                self.neurons[i].activation = input_data[i]


class InternalNeuronLayer(NonOutputNeuronLayer):
    def __init__(
        self,
        size: int,
        neurons: Optional[List[neuron_class.Neuron]] = None,
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
        neurons: Optional[List[neuron_class.Neuron]] = None,
        previous_layer: Optional[BaseNeuronLayer] = None,
    ) -> None:
        super().__init__(
            size=size,
            neurons=neurons,
            next_layer=None,
            previous_layer=previous_layer,
        )
