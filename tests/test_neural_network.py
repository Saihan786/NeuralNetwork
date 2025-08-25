from typing import List
import pytest
from neural_network import neuron_classes as neuron_classes


@pytest.fixture
def single_neuron_network():
    """Setup for single neuron network tests."""
    # Make three neurons
    initial_neuron = neuron_classes.Neuron()
    neuron = neuron_classes.Neuron()
    output_neuron = neuron_classes.Neuron()

    # Make a neuron layer for each neuron
    output_neuron_layer = neuron_classes.NeuronLayer(
        size=-1,
        neurons=[output_neuron],
    )
    neuron_layer = neuron_classes.NeuronLayer(size=-1, neurons=[neuron], next_layer=output_neuron_layer)
    initial_neuron_layer = neuron_classes.NeuronLayer(
        size=-1, neurons=[initial_neuron], next_layer=neuron_layer, initial_layer=True
    )

    # Set weights and biases for single neuron
    initial_neuron.weights = {neuron: 1}
    neuron.weights = {output_neuron: 1}

    neuron.bias = 5
    output_neuron.bias = 5

    # Make a neural network for the layers
    neural_network = neuron_classes.Network(layers=[initial_neuron_layer, neuron_layer, output_neuron_layer])

    return {
        "network": neural_network,
        "initial_neuron": initial_neuron,
        "neuron": neuron,
        "output_neuron": output_neuron,
        "initial_neuron_layer": initial_neuron_layer,
        "neuron_layer": neuron_layer,
        "output_neuron_layer": output_neuron_layer,
    }


@pytest.fixture
def ten_neuron_network():
    """Setup for ten neuron network tests."""
    # Make three lists of 10 neurons each
    output_ten_neurons = [neuron_classes.Neuron() for _ in range(10)]
    hidden_ten_neurons = [neuron_classes.Neuron() for _ in range(10)]
    input_ten_neurons = [neuron_classes.Neuron() for _ in range(10)]

    # Make the ten neuron layers
    output_layer_ten_neurons = neuron_classes.NeuronLayer(size=10, neurons=output_ten_neurons)
    hidden_layer_ten_neurons = neuron_classes.NeuronLayer(
        size=10, neurons=hidden_ten_neurons, next_layer=output_layer_ten_neurons
    )
    input_layer_ten_neurons = neuron_classes.NeuronLayer(
        size=10, neurons=input_ten_neurons, next_layer=hidden_layer_ten_neurons, initial_layer=True
    )

    # Set weights and biases for ten neurons
    for input_neuron in input_ten_neurons:
        input_neuron.weights = {hidden_neuron: 1 for hidden_neuron in hidden_ten_neurons}

    for hidden_neuron in hidden_ten_neurons:
        hidden_neuron.bias = 5
        hidden_neuron.weights = {output_neuron: 1 for output_neuron in output_ten_neurons}

    for output_neuron in output_ten_neurons:
        output_neuron.bias = 5

    expected_output_ten_neurons = [155] * 10

    return {
        "input_layer": input_layer_ten_neurons,
        "hidden_layer": hidden_layer_ten_neurons,
        "output_layer": output_layer_ten_neurons,
        "expected_output": expected_output_ten_neurons,
    }


def test_get_layers(single_neuron_network):
    network = single_neuron_network["network"]
    initial_neuron_layer = single_neuron_network["initial_neuron_layer"]
    neuron_layer = single_neuron_network["neuron_layer"]
    output_neuron_layer = single_neuron_network["output_neuron_layer"]

    assert network.layers == [initial_neuron_layer, neuron_layer, output_neuron_layer]


def test_activate_layers_one_neuron(single_neuron_network):
    network = single_neuron_network["network"]
    initial_neuron = single_neuron_network["initial_neuron"]
    neuron = single_neuron_network["neuron"]
    output_neuron = single_neuron_network["output_neuron"]

    initial_neuron.weights = {neuron: 1}
    neuron.bias = 5
    neuron.weights = {output_neuron: 1}
    output_neuron.bias = 5

    assert network.activate_layers([10]) == [20]
    assert network.output_layer.activations == [20]

    initial_neuron.weights = {neuron: 2}
    neuron.weights = {output_neuron: 2}

    assert network.activate_layers([10]) == [55]
    assert network.output_layer.activations == [55]


def test_activate_layers_ten_neurons(ten_neuron_network):
    # Create network
    network = neuron_classes.Network(
        layers=[
            ten_neuron_network["input_layer"],
            ten_neuron_network["hidden_layer"],
            ten_neuron_network["output_layer"],
        ]
    )

    # Test first activation
    input_data = [1] * 10
    assert network.activate_layers(input_data) == ten_neuron_network["expected_output"]
    assert network.output_layer.activations == ten_neuron_network["expected_output"]


def test_cost_function_with_input_data(single_neuron_network):
    """
    TODO:
        - Use conftest to establish a neural network that already has
        weights and biases.
            - Test `network.activate_layers()` separately.
    """
    network = single_neuron_network["network"]
    initial_neuron = single_neuron_network["initial_neuron"]
    neuron = single_neuron_network["neuron"]
    output_neuron = single_neuron_network["output_neuron"]

    OUTPUT_ACTIVATION_AFTER_INPUT_DATA = 142
    initial_neuron.bias = 1
    initial_neuron.weights[neuron] = 2

    neuron.bias = 10
    neuron.weights[output_neuron] = 11

    output_neuron.bias = 10
    output_neuron.activation = 10

    assert network.cost_function(desired_output=[20]) == [-10]

    applied_new_activation = network.cost_function(desired_output=[20], input_data=[1])

    output_neuron.activation = OUTPUT_ACTIVATION_AFTER_INPUT_DATA
    assert applied_new_activation == network.cost_function(desired_output=[20])


def test_cost_function_with_input_data_ten_neurons(ten_neuron_network):
    network = neuron_classes.Network(
        layers=[
            ten_neuron_network["input_layer"],
            ten_neuron_network["hidden_layer"],
            ten_neuron_network["output_layer"],
        ]
    )

    assert network.cost_function(desired_output=ten_neuron_network["expected_output"], input_data=[1] * 10) == [0] * 10
    assert network.cost_function(desired_output=[255] * 10, input_data=[2] * 10) == [0] * 10


def test_cost_function_with_incorrect_desired_output(single_neuron_network):
    """
    TODO:
        - Use conftest to establish a neural network that already has
        weights and biases.
            - Test `network.activate_layers()` separately.
    """
    network = single_neuron_network["network"]
    initial_neuron = single_neuron_network["initial_neuron"]
    neuron = single_neuron_network["neuron"]
    output_neuron = single_neuron_network["output_neuron"]

    initial_neuron.bias = 1
    initial_neuron.weights[neuron] = 2

    neuron.bias = 10
    neuron.weights[output_neuron] = 11

    output_neuron.bias = 10
    output_neuron.activation = 10

    NUM_OUTPUT_NEURONS = 1

    assert network.cost_function(desired_output=([20] * (NUM_OUTPUT_NEURONS + 1))) == []
    assert network.cost_function(desired_output=([20] * (NUM_OUTPUT_NEURONS))) != []


def test_backpropagate_single_neuron(single_neuron_network):
    network = single_neuron_network["network"]
    desired_output: List[int] = [100]

    costs_before_backprop: List[int] = network.cost_function(desired_output=desired_output, input_data=[1])

    print(f"costs_before_backprop={costs_before_backprop}")
    assert False
