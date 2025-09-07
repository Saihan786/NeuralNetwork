from typing import List
import pytest
from neural_network import neuron_classes as neuron_classes


@pytest.fixture
def single_neuron_network():
    """Setup for one neuron network tests."""

    # Make three lists of 10 neurons each
    output_one_neuron = [neuron_classes.Neuron() for _ in range(1)]
    hidden_one_neuron = [neuron_classes.Neuron() for _ in range(1)]
    input_one_neuron = [neuron_classes.Neuron() for _ in range(1)]

    # Make the one neuron layers
    output_layer_one_neurons = neuron_classes.NeuronLayer(size=-1, neurons=output_one_neuron)
    hidden_layer_one_neurons = neuron_classes.NeuronLayer(
        size=-1, neurons=hidden_one_neuron, next_layer=output_layer_one_neurons
    )
    input_layer_one_neurons = neuron_classes.NeuronLayer(
        size=-1, neurons=input_one_neuron, next_layer=hidden_layer_one_neurons, initial_layer=True
    )

    # Set previous layers
    output_layer_one_neurons.previous_layer = hidden_layer_one_neurons
    hidden_layer_one_neurons.previous_layer = input_layer_one_neurons

    return {
        "input_layer": input_layer_one_neurons,
        "hidden_layer": hidden_layer_one_neurons,
        "output_layer": output_layer_one_neurons,
    }


@pytest.fixture
def ten_neuron_network():
    """Setup for ten neuron network tests."""

    # Make three lists of 10 neurons each
    output_ten_neuron = [neuron_classes.Neuron() for _ in range(10)]
    hidden_ten_neuron = [neuron_classes.Neuron() for _ in range(10)]
    input_ten_neuron = [neuron_classes.Neuron() for _ in range(10)]

    # Make the ten neuron layers
    output_layer_ten_neurons = neuron_classes.NeuronLayer(size=-1, neurons=output_ten_neuron)
    hidden_layer_ten_neurons = neuron_classes.NeuronLayer(
        size=-1, neurons=hidden_ten_neuron, next_layer=output_layer_ten_neurons
    )
    input_layer_ten_neurons = neuron_classes.NeuronLayer(
        size=-1, neurons=input_ten_neuron, next_layer=hidden_layer_ten_neurons, initial_layer=True
    )

    # Set previous layers
    output_layer_ten_neurons.previous_layer = hidden_layer_ten_neurons
    hidden_layer_ten_neurons.previous_layer = input_layer_ten_neurons

    return {
        "input_layer": input_layer_ten_neurons,
        "hidden_layer": hidden_layer_ten_neurons,
        "output_layer": output_layer_ten_neurons,
    }


@pytest.fixture
def five_neuron_network():
    """Setup for five neuron network tests."""

    # Make three lists of 10 neurons each
    output_five_neurons = [neuron_classes.Neuron() for _ in range(5)]
    hidden_five_neurons = [neuron_classes.Neuron() for _ in range(5)]
    input_five_neurons = [neuron_classes.Neuron() for _ in range(5)]

    # Make the five neuron layers
    output_layer_five_neurons = neuron_classes.NeuronLayer(size=-1, neurons=output_five_neurons)
    hidden_layer_five_neurons = neuron_classes.NeuronLayer(
        size=-1, neurons=hidden_five_neurons, next_layer=output_layer_five_neurons
    )
    input_layer_five_neurons = neuron_classes.NeuronLayer(
        size=-1, neurons=input_five_neurons, next_layer=hidden_layer_five_neurons, initial_layer=True
    )

    # Set previous layers
    output_layer_five_neurons.previous_layer = hidden_layer_five_neurons
    hidden_layer_five_neurons.previous_layer = input_layer_five_neurons

    return {
        "input_layer": input_layer_five_neurons,
        "hidden_layer": hidden_layer_five_neurons,
        "output_layer": output_layer_five_neurons,
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


def test_cost_function_no_input_data(single_neuron_network):
    """Basic cost function execution with 3 layers with one neuron each and predetermined activation values."""

    input_layer = single_neuron_network['input_layer']
    hidden_layer = single_neuron_network['hidden_layer']
    output_layer = single_neuron_network['output_layer']

    for neuron in input_layer.neurons + hidden_layer.neurons + output_layer.neurons:
        neuron.activation = 1

    input_layer.weights = [[1.0] for _ in range(1)]
    hidden_layer.weights = [[1.0] for _ in range(1)]

    network = neuron_classes.Network(layers = [input_layer, hidden_layer, output_layer])
    cost: float = network.cost_function(desired_activation_values=[1.0])

    assert cost == 0.0


def test_cost_function_with_input_data(single_neuron_network):
    """Basic cost function execution but activation is calculated."""

    input_layer = single_neuron_network['input_layer']
    hidden_layer = single_neuron_network['hidden_layer']
    output_layer = single_neuron_network['output_layer']

    input_layer.weights = [[1.0] for _ in range(1)]
    hidden_layer.weights = [[1.0] for _ in range(1)]

    network = neuron_classes.Network(layers = [input_layer, hidden_layer, output_layer])

    # Activation for the three neurons should be 1.0 each
    cost: float = network.cost_function(desired_activation_values=[1.0], input_data=[1.0])

    assert cost == 0.0


def test_cost_function_with_input_data_ten_neurons(ten_neuron_network):
    input_layer = ten_neuron_network['input_layer']
    hidden_layer = ten_neuron_network['hidden_layer']
    output_layer = ten_neuron_network['output_layer']

    for neuron in output_layer.neurons:
        neuron.activation = 1
    
    network = neuron_classes.Network(layers=[input_layer, hidden_layer, output_layer])
    cost_1: float = network.cost_function(desired_activation_values=[3.0]*10)
    network.print()

    # (3.0 - 1.0)^2 + (3.0 - 1.0)^2 + ...
    assert cost_1 == 40.0

    # calculates new activation values, but all biases and weights are set to 0 so most actvals are set to 0
    # (most, because input layer neurons depend on the input_data, not on the calculation)
    cost_2: float = network.cost_function(desired_activation_values=[3.0]*10, input_data=[1.0]*10)

    # (3.0 - 0.0)^2 + (3.0 - 0.0)^2 + ...
    assert cost_2 == 90.0

    # after setting up w+b below, one h_neuron should have actval (1*1)*10 + 5 = 15, and one o_neuron should
    # have actval (15*1)*10 + 5 = 155
    input_layer.weights = [[1.0]*10 for _ in range(10)]
    hidden_layer.weights = [[1.0]*10 for _ in range(10)]
    for neuron in hidden_layer.neurons + output_layer.neurons:
        neuron.bias = 5

    cost_3: float = network.cost_function(desired_activation_values=[3.0]*10, input_data=[1.0]*10)

    assert cost_3 == (3.0 - 155.0) * (3.0 - 155.0) * 10


def test_cost_function_with_incorrect_desired_activation_values(single_neuron_network):
    """Basic cost function execution but activation is calculated."""

    INCORRECT_NUM_OUTPUT_NEURONS = 2

    input_layer = single_neuron_network['input_layer']
    hidden_layer = single_neuron_network['hidden_layer']
    output_layer = single_neuron_network['output_layer']

    network = neuron_classes.Network(layers = [input_layer, hidden_layer, output_layer])
    network.print()

    with pytest.raises(neuron_classes.IncorrectInputError):
        network.cost_function(desired_activation_values=([0] * (INCORRECT_NUM_OUTPUT_NEURONS)))


def test_backpropagate_decreases_cost(five_neuron_network):
    input_layer = five_neuron_network['input_layer']
    hidden_layer = five_neuron_network['hidden_layer']
    output_layer = five_neuron_network['output_layer']

    # Set up predictable weights
    hidden_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
    input_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]

    network = neuron_classes.Network([input_layer, hidden_layer, output_layer])

    old_cost = network.cost_function(
        desired_activation_values=[2.0, 0.0, 0.0, 0.0, 0.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    network.backpropagate([2.0, 0.0, 0.0, 0.0, 0.0])

    new_cost = network.cost_function(
        desired_activation_values=[2.0, 0.0, 0.0, 0.0, 0.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert new_cost < old_cost


def test_backpropagate_repeatedly_decreases_cost(five_neuron_network):
    input_layer = five_neuron_network['input_layer']
    hidden_layer = five_neuron_network['hidden_layer']
    output_layer = five_neuron_network['output_layer']

    # Set up predictable weights
    hidden_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
    input_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]

    network = neuron_classes.Network([input_layer, hidden_layer, output_layer])

    old_cost = network.cost_function(
        desired_activation_values=[2.0, 0.0, 0.0, 0.0, 0.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )

    for i in range(100):
        network.backpropagate([2.0, 0.0, 0.0, 0.0, 0.0])
        
        new_cost = network.cost_function(
            desired_activation_values=[2.0, 0.0, 0.0, 0.0, 0.0],
            input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
        )

        assert new_cost < old_cost
        old_cost = new_cost


def test_backpropagate_with_zero_cost(five_neuron_network):
    """Test backpropagate doesn't change weights if cost is zero."""
    input_layer = five_neuron_network['input_layer']
    hidden_layer = five_neuron_network['hidden_layer']
    output_layer = five_neuron_network['output_layer']
    
    # Set up weights and activations
    hidden_layer.weights = [[1, 1, 1, 1, 1] for _ in range(5)]
    input_layer.weights = [[1, 1, 1, 1, 1] for _ in range(5)]
    for neuron in hidden_layer.neurons + output_layer.neurons:
        neuron.activation = 1
    
    network = neuron_classes.Network([input_layer, hidden_layer, output_layer])

    old_cost = network.cost_function(
        desired_activation_values=[1.0, 1.0, 1.0, 1.0, 1.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    activation_values_after_providing_input_data = network.output_layer.activations

    network.backpropagate(desired_outputs=activation_values_after_providing_input_data)

    new_cost = network.cost_function(
        desired_activation_values=[1.0, 1.0, 1.0, 1.0, 1.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert new_cost == old_cost
