from typing import List
import pytest
from neural_network import neuron_class, neuron_layer_classes, network_class


@pytest.fixture
def single_neuron_network():
    """Setup for one neuron network tests."""

    # Make three lists of 10 neurons each
    output_one_neuron = [neuron_class.Neuron() for _ in range(1)]
    hidden_one_neuron = [neuron_class.Neuron() for _ in range(1)]
    input_one_neuron = [neuron_class.Neuron() for _ in range(1)]

    # Make the one neuron layers
    output_layer_one_neurons = neuron_layer_classes.BaseNeuronLayer(size=-1, neurons=output_one_neuron)
    hidden_layer_one_neurons = neuron_layer_classes.BaseNeuronLayer(
        size=-1, neurons=hidden_one_neuron, next_layer=output_layer_one_neurons
    )
    input_layer_one_neurons = neuron_layer_classes.BaseNeuronLayer(
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
    output_ten_neuron = [neuron_class.Neuron() for _ in range(10)]
    hidden_ten_neuron = [neuron_class.Neuron() for _ in range(10)]
    input_ten_neuron = [neuron_class.Neuron() for _ in range(10)]

    # Make the ten neuron layers
    output_layer_ten_neurons = neuron_layer_classes.BaseNeuronLayer(size=-1, neurons=output_ten_neuron)
    hidden_layer_ten_neurons = neuron_layer_classes.BaseNeuronLayer(
        size=-1, neurons=hidden_ten_neuron, next_layer=output_layer_ten_neurons
    )
    input_layer_ten_neurons = neuron_layer_classes.BaseNeuronLayer(
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
    # Make the three neuron layers
    layers = [
        neuron_layer_classes.InitialNeuronLayer(
            size=-1,
            neurons=[neuron_class.Neuron() for _ in range(5)],
        ),
        neuron_layer_classes.InternalNeuronLayer(
            size=-1,
            neurons=[neuron_class.Neuron() for _ in range(5)],
        ),
        neuron_layer_classes.OutputNeuronLayer(
            size=-1,
            neurons=[neuron_class.Neuron() for _ in range(5)],
        )
    ]


    # Set previous layers
    for i in range(1, 3):
        layers[i].previous_layer = layers[i - 1]

    # Set next layers
    for i in range(0, 2):
        layers[i].next_layer = layers[i + 1]

    # Set up network
    network = network_class.Network(layers=layers)

    return {
        "network": network,
        "layers": layers,
        "input_layer": layers[0],
        "hidden_layer": layers[1],
        "output_layer": layers[2],
    }





    # Make three lists of 10 neurons each
    output_five_neurons = [neuron_class.Neuron() for _ in range(5)]
    hidden_five_neurons = [neuron_class.Neuron() for _ in range(5)]
    input_five_neurons = [neuron_class.Neuron() for _ in range(5)]

    # Make the five neuron layers
    output_layer_five_neurons = neuron_layer_classes.BaseNeuronLayer(size=-1, neurons=output_five_neurons)
    hidden_layer_five_neurons = neuron_layer_classes.BaseNeuronLayer(
        size=-1, neurons=hidden_five_neurons, next_layer=output_layer_five_neurons
    )
    input_layer_five_neurons = neuron_layer_classes.BaseNeuronLayer(
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


@pytest.fixture
def ten_layer_network():
    # Make the ten neuron layers
    layers = [
        neuron_layer_classes.InitialNeuronLayer(
            size=-1,
            neurons=[neuron_class.Neuron() for _ in range(5)],
        ),
    ]

    layers += [neuron_layer_classes.InternalNeuronLayer(
        size=-1,
        neurons=[neuron_class.Neuron() for _ in range(5)],
    ) for i in range(8)]

    layers += [neuron_layer_classes.OutputNeuronLayer(
        size=-1,
        neurons=[neuron_class.Neuron() for _ in range(5)],
    )]

    # Set previous layers
    for i in range(1, 10):
        layers[i].previous_layer = layers[i - 1]

    # Set next layers
    for i in range(0, 9):
        layers[i].next_layer = layers[i + 1]

    # Set up network
    network = network_class.Network(layers=layers)

    return {
        "network": network,
        "layers": layers,
        "layer_0": layers[0],
        "layer_1": layers[1],
        "layer_2": layers[2],
        "layer_3": layers[3],
        "layer_4": layers[4],
        "layer_5": layers[5],
        "layer_6": layers[6],
        "layer_7": layers[7],
        "layer_8": layers[8],
        "layer_9": layers[9],
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
    network = network_class.Network(
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

    network = network_class.Network(layers = [input_layer, hidden_layer, output_layer])
    cost: float = network.cost_function(desired_activation_values=[1.0])

    assert cost == 0.0


def test_cost_function_with_input_data(single_neuron_network):
    """Basic cost function execution but activation is calculated."""

    input_layer = single_neuron_network['input_layer']
    hidden_layer = single_neuron_network['hidden_layer']
    output_layer = single_neuron_network['output_layer']

    input_layer.weights = [[1.0] for _ in range(1)]
    hidden_layer.weights = [[1.0] for _ in range(1)]

    network = network_class.Network(layers = [input_layer, hidden_layer, output_layer])

    # Activation for the three neurons should be 1.0 each
    cost: float = network.cost_function(desired_activation_values=[1.0], input_data=[1.0])

    assert cost == 0.0


def test_cost_function_with_input_data_ten_neurons(ten_neuron_network):
    input_layer = ten_neuron_network['input_layer']
    hidden_layer = ten_neuron_network['hidden_layer']
    output_layer = ten_neuron_network['output_layer']

    for neuron in output_layer.neurons:
        neuron.activation = 1
    
    network = network_class.Network(layers=[input_layer, hidden_layer, output_layer])
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

    network = network_class.Network(layers = [input_layer, hidden_layer, output_layer])
    network.print()

    with pytest.raises(neuron_class.IncorrectInputError):
        network.cost_function(desired_activation_values=([0] * (INCORRECT_NUM_OUTPUT_NEURONS)))


def test_backpropagate_weights_decreases_cost(five_neuron_network):
    input_layer = five_neuron_network['input_layer']
    hidden_layer = five_neuron_network['hidden_layer']
    output_layer = five_neuron_network['output_layer']

    # Set up predictable weights
    hidden_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
    input_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]

    network = network_class.Network([input_layer, hidden_layer, output_layer])

    network.print()
    old_cost = network.cost_function(
        desired_activation_values=[2.0, 0.0, 0.0, 0.0, 0.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    network.backpropagate_weights([2.0, 0.0, 0.0, 0.0, 0.0])
    network.print()

    new_cost = network.cost_function(
        desired_activation_values=[2.0, 0.0, 0.0, 0.0, 0.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert new_cost < old_cost


def test_backpropagate_weights_repeatedly_decreases_cost(five_neuron_network):
    input_layer = five_neuron_network['input_layer']
    hidden_layer = five_neuron_network['hidden_layer']
    output_layer = five_neuron_network['output_layer']

    # Set up predictable weights
    hidden_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
    input_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]

    network = network_class.Network([input_layer, hidden_layer, output_layer])

    old_cost = network.cost_function(
        desired_activation_values=[2.0, 0.0, 0.0, 0.0, 0.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )

    for i in range(100):
        network.backpropagate_weights([2.0, 0.0, 0.0, 0.0, 0.0])
        
        new_cost = network.cost_function(
            desired_activation_values=[2.0, 0.0, 0.0, 0.0, 0.0],
            input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
        )

        assert new_cost < old_cost
        old_cost = new_cost


def test_backpropagate_weights_ten_layers(ten_layer_network):
    """
    This fails as the learning rate needs to be very precise (a particular number of dps) which is different to the
    learning rate required for networks with smaller numbers of layers.
    
    The reason for this high precision is because activation values increase rapidly as the number of layers increases.
    
    To remedy this, the sigmoid function would force activation values to be within 0 and 1 so a consistent learning
    rate can be applied to networks of varying layers. This would affect the activation process and backpropagation (a
    derivative to the sigmoid function must be applied). TBD.
    """
    
    network = ten_layer_network['network']

    for layer in network.layers:
        layer.weights = [[1.0] * 5] * 10
        for neuron in layer.neurons:
            neuron.bias = 0.0

    old_cost = network.cost_function(
        desired_activation_values=[0.0, 0.0, 0.0, 0.0, 0.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    

    network.backpropagate_weights([0.0, 0.0, 0.0, 0.0, 0.0])
    new_cost = network.cost_function(
        desired_activation_values=[0.0, 0.0, 0.0, 0.0, 0.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )

    print(f"old_cost={old_cost}")
    print(f"new_cost={new_cost}")

    assert new_cost < old_cost
    assert False


def test_backpropagate_weights_with_zero_cost(five_neuron_network):
    """Test backpropagate_weights doesn't change weights if cost is zero."""
    input_layer = five_neuron_network['input_layer']
    hidden_layer = five_neuron_network['hidden_layer']
    output_layer = five_neuron_network['output_layer']
    
    # Set up weights and activations
    hidden_layer.weights = [[1, 1, 1, 1, 1] for _ in range(5)]
    input_layer.weights = [[1, 1, 1, 1, 1] for _ in range(5)]
    for neuron in hidden_layer.neurons + output_layer.neurons:
        neuron.activation = 1
    
    network = network_class.Network([input_layer, hidden_layer, output_layer])

    old_cost = network.cost_function(
        desired_activation_values=[1.0, 1.0, 1.0, 1.0, 1.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    activation_values_after_providing_input_data = network.output_layer.activations

    network.backpropagate_weights(desired_outputs=activation_values_after_providing_input_data)

    new_cost = network.cost_function(
        desired_activation_values=[1.0, 1.0, 1.0, 1.0, 1.0],
        input_data=[1.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert new_cost == old_cost


def test_backpropagate_multiple_examples(five_neuron_network):
    """This trains the network over 5 different sets of input data and tests that costs are generated at the end."""

    input_layer = five_neuron_network['input_layer']
    hidden_layer = five_neuron_network['hidden_layer']
    output_layer = five_neuron_network['output_layer']

    # Set up predictable weights
    hidden_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
    input_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]

    # Set up training data - contains data to pass into the network and contains desired activation values
    training_data = {
        'input_data': [
            [3.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 3.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 3.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 3.0],
        ],
        'desired_activation_values': [
            [3.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 3.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 3.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 3.0],
        ]
    }
    network = network_class.Network([input_layer, hidden_layer, output_layer])

    for i in range(5):
        old_cost = network.cost_function(
            input_data=training_data['input_data'][i],
            desired_activation_values=training_data['desired_activation_values'][i],
        )
        for _ in range(20):
            new_cost = network.cost_function(
                input_data=training_data['input_data'][i],
                desired_activation_values=training_data['desired_activation_values'][i],
            )
            network.backpropagate_weights(training_data['desired_activation_values'][i])
            print(f"old_cost={old_cost}")
            print(f"new_cost={new_cost}")

            old_cost = new_cost

    network.print()
    training_costs = [network.cost_function(
        input_data=training_data['input_data'][i],
        desired_activation_values=training_data['desired_activation_values'][i],
    ) for i in range(5)]

    for costs in training_costs:
        print(f"cost={sum(costs)}")

    assert len(costs) == 5

