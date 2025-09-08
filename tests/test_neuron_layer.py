import pytest
from neural_network import neuron_classes as neuron_classes


@pytest.fixture
def initial_neuron():
    return neuron_classes.Neuron()


@pytest.fixture
def neuron():
    return neuron_classes.Neuron()


@pytest.fixture
def forward_neuron():
    return neuron_classes.Neuron()


@pytest.fixture
def forward_neuron_layer_small(forward_neuron):
    return neuron_classes.NeuronLayer(
        size=-1,
        neurons=[forward_neuron],
    )


@pytest.fixture
def neuron_layer_small(neuron, forward_neuron_layer_small):
    return neuron_classes.NeuronLayer(
        size=-1,
        neurons=[neuron],
        next_layer=forward_neuron_layer_small,
    )


@pytest.fixture
def initial_neuron_layer_small(initial_neuron, neuron_layer_small):
    return neuron_classes.NeuronLayer(
        size=-1,
        neurons=[initial_neuron],
        next_layer=neuron_layer_small,
        initial_layer=True,
    )


@pytest.fixture
def small_neuron_layers(forward_neuron_layer_small, neuron_layer_small, initial_neuron_layer_small):
    return {
        'forward_neuron_layer_small': forward_neuron_layer_small,
        'neuron_layer_small': neuron_layer_small,
        'initial_neuron_layer_small': initial_neuron_layer_small,
    }


@pytest.fixture
def output_neuron_layer_medium():
    return neuron_classes.NeuronLayer(
        size=5,
    )


@pytest.fixture
def hidden_neuron_layer_medium(output_neuron_layer_medium):
    return neuron_classes.NeuronLayer(
        size=5,
        next_layer=output_neuron_layer_medium,
    )


@pytest.fixture
def input_neuron_layer_medium(hidden_neuron_layer_medium):
    return neuron_classes.NeuronLayer(
        size=5,
        next_layer=hidden_neuron_layer_medium,
        initial_layer=True,
    )


@pytest.fixture
def neuron_layers_size_5(output_neuron_layer_medium, hidden_neuron_layer_medium, input_neuron_layer_medium):
    output_neuron_layer_medium.previous_layer = hidden_neuron_layer_medium
    hidden_neuron_layer_medium.previous_layer = input_neuron_layer_medium
    
    return {
        'output_neuron_layer_medium': output_neuron_layer_medium,
        'hidden_neuron_layer_medium': hidden_neuron_layer_medium,
        'input_neuron_layer_medium': input_neuron_layer_medium,
    }


def test_initialise_with_size():
    five_neuron_layer = neuron_classes.NeuronLayer(size=5)
    assert len(five_neuron_layer.neurons) == 5


def test_get_neurons(initial_neuron_layer_small, neuron_layer_small, forward_neuron_layer_small, initial_neuron, neuron, forward_neuron):
    assert initial_neuron_layer_small.neurons == [initial_neuron]
    assert neuron_layer_small.neurons == [neuron]
    assert forward_neuron_layer_small.neurons == [forward_neuron]


def test_get_next_layer(initial_neuron_layer_small, neuron_layer_small, forward_neuron_layer_small):
    assert initial_neuron_layer_small.next_layer == neuron_layer_small
    assert neuron_layer_small.next_layer == forward_neuron_layer_small
    assert forward_neuron_layer_small.next_layer is None


def test_get_biases(initial_neuron_layer_small, neuron_layer_small, forward_neuron_layer_small, initial_neuron, neuron, forward_neuron):
    initial_neuron.bias = 1
    neuron.bias = 2
    forward_neuron.bias = 3

    assert initial_neuron_layer_small.biases == [1]
    assert neuron_layer_small.biases == [2]
    assert forward_neuron_layer_small.biases == [3]


def test_activate_initial_layer(
    initial_neuron_layer_small, neuron_layer_small, forward_neuron_layer_small, initial_neuron, neuron, forward_neuron
):
    initial_neuron_layer_small.activate_initial_layer([])
    assert initial_neuron.activation == 0

    initial_neuron_layer_small.activate_initial_layer([5, 5])
    assert initial_neuron.activation == 0

    initial_neuron_layer_small.activate_initial_layer([5])
    assert initial_neuron.activation == 5

    neuron_layer_small.activate_initial_layer([5])
    forward_neuron_layer_small.activate_initial_layer([5])

    assert neuron.activation == 0
    assert forward_neuron.activation == 0


def test_get_activations(neuron_layer_small, neuron):
    neuron.activation = 5
    assert neuron_layer_small.activations == [5]


def test_set_biases(neuron_layer_small, neuron):
    neuron_layer_small.biases = [5]
    assert neuron_layer_small.biases == [5]


def test_set_weights(neuron_layer_small, neuron, forward_neuron):
    neuron_layer_small.weights = [[5]]
    assert neuron.weights == {forward_neuron: 5}


def test_neuron_layer_activate_next_layer():
    size = 3
    fneurons = [neuron_classes.Neuron(bias=i) for i in range(1, size + 1)]
    forward_large_layer = neuron_classes.NeuronLayer(
        size=size,
        neurons=fneurons,
    )
    assert forward_large_layer.biases == [1, 2, 3]

    large_layer = neuron_classes.NeuronLayer(
        size=size,
        next_layer=forward_large_layer,
    )

    inc = 1
    for neuron in large_layer.neurons:
        neuron.activation = inc
        neuron.weights = {
            forward_large_layer.neurons[0]: 1,
            forward_large_layer.neurons[1]: 2,
            forward_large_layer.neurons[2]: 3,
        }
        inc += 1
    assert [list(n.weights.values()) for n in large_layer.neurons] == [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ]

    large_layer.activate_next_layer()
    assert forward_large_layer.activations == [7, 14, 21]


def test_proportional_changes(neuron_layers_size_5):
    current_layer: neuron_classes.NeuronLayer = neuron_layers_size_5['output_neuron_layer_medium']
    previous_layer: neuron_classes.NeuronLayer = neuron_layers_size_5['hidden_neuron_layer_medium']

    previous_layer.weights = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
    for neuron in previous_layer.neurons + current_layer.neurons:
        neuron.activation = 1
    
    cost = 10
    
    assert False

    # cneurons_to_changes = current_layer.proportional_changes(costs=costs)
