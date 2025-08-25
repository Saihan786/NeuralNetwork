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
def forward_neuron_layer(forward_neuron):
    return neuron_classes.NeuronLayer(
        size=-1,
        neurons=[forward_neuron],
    )


@pytest.fixture
def neuron_layer(neuron, forward_neuron_layer):
    return neuron_classes.NeuronLayer(
        size=-1,
        neurons=[neuron],
        next_layer=forward_neuron_layer,
    )


@pytest.fixture
def initial_neuron_layer(initial_neuron, neuron_layer):
    return neuron_classes.NeuronLayer(
        size=-1,
        neurons=[initial_neuron],
        next_layer=neuron_layer,
        initial_layer=True,
    )


def test_initialise_with_size():
    five_neuron_layer = neuron_classes.NeuronLayer(size=5)
    assert len(five_neuron_layer.neurons) == 5


def test_get_neurons(initial_neuron_layer, neuron_layer, forward_neuron_layer, initial_neuron, neuron, forward_neuron):
    assert initial_neuron_layer.neurons == [initial_neuron]
    assert neuron_layer.neurons == [neuron]
    assert forward_neuron_layer.neurons == [forward_neuron]


def test_get_next_layer(initial_neuron_layer, neuron_layer, forward_neuron_layer):
    assert initial_neuron_layer.next_layer == neuron_layer
    assert neuron_layer.next_layer == forward_neuron_layer
    assert forward_neuron_layer.next_layer is None


def test_get_biases(initial_neuron_layer, neuron_layer, forward_neuron_layer, initial_neuron, neuron, forward_neuron):
    initial_neuron.bias = 1
    neuron.bias = 2
    forward_neuron.bias = 3

    assert initial_neuron_layer.biases == [1]
    assert neuron_layer.biases == [2]
    assert forward_neuron_layer.biases == [3]


def test_activate_initial_layer(
    initial_neuron_layer, neuron_layer, forward_neuron_layer, initial_neuron, neuron, forward_neuron
):
    initial_neuron_layer.activate_initial_layer([])
    assert initial_neuron.activation == 0

    initial_neuron_layer.activate_initial_layer([5, 5])
    assert initial_neuron.activation == 0

    initial_neuron_layer.activate_initial_layer([5])
    assert initial_neuron.activation == 5

    neuron_layer.activate_initial_layer([5])
    forward_neuron_layer.activate_initial_layer([5])

    assert neuron.activation == 0
    assert forward_neuron.activation == 0


def test_get_activations(neuron_layer, neuron):
    neuron.activation = 5
    assert neuron_layer.activations == [5]


def test_set_biases(neuron_layer, neuron):
    neuron_layer.biases = [5]
    assert neuron_layer.biases == [5]


def test_set_weights(neuron_layer, neuron, forward_neuron):
    neuron_layer.weights = [[5]]
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
