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
def forward_neuron_layer_medium():
    return neuron_classes.NeuronLayer(
        size=5,
    )


@pytest.fixture
def neuron_layer_medium(forward_neuron_layer_medium):
    return neuron_classes.NeuronLayer(
        size=5,
        next_layer=forward_neuron_layer_medium,
    )


@pytest.fixture
def initial_neuron_layer_medium(neuron_layer_medium):
    return neuron_classes.NeuronLayer(
        size=5,
        next_layer=neuron_layer_medium,
        initial_layer=True,
    )


@pytest.fixture
def medium_neuron_layers(forward_neuron_layer_medium, neuron_layer_medium, initial_neuron_layer_medium):
    forward_neuron_layer_medium.previous_layer = neuron_layer_medium
    neuron_layer_medium.previous_layer = initial_neuron_layer_medium
    
    return {
        'forward_neuron_layer_medium': forward_neuron_layer_medium,
        'neuron_layer_medium': neuron_layer_medium,
        'initial_neuron_layer_medium': initial_neuron_layer_medium,
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


@pytest.mark.parametrize(
    ["weights", "activations", "costs", "expected_changes_for_pneurons"],
    [
        (
            # basic example
            [
                [100, 0, 0, 0, 0],
                [0, 100, 0, 0, 0],
                [0, 0, 100, 0, 0],
                [0, 0, 0, 100, 0],
                [0, 0, 0, 0, 100],
            ],                          # weights from previous layer to current layer
            [1, 1, 1, 1, 1],            # activations of previous layer neurons
            [100, 0, 0, 0, 0],          # costs of current layer neurons
            [
                [-100, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],                          # only cneuron_1 wants pneuron_1 to change
        ),

        (
            # check that pneurons with larger weight are given higher change values
            [
                [100, 0, 0, 0, 0],
                [10, 0, 0, 0, 0],
                [20, 0, 0, 0, 0],
                [30, 0, 0, 0, 0],
                [40, 0, 0, 0, 0],
            ],                          # each previous neuron is only tied to the first current neuron
            [1, 1, 1, 1, 1],            # activations of previous layer neurons
            [100, 0, 0, 0, 0],          # costs of current layer neurons
            [
                [-50.0, -5.0, -10.0, -15.0, -20.0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],                          # cneuron_1 wants some pneurons to change more than others based on weight
        ),

        (
            # check that pneurons with larger activation are given higher change values
            [
                [100, 0, 0, 0, 0],
                [100, 0, 0, 0, 0],
                [100, 0, 0, 0, 0],
                [100, 0, 0, 0, 0],
                [100, 0, 0, 0, 0],
            ],                          # each previous neuron is only tied to the first current neuron, each by the same amount
            [100, 10, 20, 30, 40],            # activations of previous layer neurons
            [100, 0, 0, 0, 0],          # costs of current layer neurons
            [
                [-50.0, -5.0, -10.0, -15.0, -20.0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],                          # cneuron_1 wants some pneurons to change more than others based on activation
        ),

        (
            # check that cneurons with no connections do not want any change if their cost is 0
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],                          # no connections between neurons
            [10, 1, 2, 3, 4],           # no activations
            [100, 0, 0, 0, 0],          # costs of current layer neurons
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],                          # each cneuron wants a change of 0 for each pneuron
        ),
        (
            # multiple costs
            [
                [50, 50, 0, 0, 0],
                [0, 50, 50, 0, 0],
                [0, 0, 50, 50, 0],
                [0, 0, 0, 50, 50],
                [50, 0, 0, 0, 50],
            ],                          # pneuron_1 and pneuron_5 are both connected to cneuron_1
            [0.5, 0.5, 0.5, 0.5, 0.5],  # activations of previous layer neurons
            [50, 50, 0, 0, 0],          # costs of current layer neurons
            [
                [-25.0, -0.0, -0.0, -0.0, -25.0],
                [-25.0, -25.0, -0.0, -0.0, -0.0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],                          # cneuron_1 and cneuron_2 both want some pneurons to change
        ),
    ]
)
def test_proportional_changes(medium_neuron_layers, weights, activations, costs, expected_changes_for_pneurons):
    current_layer: neuron_classes.NeuronLayer = medium_neuron_layers['forward_neuron_layer_medium']
    previous_layer: neuron_classes.NeuronLayer = medium_neuron_layers['neuron_layer_medium']

    previous_layer.weights = weights

    for i in range(len(previous_layer.neurons)):
        pneuron = previous_layer.neurons[i]
        pneuron.activation = activations[i]

    cneurons_to_changes = current_layer.proportional_changes(costs=costs)

    assert list(cneurons_to_changes.values()) == expected_changes_for_pneurons
