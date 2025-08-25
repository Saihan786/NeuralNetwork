import pytest
from neural_network import neuron_classes as neuron_classes


@pytest.fixture
def neuron():
    return neuron_classes.Neuron()


@pytest.fixture
def forward_neuron():
    return neuron_classes.Neuron()


def test_repeat_init(forward_neuron):
    neuron = neuron_classes.Neuron()
    neuron.bias = 5
    neuron.weights = {forward_neuron: 5}
    neuron.weights[forward_neuron] = 5
    neuron = neuron_classes.Neuron()
    assert neuron.bias == 0
    assert neuron.weights == {}
    assert forward_neuron.weights == {}


def test_bias(neuron, forward_neuron):
    neuron.bias = 5
    assert neuron.bias == 5
    assert forward_neuron.bias == 0


def test_initial_weights(neuron, forward_neuron):
    assert neuron.weights == {}
    assert forward_neuron.weights == {}


def test_weights(neuron, forward_neuron):
    assert forward_neuron.weights == {}
    neuron.weights = {forward_neuron: 5}
    assert neuron.weights == {forward_neuron: 5}
    assert forward_neuron.weights == {}


def test_weight(neuron, forward_neuron):
    neuron.weights[forward_neuron] = 5
    assert neuron.weights == {forward_neuron: 5}
    assert forward_neuron.weights == {}


def test_activation(neuron, forward_neuron):
    neuron.activation = 5
    assert neuron.activation == 5
    assert forward_neuron.activation == 0
