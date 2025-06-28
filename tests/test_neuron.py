import unittest
import src.neuron_classes as neuron_classes


class TestNeuron(unittest.TestCase):
    """Tests for the `Neuron` class."""

    def setUp(self):
        self.neuron = neuron_classes.Neuron()
        self.forward_neuron = neuron_classes.Neuron()

    def test_repeat_init(self):
        self.neuron.bias = 5
        self.neuron.weights = {self.forward_neuron: 5}
        self.neuron.weights[self.forward_neuron] = 5
        self.neuron = neuron_classes.Neuron()
        assert self.neuron.bias == 0
        assert self.neuron.weights == {}
        assert self.forward_neuron.weights == {}

    def test_bias(self):
        self.neuron.bias = 5
        assert self.neuron.bias == 5
        assert self.forward_neuron.bias == 0

    def test_initial_weights(self):
        assert self.neuron.weights == {}
        assert self.forward_neuron.weights == {}

    def test_weights(self):
        assert self.forward_neuron.weights == {}
        self.neuron.weights = {self.forward_neuron: 5}
        assert self.neuron.weights == {self.forward_neuron: 5}
        assert self.forward_neuron.weights == {}

    def test_weight(self):
        self.neuron.weights[self.forward_neuron] = 5
        assert self.neuron.weights == {self.forward_neuron: 5}
        assert self.forward_neuron.weights == {}

    def test_activation(self):
        self.neuron.activation = 5
        assert self.neuron.activation == 5
        assert self.forward_neuron.activation == 0


class TestNeuronLayer(unittest.TestCase):
    """Tests for the `NeuronLayer` class."""

    def setUp(self):
        self.initial_neuron = neuron_classes.Neuron()
        self.neuron = neuron_classes.Neuron()
        self.forward_neuron = neuron_classes.Neuron()

        self.forward_neuron_layer = neuron_classes.NeuronLayer(
            size=-1,
            neurons=[self.forward_neuron],
        )
        self.neuron_layer = neuron_classes.NeuronLayer(
            size=-1,
            neurons=[self.neuron],
            next_layer=self.forward_neuron_layer,
        )
        self.initial_neuron_layer = neuron_classes.NeuronLayer(
            size=-1,
            neurons=[self.initial_neuron],
            next_layer=self.neuron_layer,
            initial_layer=True,
        )

    def test_initialise_with_size(self):
        five_neuron_layer = neuron_classes.NeuronLayer(size=5)
        assert len(five_neuron_layer.neurons) == 5

    def test_get_neurons(self):
        assert self.initial_neuron_layer.neurons == [self.initial_neuron]
        assert self.neuron_layer.neurons == [self.neuron]
        assert self.forward_neuron_layer.neurons == [self.forward_neuron]

    def test_get_next_layer(self):
        assert self.initial_neuron_layer.next_layer == self.neuron_layer
        assert self.neuron_layer.next_layer == self.forward_neuron_layer
        assert self.forward_neuron_layer.next_layer is None

    def test_get_biases(self):
        self.initial_neuron.bias = 1
        self.neuron.bias = 2
        self.forward_neuron.bias = 3

        assert self.initial_neuron_layer.biases == [1]
        assert self.neuron_layer.biases == [2]
        assert self.forward_neuron_layer.biases == [3]

    def test_activate_initial_layer(self):
        self.initial_neuron_layer.activate_initial_layer([])
        assert self.initial_neuron.activation == 0

        self.initial_neuron_layer.activate_initial_layer([5, 5])
        assert self.initial_neuron.activation == 0

        self.initial_neuron_layer.activate_initial_layer([5])
        assert self.initial_neuron.activation == 5

        self.neuron_layer.activate_initial_layer([5])
        self.forward_neuron_layer.activate_initial_layer([5])
        assert self.neuron.activation == 0
        assert self.forward_neuron.activation == 0

    def test_get_activations(self):
        self.neuron.activation = 5
        assert self.neuron_layer.activations == [5]

    def test_set_biases(self):
        self.neuron_layer.biases = [5]
        assert self.neuron_layer.biases == [5]
        assert self.neuron.bias == 5

    def test_set_weights(self):
        self.neuron_layer.weights = [[5]]
        assert self.neuron.weights == {self.forward_neuron: 5}

    def test_neuron_layer_activate_next_layer(self):
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


class TestNeuralNetwork(unittest.TestCase):
    """Tests for the `Network` class."""

    def setUp(self):
        self.initial_neuron = neuron_classes.Neuron()
        self.neuron = neuron_classes.Neuron()
        self.forward_neuron = neuron_classes.Neuron()

        self.forward_neuron_layer = neuron_classes.NeuronLayer(
            size=-1,
            neurons=[self.forward_neuron],
        )
        self.neuron_layer = neuron_classes.NeuronLayer(
            size=-1,
            neurons=[self.neuron],
            next_layer=self.forward_neuron_layer,
        )
        self.initial_neuron_layer = neuron_classes.NeuronLayer(
            size=-1,
            neurons=[self.initial_neuron],
            next_layer=self.neuron_layer,
            initial_layer=True,
        )

        self.neural_network = neuron_classes.Network(
            layers=[
                self.initial_neuron_layer,
                self.neuron_layer,
                self.forward_neuron_layer,
            ]
        )

    def test_get_layers(self):
        assert self.neural_network.layers == [
            self.initial_neuron_layer,
            self.neuron_layer,
            self.forward_neuron_layer,
        ]

    def test_activate_layers(self):
        self.initial_neuron.weights = {self.neuron: 1}

        self.neuron.bias = 5
        self.neuron.weights = {self.forward_neuron: 1}

        self.forward_neuron.bias = 5

        assert self.neural_network.activate_layers([10]) == [20]
        assert self.neural_network.output_layer.activations == [20]

        self.initial_neuron.weights = {self.neuron: 2}
        self.neuron.weights = {self.forward_neuron: 2}

        assert self.neural_network.activate_layers([10]) == [55]
        assert self.neural_network.output_layer.activations == [55]

    def test_cost_function_no_input_data(self):
        """
        TODO:
            - Use conftest to establish a neural network that already has
            weights and biases.
                - Test `network.activate_layers()` separately.
        """

        self.forward_neuron.activation = 10

        assert self.neural_network.cost_function(desired_output=[10]) == [0]
        assert self.neural_network.cost_function(desired_output=[20]) == [100]

    def test_cost_function_with_input_data(self):
        """
        TODO:
            - Use conftest to establish a neural network that already has
            weights and biases.
                - Test `network.activate_layers()` separately.
        """

        OUTPUT_ACTIVATION_AFTER_INPUT_DATA = 142
        self.initial_neuron.bias = 1
        self.initial_neuron.weights[self.neuron] = 2

        self.neuron.bias = 10
        self.neuron.weights[self.forward_neuron] = 11

        self.forward_neuron.bias = 10
        self.forward_neuron.activation = 10

        assert self.neural_network.cost_function(desired_output=[20]) == [100]

        applied_new_activation = self.neural_network.cost_function(
            desired_output=[20], input_data=[1]
        )

        self.forward_neuron.activation = OUTPUT_ACTIVATION_AFTER_INPUT_DATA
        assert applied_new_activation == self.neural_network.cost_function(
            desired_output=[20]
        )

    def test_think(self):
        """TODO"""

    def test_train(self):
        """TODO"""


if __name__ == "__main__":
    unittest.main()
