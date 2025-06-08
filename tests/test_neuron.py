import unittest
import src.neuron_classes as neuron_classes


# class TestNeuron(unittest.TestCase):
#     """Tests for the `Neuron` class."""

#     def setUp(self):
#         self.neuron = neuron_classes.Neuron()
#         self.forward_neuron = neuron_classes.Neuron()

#     def test_repeat_init(self):
#         self.neuron.set_bias(5)
#         self.neuron.set_weights({self.forward_neuron: 5})
#         self.neuron.set_weight(self.forward_neuron, 5)
#         self.neuron = neuron_classes.Neuron()
#         assert self.neuron.get_bias() == 0
#         assert self.neuron.get_weights() == {}
#         assert self.forward_neuron.get_weights() == {}

#     def test_bias(self):
#         self.neuron.set_bias(5)
#         assert self.neuron.get_bias() == 5
#         assert self.forward_neuron.get_bias() == 0

#     def test_initial_weights(self):
#         assert self.neuron.get_weights() == {}
#         assert self.forward_neuron.get_weights() == {}

#     def test_weights(self):
#         assert self.forward_neuron.get_weights() == {}
#         self.neuron.set_weights({self.forward_neuron: 5})
#         assert self.neuron.get_weights() == {self.forward_neuron: 5}
#         assert self.forward_neuron.get_weights() == {}

#     def test_weight(self):
#         self.neuron.set_weight(self.forward_neuron, 5)
#         assert self.neuron.get_weights() == {self.forward_neuron: 5}
#         assert self.forward_neuron.get_weights() == {}

#     def test_activation(self):
#         self.neuron.set_activation(5)
#         assert self.neuron.get_activation() == 5
#         assert self.forward_neuron.get_activation() == 0


# class TestNeuronLayer(unittest.TestCase):
#     """Tests for the `NeuronLayer` class."""

#     def setUp(self):
#         self.initial_neuron = neuron_classes.Neuron()
#         self.neuron = neuron_classes.Neuron()
#         self.forward_neuron = neuron_classes.Neuron()

#         self.forward_neuron_layer = neuron_classes.NeuronLayer(
#             size=-1,
#             neurons=[self.forward_neuron],
#         )
#         self.neuron_layer = neuron_classes.NeuronLayer(
#             size=-1,
#             neurons=[self.neuron],
#             next_layer=self.forward_neuron_layer,
#         )
#         self.initial_neuron_layer = neuron_classes.NeuronLayer(
#             size=-1,
#             neurons=[self.initial_neuron],
#             next_layer=self.neuron_layer,
#             initial_layer=True,
#         )

#     def test_get_neurons(self):
#         assert self.initial_neuron_layer.get_neurons() == [self.initial_neuron]
#         assert self.neuron_layer.get_neurons() == [self.neuron]
#         assert self.forward_neuron_layer.get_neurons() == [self.forward_neuron]

#     def test_get_next_layer(self):
#         assert self.initial_neuron_layer.get_next_layer() == self.neuron_layer
#         assert self.neuron_layer.get_next_layer() == self.forward_neuron_layer
#         assert self.forward_neuron_layer.get_next_layer() is None

#     def test_get_biases(self):
#         self.initial_neuron.set_bias(1)
#         self.neuron.set_bias(2)
#         self.forward_neuron.set_bias(3)

#         assert self.initial_neuron_layer.get_biases() == [1]
#         assert self.neuron_layer.get_biases() == [2]
#         assert self.forward_neuron_layer.get_biases() == [3]

#     def test_activate_initial_layer(self):
#         self.initial_neuron_layer.activate_initial_layer([])
#         assert self.initial_neuron.get_activation() == 0

#         self.initial_neuron_layer.activate_initial_layer([5, 5])
#         assert self.initial_neuron.get_activation() == 0

#         self.initial_neuron_layer.activate_initial_layer([5])
#         assert self.initial_neuron.get_activation() == 5

#         self.neuron_layer.activate_initial_layer([5])
#         self.forward_neuron_layer.activate_initial_layer([5])
#         assert self.neuron.get_activation() == 0
#         assert self.forward_neuron.get_activation() == 0

#     def test_neuron_layer_activate_next_layer(self):
#         """TODO"""


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

    def test_get_initial_layer(self):
        assert self.neural_network.get_initial_layer() == self.initial_neuron_layer

    def test_get_output_layer(self):
        assert self.neural_network.get_output_layer() == self.forward_neuron_layer

    def test_get_layers(self):
        assert self.neural_network.get_layers() == [
            self.initial_neuron_layer,
            self.neuron_layer,
            self.forward_neuron_layer,
        ]

    def test_think(self):
        """TODO"""

    def test_train(self):
        """TODO"""


if __name__ == "__main__":
    unittest.main()
