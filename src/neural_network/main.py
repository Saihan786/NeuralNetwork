"""Here, you can make your own neural network and train it on some data."""

from neural_network import neuron_class, neuron_layer_classes, network_class
import random


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


def initialise_weights(layer: neuron_layer_classes.BaseNeuronLayer):
    layer.weights = [
        [0.23, -0.41, 0.67, -0.12, 0.89],
        [-0.34, 0.78, -0.56, 0.91, -0.23],
        [0.45, -0.67, 0.12, -0.89, 0.34],
        [-0.78, 0.56, -0.91, 0.23, -0.45],
        [0.67, -0.12, 0.89, -0.34, 0.78]
    ]


if __name__ == "__main__":
    network = ten_layer_network()['network']

    for layer in network.layers:
        initialise_weights(layer)

    # For 5-neuron layers, try these small random biases:
    for layer in network.layers[1:]:
        layer.biases = [random.uniform(-0.1, 0.1) for _ in range(5)]

    # Set up training data - contains data to pass into the network and contains desired activation values
    training_data = {
        'input_data': [
            # Basic vectors
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            
            # Two-element combinations
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.6],
            
            # Three-element combinations
            [0.2, 0.3, 0.5, 0.0, 0.0],
            [0.1, 0.4, 0.5, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.5, 0.0],
            [0.0, 0.0, 0.3, 0.4, 0.3],
            
            # More diverse patterns
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.9],
            
            # Uniform distributions
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.15, 0.25, 0.2, 0.25, 0.15],
            
            # Edge cases with small values
            [0.05, 0.0, 0.0, 0.0, 0.95],
            [0.95, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.95, 0.05, 0.0],
        ],
        'desired_activation_values': [
            # Same as inputs (identity mapping)
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.6],
            [0.2, 0.3, 0.5, 0.0, 0.0],
            [0.1, 0.4, 0.5, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.5, 0.0],
            [0.0, 0.0, 0.3, 0.4, 0.3],
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.9],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.15, 0.25, 0.2, 0.25, 0.15],
            [0.05, 0.0, 0.0, 0.0, 0.95],
            [0.95, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.95, 0.05, 0.0],
        ]
    }

    network.print()
    training_costs = [network.cost_function(
        input_data=training_data['input_data'][i],
        desired_activation_values=training_data['desired_activation_values'][i],
    ) for i in range(len(training_data['input_data']))]

    for costs in training_costs:
        print(f"cost before training = {sum(costs)}")
    
    for j in range(20):
        print(f"epoch={j}")

        for i in range(len(training_data['input_data'])):
            old_cost = network.cost_function(
                input_data=training_data['input_data'][i],
                desired_activation_values=training_data['desired_activation_values'][i],
            )
            for _ in range(50):
                new_cost = network.cost_function(
                    input_data=training_data['input_data'][i],
                    desired_activation_values=training_data['desired_activation_values'][i],
                )
                network.backpropagate_weights(training_data['desired_activation_values'][i])
                old_cost = new_cost

    network.print()
    training_costs = [network.cost_function(
        input_data=training_data['input_data'][i],
        desired_activation_values=training_data['desired_activation_values'][i],
    ) for i in range(len(training_data['input_data']))]

    for costs in training_costs:
        print(f"cost={sum(costs)}")
