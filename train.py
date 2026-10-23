import numpy as np
import random

class neuron:
    def __init__(self, num_weights):
        self.weights = np.array([random.random() for i in range(num_weights)])

    def forward(self, x):
        total = 0
        for i in range(x.size):
            total += self.weights[i] * x[i]
        return max(total, 0)

class hidden_layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, x):
        y = np.array([])
        for neuron in self.neurons:
            y = np.append(y, neuron.forward(x))
        return y

class neural_network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

def main():
    test_neuron()
    test_hidden_layer()
    test_neural_network()
    print("all tests work")
"""
    inputs = np.array([1, 2, 5, 2, 0])
    neurons = np.array([neuron(5) for i in range(10)])
    layer1 = hidden_layer(neurons)
    layers = np.array([layer1])
    nn = neural_network(layers)
    sum  = nn.forward(inputs).sum()
    print(sum)
"""

def test_neural_network():
    n1 = neuron(2)
    n1.weights = np.array([1.0, 2.0])

    n2 = neuron(2)
    n2.weights = np.array([-1.0, 1.0])

    n3 = neuron(2)
    n3.weights = np.array([2.0, 3.0])

    layer1 = hidden_layer([n1, n2])
    layer2 = hidden_layer([n3])
    nn = neural_network([layer1, layer2])

    inputs = np.array([2.0, 3.0])

    # Layer 1: [max(2 + 6, 0), max(-2 + 3, 0)] = [8, 1]
    # Layer 2: [max(2*8 + 3*1, 0)] = [19]
    expected = np.array([19.0])
    actual = nn.forward(inputs)

    assert actual.shape == expected.shape, (
        f"Expected shape {expected.shape}, got {actual.shape}"
    )
    assert np.allclose(actual, expected), (
        f"Expected {expected}, got {actual}"
    )


def test_hidden_layer():
    inputs = np.array([1, 2])
    n1 = neuron(2)
    n1.weights = np.array([1.0, 2.0])
    n2 = neuron(2)
    n2.weights = np.array([0.0, 4.0])
    neurons1 = np.array([n1, n2])

    layer1 = hidden_layer(neurons1)


    actual = layer1.forward(inputs).sum()
    expected = 13.0

    assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"

def test_neuron():
    n = neuron(2)
    n.weights = np.array([2.0, 3.0])
    inputs = np.array([1.0, 2.0])

    actual = n.forward(inputs)
    expected = 8.0  # 2*1 + 3*2

    assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"
if __name__ == "__main__":
    main()

    