import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):
        """Initialize a simple feedforward neural network.

        Network structure: input -> hidden -> output
        Activation: tanh on both layers

        Args:
            input_size: Dimension of input (observation size)
            output_size: Dimension of output (action size)
            hidden_size: Number of hidden neurons
        """
        # Here we randomly initialize our neural network layers,
        # as well as our input and output size.
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        # TODO: Initialize weight matrices with uniform random values in [-1, 1]
        # - self.input_to_hidden: shape (hidden_size, input_size)
        # - self.hidden_to_output: shape (output_size, hidden_size)
        # Hint: Use np.random.uniform(-1, 1, (rows, cols))
        self.input_to_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.hidden_to_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.b1 = np.random.uniform(-1, 1, hidden_size)
        self.b2 = np.random.uniform(-1, 1, output_size)

        # TODO: Compute number of parameters in each layer
        self.n_params_i2h = hidden_size * input_size + hidden_size  # W1 + b1
        self.n_params_h2o = output_size * hidden_size + output_size  # W2 + b2

        self.n_params = self.get_num_params()


    def get_action(self, state):
        """Forward pass through the network.

        Args:
            state: Observation array, shape (input_size,) or (batch_size, input_size)

        Returns:
            action: Output array, shape (output_size,) or (batch_size, output_size)
        """
        # TODO: Perform forward pass computation
        # 1. Hidden layer: hidden = tanh(state @ W1.T + b1)
        # 2. Output layer: output = tanh(hidden @ W2.T + b2)
        # 3. Clip output to [-1, 1] using np.clip()
        #
        # Hint: Use @ operator or np.matmul for matrix multiplication
        # Hint: .T transposes a matrix
        # Hint: np.tanh() applies tanh element-wise
        hidden = np.tanh(state @ self.input_to_hidden.T + self.b1)
        output = np.tanh(hidden @ self.hidden_to_output.T + self.b2)
        return np.clip(output, -1.0, 1.0)

    def set_weights(self, encoding):
        """Set network weights from a flat parameter vector.

        Args:
            encoding: Flat array of size (n_params,) containing all weights
        """
        # TODO: Map the flat encoding to weight matrices
        # Encoding layout: [W1 (flattened), b1, W2 (flattened), b2]
        # 1. Split encoding into four parts by layer
        # 2. Reshape each part to match the weight matrix shapes
        #
        # Hint: Use array slicing: encoding[:n] and encoding[n:]
        # Hint: Use np.reshape(array, (rows, cols)) or array.reshape((rows, cols))
        w1_size = self.n_hidden * self.n_input
        b1_size = self.n_hidden
        w2_size = self.n_output * self.n_hidden

        self.input_to_hidden = encoding[:w1_size].reshape(self.n_hidden, self.n_input)
        self.b1 = encoding[w1_size:w1_size + b1_size]
        self.hidden_to_output = encoding[w1_size + b1_size:w1_size + b1_size + w2_size].reshape(self.n_output, self.n_hidden)
        self.b2 = encoding[w1_size + b1_size + w2_size:]

    def geno2pheno(self, genotype):
        """Alias for set_weights (genotype to phenotype mapping)."""
        self.set_weights(genotype)

    def get_num_params(self):
        # TODO: Return the total number of parameters in both layers!
        return self.n_params_i2h + self.n_params_h2o

    def reset_controller(self, batch_size=1) -> None:
        pass
