import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
'''
SLR_BasicTF: Simple Linear Regression Basic TensorFlow
In this example: 1. Create random dataset, 2. save "inputs, and "targets", then load them,
3. Initialize and train the model.

when we are employing TensorFlow we must actually BUILD the model. we stored in variable call "model".
'Sequential': stacks layers --> takes inputs apply a single linear transformation, and provide output
from numpy we knew: output = np.dot(inputs, weights) + bias, but here:
output = model = tf.keras.layers.Dense(output_size): takes the inputs provided to the model and calculate the dot product of the
inputs and the weights and adds the bias. Also applies the activation function(optional)


Theoretical framework of ML:
1. Data
2. Model
3. Objective function
4. Optimization algorithm

The method which allows us to specify steps 3 and 4 is called compile.
model.compile(optimizer, loss): configures the model for training

Optimization algorithm:
sgd: stochastic gradient descent --> go online for possible optimizer
go online and check TF Keras optimizers we'll see a list of the names of different optimizers.

Objective function:
1. L2-norm loss: Least sum of squares (least sum of squared error) 
2. scaling by number of observations = average (mean) --> go online for possible losses

model.fit(input, target):
This same method is also the place where we set the number of iterations. We'll use this term to describe iterations 
and number of iterations. each iteration over the full data set in ML is called an epoch

verbose: visualization for loss details

We can't call 'SLR_BasicTF' and 'LinearRegression_For_Loop' deep learning yet.

in SLR_AdvanceTF network:
our model is as close as possible to our NumPy model. How can we adjust the hyperparameters in TensorFlow:
kernel_initializer: initialize weight 
bias_initializer: initialize bias
tf.keras.optimizers.SGD for learning_rate


Neural Network Architecture:
Input Layer:
2 neurons (because input_size=2, meaning it takes two features: x and z).
Output Layer (Dense Layer):
1 neuron --> No hidden layers (just direct mapping from input to output).

  x (Input 1)   ---> [ W1 ] \
                             \
                              ---> [ ∑(W1*x + W2*z + b) ] ---> y (Output)
                             /
  z (Input 2)   ---> [ W2 ] /


Till here we talked about simple NN (input layer, output layer)
-----------------
deep NN: 
width: number of neurons in hidden layer
Depth: number of hidden layers

'''

class LinearRegressionBasicNumpyNet_For_Loop:
    def __init__(self, input_size=2, learning_rate=0.02, init_range=0.1):
        """
        Initialize weights, biases, and learning rate.
        """
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(low=-init_range, high=init_range, size=(input_size, 1))
        self.biases = np.random.uniform(low=-init_range, high=init_range, size=1)

    def train(self, inputs, targets, epochs=100):
        """
        Train the model using gradient descent.
        """
        observations = inputs.shape[0]  # Number of training samples

        for i in range(epochs):
            # Forward pass: Compute predictions
            outputs = np.dot(inputs, self.weights) + self.biases

            # Compute the error (deltas)
            deltas = outputs - targets

            # Compute loss (L2 norm loss)
            loss = np.sum(deltas ** 2) / (2 * observations)
            print(f"Epoch {i + 1}, Loss: {loss}")

            # Compute gradients
            deltas_scaled = deltas / observations
            weight_gradient = np.dot(inputs.T, deltas_scaled)
            bias_gradient = np.sum(deltas_scaled)

            # Update weights and biases using gradient descent
            self.weights -= self.learning_rate * weight_gradient
            self.biases -= self.learning_rate * bias_gradient

    def predict(self, inputs):
        """
        Make predictions using trained weights and biases.
        """
        return np.dot(inputs, self.weights) + self.biases

    def get_parameters(self):
        """
        Get the current weights and biases.
        """
        return self.weights, self.biases

class SLR_BasicTF:
    def __init__(self, input_size=2, output_size=1):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(output_size)
        ])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')

    def train(self, training_inputs, training_targets, epochs=100, verbose=0):
        self.model.fit(training_inputs, training_targets, epochs=epochs, verbose=verbose)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_weights(self):
        """Returns the weights and biases of the first layer."""
        return self.model.layers[0].get_weights()

class SLR_AdvanceTF:
    def __init__(self, input_size=2, output_size=1):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(output_size,
                                  kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                  bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
                                  )
        ])
        #self.model.compile(optimizer='sgd', loss='mean_squared_error')
        self.custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
        self.model.compile(optimizer=self.custom_optimizer, loss='mean_squared_error')
    def train(self, training_inputs, training_targets, epochs=100, verbose=0):
        self.model.fit(training_inputs, training_targets, epochs=epochs, verbose=verbose)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_weights(self):
        """Returns the weights and biases of the first layer."""
        return self.model.layers[0].get_weights()


### Basic NN Example with TF Exercises
class SLR_in_Tensor_env:
    def __init__(self, input_size=2, output_size=1, learning_rate=0.05, init_range=0.1):
        """
        Initialize the linear regression model using TensorFlow 2.x.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Define model variables (weights and biases) using tf.Variable
        self.weights = tf.Variable(tf.random.uniform([input_size, output_size], minval=-init_range, maxval=init_range), name="weights")
        self.biases = tf.Variable(tf.random.uniform([output_size], minval=-init_range, maxval=init_range), name="biases")

        # Define optimizer
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)

    def train(self, training_inputs, training_targets, epochs=100):
        """
        Train the model using the provided dataset.
        """
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                outputs = tf.matmul(training_inputs, self.weights) + self.biases
                #loss_fn = tf.keras.losses.MeanSquaredError() # Mean Squared Error (MSE)
                loss_fn = tf.keras.losses.Huber() # Huber Loss
                loss = loss_fn(training_targets, outputs) / 2.0

            # Compute gradients and update parameters
            gradients = tape.gradient(loss, [self.weights, self.biases])
            self.optimizer.apply_gradients(zip(gradients, [self.weights, self.biases]))

            print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.6f}")

    def predict(self, test_inputs):
        """
        Generate predictions using the trained model.
        """
        return tf.matmul(test_inputs, self.weights) + self.biases

    def get_parameters(self):
        """
        Retrieve the trained model parameters (weights and biases).
        """
        return self.weights.numpy(), self.biases.numpy()


class MNISTClassifierTF2:
    def __init__(self, input_size=784, hidden_layer_size=50, output_size=10, learning_rate=0.001):
        """
        Initialize the MNIST classifier model.
        Args:
            input_size (int): Dimensionality of the flattened input images.
            hidden_layer_size (int): Number of neurons in each hidden layer.
            output_size (int): Number of classes.
            learning_rate (float): Learning rate for the optimizer.
        """

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        # Build the model using tf.keras.Sequential
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
            tf.keras.layers.Dense(output_size)  # logits output (no activation here)
        ])

        # Compile the model with Adam optimizer and CategoricalCrossentropy (from_logits=True)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def train(self, train_data, train_labels, validation_data, validation_labels, batch_size=100, max_epochs=15):
        """
        Train the model on the provided training data, validating on the provided validation data.
        Early stopping is applied if the validation loss increases.
        Args:
            train_data (np.array): Training images of shape (num_samples, input_size).
            train_labels (np.array): One-hot encoded training labels.
            validation_data (np.array): Validation images.
            validation_labels (np.array): One-hot encoded validation labels.
            batch_size (int): Number of samples per batch.
            max_epochs (int): Maximum number of epochs to train.
        Returns:
            history: The History object returned by model.fit.
        """
        # Early stopping callback: stop if the validation loss doesn't improve for 1 epoch.
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=1, restore_best_weights=True
        )

        start_time = time.time()
        '''
        TensorFlow automatically prints the batch progress during training.
        Since verbose=2 is set, TensorFlow prints one line per epoch with summary stats instead of showing per-batch progress.
        If you want to see per-batch progress: Change verbose=2 to verbose=1 
        '''
        history = self.model.fit(
            train_data, train_labels,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_data=(validation_data, validation_labels),
            callbacks=[early_stop],
            verbose=2  # Change to 1 for more detailed per-epoch logging if needed.
        )
        self.history = history
        training_time = time.time() - start_time
        print("Training time: {:.2f} seconds".format(training_time))
        return history
    def plot_loss(self):
        """
        Plot training and validation loss over epochs.
        """
        if self.history is None:
            print("Error: You must train the model first!")
            return

        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()
    def evaluate(self, test_data, test_labels):
        """
        Evaluate the trained model on test data.
        Args:
            test_data (np.array): Test images.
            test_labels (np.array): One-hot encoded test labels.
        Returns:
            results (list): [loss, accuracy] on the test set.
        """
        results = self.model.evaluate(test_data, test_labels, verbose=0)
        print("Test loss: {:.3f}, Test accuracy: {:.2f}%".format(results[0], results[1] * 100))
        return results


    def predict(self, inputs):
        """
        Generate predictions for the given inputs.
        Args:
            inputs (np.array): Input data of shape (num_samples, input_size).
        Returns:
            predictions (np.array): Model outputs (logits). To obtain probabilities, apply softmax.
        """
        predictions = self.model.predict(inputs)
        return predictions