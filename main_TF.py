# we must import the libraries once again since we haven't imported them in this file
import numpy as np
import tensorflow as tf

# Data

#npz = np.load('Audiobooks_data_train.npz')
npz = np.load('data/CH3D_8000_Nt_101_Nx_32_compressed.npz')

# Check the keys in the npz file (these represent different arrays stored)
print("Keys in the file:", npz.files)

# Get the shape and size of each array
for key in npz.files:
    data = npz[key]
    print(f"Data for {key}:")
    print("Shape:", data.shape)
    print("Size:", data.size)

# we extract the inputs using the keyword under which we saved them
# to ensure that they are all floats, let's also take care of that
train_inputs = npz['inputs'].astype(np.float64)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(int)

# we load the validation data in the temporary variable
npz = np.load('Audiobooks_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs, validation_targets = npz['inputs'].astype(np.float64), npz['targets'].astype(int)

# we load the test data in the temporary variable
npz = np.load('Audiobooks_data_test.npz')
# we create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = npz['inputs'].astype(np.float64), npz['targets'].astype(int)


# Model
#Outline, optimizers, loss, early stopping and training

# Set the input and output sizes
input_size = 10
output_size = 2
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 50

# define how the model will look like
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # 2nd hidden layer
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax')  # output layer
])

### Choose the optimizer and the loss function

# we define the optimizer we'd like to use,
# the loss function,
# and the metrics we are interested in obtaining at each iteration
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Training
# That's where we train the model we have built.

# set the batch size
batch_size = 100

# set a maximum number of training epochs
max_epochs = 100

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
# note that this time the train, validation and test data are not iterable
model.fit(train_inputs,  # train inputs
          train_targets,  # train targets
          batch_size=batch_size,  # batch size
          epochs=max_epochs,  # epochs that we will train for (assuming early stopping doesn't kick in)
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks=[early_stopping],  # early stopping
          validation_data=(validation_inputs, validation_targets),  # validation data
          verbose=2  # making sure we get enough information about the training process
          )

'''
Test the model
As we discussed in the lectures, after training on the training data and validating on the validation data,
 we test the final prediction power of our model by running it on the test dataset that the algorithm has NEVER seen before.

It is very important to realize that fiddling with the hyperparameters overfits the validation dataset.
The test is the absolute final instance. You should not test before you are completely done with adjusting your model.
If you adjust your model after testing, you will start overfitting the test dataset, which will defeat its purpose.
'''

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

'''
Using the initial model and hyperparameters given in this notebook, the final test accuracy should be roughly around 91%.
Note that each time the code is rerun, we get a different accuracy because each training is different.
We have intentionally reached a suboptimal solution, so you can have space to build on it!
'''