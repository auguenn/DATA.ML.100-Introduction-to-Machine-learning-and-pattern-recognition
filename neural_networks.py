import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense


# Setting Hyperparameters
batch_size = 132
num_classes = 10
epochs = 10

# load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# shuffle the samples and split them
indexes = np.arange(x_train.shape[0], dtype = int)
np.random.shuffle(indexes)
x_train = x_train[indexes]
y_train = y_train[indexes]

nsplit = int(0.9 * x_train.shape[0])

# Train and validation split
X_train = x_train[:nsplit]
Y_train = y_train[:nsplit]
X_val = x_train[nsplit:]
Y_val = y_train[nsplit:]

# Converting class vectors to binary class matrices
Y_train_2 = to_categorical(Y_train)
Y_val_2 = to_categorical(Y_val)
Y_test_2 = to_categorical(y_test)


# Data scaling and standardization
norm_type = 0

if norm_type == 0:
    X_train = X_train/255
    X_val = X_val/255
    X_test = x_test/255
elif norm_type == 1:
    train_mean, train_std = X_train.mean(), X_train.std()
    X_train = (X_train - train_mean)/train_std
    X_val = (X_val - train_mean)/train_std
    X_test = (X_test - train_mean)/train_std
else:
    pass


# Defining model
def fully_connected_model(ishape=(32, 32, 3), k=num_classes, lr=0.001):
    model_input = tf.keras.Input(shape=ishape)
    l2 = Dense(5, activation='sigmoid')(model_input)
    l3 = Dense(128, activation='relu')(l2)

    l4 = Dense(256, activation='relu')(l3)
    l4_drop = Dropout(0.3)(l4)

    l5 = Dense(512, activation='relu')(l4_drop)
    l5_drop = Dropout(0.3)(l5)

    l1_flatten = Flatten()(l5_drop)
    l6_drop = Dropout(0.3)(l1_flatten)

    out = Dense(k, activation='sigmoid')(l6_drop)
    model = Model(inputs=model_input, outputs=out)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=["accuracy"])
    return model


# Training the model
model = fully_connected_model()
model.fit(X_train, Y_train_2, batch_size = batch_size, epochs= epochs, verbose = 1, validation_data = (X_val, Y_val_2))


# Test this trained model on our test data
scores = model.evaluate(X_test, Y_test_2)
print(f"Accuracy for the fully connected network: {scores[1]*100:.3f}%")





