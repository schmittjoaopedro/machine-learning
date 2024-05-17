import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import h5py
from tensorflow.keras.initializers import random_uniform, glorot_uniform

for device in tf.config.list_physical_devices():
    print(device)


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def convert_to_one_hot(y, c):
    return np.eye(c)[y.reshape(-1)].T


def identity_block(x, f, filters, initializer=random_uniform):
    f1, f2, f3 = filters
    x_shortcut = x
    # Stage 1
    x = k.layers.Conv2D(filters=f1, kernel_size=1, strides=(1, 1), padding='valid',
                        kernel_initializer=initializer(seed=0))(x)
    x = k.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-3)(x)  # Default axis
    x = k.layers.Activation('relu')(x)
    # Stage 2
    x = k.layers.Conv2D(filters=f2, kernel_size=f, strides=(1, 1), padding='same',
                        kernel_initializer=initializer(seed=0))(x)
    x = k.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-3)(x)
    x = k.layers.Activation('relu')(x)
    # Stage 3
    x = k.layers.Conv2D(filters=f3, kernel_size=1, strides=(1, 1), padding='valid',
                        kernel_initializer=initializer(seed=0))(x)
    x = k.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-3)(x)
    # Stage 4
    x = k.layers.Add()([x_shortcut, x])
    x = k.layers.Activation('relu')(x)
    return x


def convolutional_block(x, f, filters, s=2, initializer=glorot_uniform):
    f1, f2, f3 = filters
    x_shortcut = x

    # MAIN PATH
    # First component of main path glorot_uniform(seed=0)
    x = k.layers.Conv2D(filters=f1, kernel_size=1, strides=(s, s), padding='valid',
                        kernel_initializer=initializer(seed=0))(x)
    x = k.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-3)(x)
    x = k.layers.Activation('relu')(x)

    # Second component of main path
    x = k.layers.Conv2D(filters=f2, kernel_size=f, strides=(1, 1), padding='same',
                        kernel_initializer=initializer(seed=0))(x)
    x = k.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-3)(x)
    x = k.layers.Activation('relu')(x)

    # Third component of main path
    x = k.layers.Conv2D(filters=f3, kernel_size=1, strides=(1, 1), padding='valid',
                        kernel_initializer=initializer(seed=0))(x)
    x = k.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-3)(x)

    # SHORTCUT PATH
    x_shortcut = k.layers.Conv2D(filters=f3, kernel_size=1, strides=(s, s), padding='valid',
                                 kernel_initializer=initializer(seed=0))(x_shortcut)
    x_shortcut = k.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-3)(x_shortcut)

    # Final step: Add shortcut value to main path (Use this order [x, x_shortcut]),
    # and pass it through a RELU activation
    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.Activation('relu')(x)

    return x


def res_net_50(input_shape, outputs):
    # Define the input as a tensor with shape input_shape
    img_inputs = k.Input(shape=input_shape)

    # Zero-Padding
    x = k.layers.ZeroPadding2D((3, 3))(img_inputs)

    # Stage 1
    x = k.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(x)
    x = k.layers.BatchNormalization(axis=3, momentum=0.9, epsilon=1e-3)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = convolutional_block(x, f=3, filters=[64, 64, 256], s=1)
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])

    # Stage 3
    x = convolutional_block(x, f=3, filters=[128, 128, 512], s=2)
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])

    # Stage 4
    x = convolutional_block(x, f=3, filters=[256, 256, 1024], s=2)
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])

    # Stage 5
    x = convolutional_block(x, f=3, filters=[512, 512, 2048], s=2)
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])

    # AVGPOOL
    x = k.layers.AveragePooling2D(pool_size=(2, 2))(x)

    # output layer
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(outputs, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(x)

    # Create model
    return k.Model(inputs=img_inputs, outputs=x)


# LOADING DATASET
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# INSTANTIATING THE MODEL
model = res_net_50(input_shape=(64, 64, 3), outputs=len(classes))
print(model.summary())

np.random.seed(1)
tf.random.set_seed(2)
opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# TRAINING THE MODEL
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# EVALUATING THE MODEL
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
