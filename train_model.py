import os
import argparse
import functools
import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras
from numpy import load
from cleverhans.utils_tf import model_eval

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
MaxPooling2D = tf.keras.layers.MaxPooling2D

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset dir that will be used to train the model")
ap.add_argument("-ts", "--test-size", type=int, default=1200,
                help="How many samples from your training data to use to validate the model. Default is 1200")
ap.add_argument("-o", "--output", default="model",
                help="path to output dir for the trained model. Defaut is model")
ap.add_argument("-m", "--model-name", default="saved_model.h5",
                help="The name of the model. Default is saved_model.h5")
ap.add_argument("-e", "--epochs", type=int, default=10,
                help="The number of epochs you wish to use to train your model. Default is 10")
ap.add_argument("-s", "--batch-size", type=int, default=70,
                help="The number of data samples you wish your model to use during its training process. Default is 70.")
args = vars(ap.parse_args())


# Model architecture, keras dependent
def substitute_model(x, nb_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same",
                     activation="relu", input_shape=x.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(tf.keras.layers.Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(tf.keras.layers.Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(tf.keras.layers.Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))

    # # A dropout layer randomly drops some of the connections between layers. This helps to prevent overfitting.
    model.add(tf.keras.layers.Dropout(0.5))

    # output layer
    # model.add(Dense(1, activation="softmax"))
    model.add(Dense(nb_classes, activation="softmax"))

    return model


def train(x, y, x_train, y_train, x_test, y_test, nb_classes, nb_epochs, batch_size, output_dir, model_name):
    model = substitute_model(x, nb_classes)
    print('[INFO] Model architecture defined')

    # # Configure our model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Begin training
    print('[INFO] Model training starting...')
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=nb_epochs, validation_split=0.1)

    # Create dir where model will be saved if dir does not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_extension = os.path.splitext(model_name)[1]
    if (file_extension != '.h5'):
        model_name = model_name + '.h5'

    # Save the model
    model_save_dir = os.path.join(output_dir, model_name)
    model.save(model_save_dir)
    print('[INFO] Model saved: {}'.format(model_save_dir))

    # Perform evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    tf.keras.backend.set_learning_phase(0)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Load args
    dataset = args['dataset']
    output_dir = args['output']
    test_size = args['test_size']
    nb_epochs = args['epochs']
    batch_size = args['batch_size']
    model_name = args['model_name']

    # Load data
    x_train = pickle.load(open(os.path.join(dataset, 'x.p'), 'rb'))
    y_train = pickle.load(open(os.path.join(dataset, 'y.p'), 'rb'))
    x_test = x_train[0:test_size]
    y_test = y_train[0:test_size]

    # https://www.quora.com/When-should-I-use-tf-float32-vs-tf-float64-in-TensorFlow
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')

    # Obtain Image parameters
    img_rows, img_cols, nb_channels = x_train.shape[1:]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder. 
    # A barebone definition for input (i.e x) and output (i.e y)
    x = tf.placeholder(tf.float32, shape=(
        None, img_rows, img_cols, nb_channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    train(x, y, x_train, y_train, x_test, y_test, nb_classes,
          nb_epochs, batch_size, output_dir, model_name)
