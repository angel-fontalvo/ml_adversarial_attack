import tensorflow as tf
import pickle
import os
import numpy as np
import math
import argparse
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import model_eval
from PIL import Image


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset dir that will be used to craft the attack")
ap.add_argument("-md", "--model-dir", required=True,
                help="path to the dir that contains the model that's being attacked")
ap.add_argument("-mn", "--model-name", required=True,
                help="The name of the model that's being attacked")
ap.add_argument("-ts", "--test-size", type=int, default=5000,
                help="How many samples from your dataset to use to validate the attack success. Default is 5000")
ap.add_argument("-ao", "--adversarial-output", default="adv_imgs",
                help="path to output dir where crafted poisonous images will be stored. Defaut is adv_imgs")
ap.add_argument("-s", "--batch-size", type=int, default=70,
                help="The number of samples you wish to use while crafting the attack. Default is 32")
args = vars(ap.parse_args())

def save_images_to_folder(images, size, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    for index in range(images.shape[0]):
        if index < size:
            image_array = (np.reshape(images[index], (size, size, 3))
                           * 255).astype(np.uint8)
            Image.fromarray(image_array, 'RGB').save(
                os.path.join(path, str(index) + '.png'))


def load_data(dataset, test_size):
    x_train = pickle.load(open(os.path.join(dataset, 'x.p'), 'rb'))
    y_train = pickle.load(open(os.path.join(dataset, 'y.p'), 'rb'))

    x_test = x_train[0:test_size]
    y_test = y_train[0:test_size]

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')

    return x_train, y_train, x_test, y_test


def generate_adversarial_data(sess, x, y, predictions=None, shape=None, dType=None, features=None, labels=None,
                              feed=None, batch_size=None):

    if features is None or labels is None:
        raise ValueError(
            "features argument and labels argument must be supplied.")

    if predictions is None:
        raise ValueError("predictions argument must be supplied.")

    if shape is None:
        raise ValueError("shape argument must be supplied.")

    # Define accuracy symbolically
    # correct_preds = tf.equal(tf.argmax(y, axis=-1), tf.argmax(predictions, axis=-1))

    # Initialize empty numpy array with the same dimensions of tensor being evaluated
    # todo: see how to obtain shape and dType from tensor value directly
    generated_preds = np.zeros(shape, dtype=dType)

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(features)) / batch_size))
        assert nb_batches * batch_size >= len(features)

        X_cur = np.zeros((batch_size,) + features.shape[1:],
                         dtype=features.dtype)
        Y_cur = np.zeros((batch_size,) + labels.shape[1:],
                         dtype=labels.dtype)
        for batch in range(nb_batches):
            start = batch * batch_size
            end = min(len(features), start + batch_size)

            cur_batch_size = end - start
            X_cur[:cur_batch_size] = features[start:end]
            Y_cur[:cur_batch_size] = labels[start:end]
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)

            cur_gen_preds = predictions.eval(feed_dict=feed_dict, session=sess)

            print('Batch {} out of {}'.format(str(batch), str(nb_batches)))
            if cur_batch_size < batch_size:
                print('Done')
            else:
                generated_preds[start:end] = cur_gen_preds

        assert end >= len(features)

    return generated_preds


if __name__ == '__main__':
    # Load args
    model_dir = args['model_dir']
    model_name = args['model_name']
    adversarial_output = args['adversarial_output']
    dataset = args['dataset']
    test_size = args['test_size']
    batch_size = args['batch_size']

    # Load the target model
    model = tf.keras.models.load_model(os.path.join(model_dir, model_name))

    # Convert model to cleverhans format
    ch_model = KerasModelWrapper(model)

    sess = tf.keras.backend.get_session()

    # # Load data
    x_train, y_train, x_test, y_test = load_data(dataset, test_size)

    img_rows, img_cols, nb_channels = x_train.shape[1:]
    nb_classes = y_train.shape[1]
    size = ch_model.model.inputs[0].shape[1]

    x = tf.placeholder(tf.float32, shape=(
        None, img_rows, img_cols, nb_channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    preds = model(x)

    print('[INFO] Evaluating the accuracy of the target model')
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    print('[INFO] Accuracy using benign samples: %0.4f' % acc)

    print('[INFO] Initializing Fast Gradient Sign Method (FGSM) attack')
    fgsm = FastGradientMethod(ch_model, sess=sess)
    # Mke sure to go over eps https://github.com/tensorflow/cleverhans/issues/589
    # i.e why attack may fail to find adversarial image | eps determines aggresiveness of attack
    # used: eps: 0.01, 0.3
    fgsm_params = {'eps': 0.05, 'clip_min': 0., 'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)

    preds_adv = model(adv_x)

    # Evaluate the accuracy of the model on adversarial examples
    acc = model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_params)
    print('[INFO] Test accuracy on adversarial examples (dodging): %0.4f\n' % acc)

    # Define correct shape and size symbolically
    # This is done because if you don't specify the actual size when appending the results in model_pred,
    # And error will be thrown. In order for the results to be properly appended, both list must be the correct size
    # todo: test above statement
    print('Generating adversarial features')
    shape = (test_size,) + x_test.shape[1:]
    dType = x_test.dtype
    adv_features = generate_adversarial_data(sess=sess, x=x, y=y, predictions=adv_x, shape=shape, dType=dType,
                                             features=x_test, labels=y_test, batch_size=batch_size)

    print('Generating adversarial labels')
    shape = (test_size,) + y_test.shape[1:]
    dType = y_test.dtype
    adversarial_labels = generate_adversarial_data(sess=sess, x=x, y=y, predictions=preds_adv, shape=shape, dType=dType,
                                                   features=adv_features, labels=y_test, batch_size=batch_size)

    print('# ------------------------------------ ACCURACY ---------------------------------------- #')
    same_faces_index = np.where((np.argmax(y_test, axis=-1) == 0))
    different_faces_index = np.where((np.argmax(y_test, axis=-1) == 1))

    acc = np.mean(
        (np.argmax(y_test[same_faces_index], axis=-1)) ==
        (np.argmax(adversarial_labels[same_faces_index], axis=-1))
    )
    print('Accuracy against adversarial examples for '
          + 'same object (dodging): '
          + str(acc * 100)
          + '%')

    acc = np.mean(
        (np.argmax(y_test[different_faces_index], axis=-1)) == (
            np.argmax(adversarial_labels[different_faces_index], axis=-1))
    )
    print('Accuracy against adversarial examples for '
          + 'different object (impersonation): '
          + str(acc * 100)
          + '%')



    save_images_to_folder(adv_features, size, os.path.join(
        adversarial_output, 'adversarial'))
    save_images_to_folder(0.5 + (adv_features - x_test),
                          size, os.path.join(adversarial_output, 'noise'))
    save_images_to_folder(
        x_test, size, os.path.join(adversarial_output, 'benign'))
