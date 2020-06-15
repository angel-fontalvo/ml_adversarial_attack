import os
import cv2
import random
import pickle
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_in", required=True,
                help="path to input dataset dir which contains unprocessed data. " +
                     "The data in this folder will be in the format: " +
                     "$dataset_dir/LABEL_A/data1.jpg, " +
                     "$dataset_dir/LABEL_A/data2.jpg, " +
                     "$dataset_dir/LABEL_B/data1.jpg, " +
                     "$dataset_dir/LABEL_B/data2.jpg,")
ap.add_argument("-o", "--data_out", required=True,
                help="path to output dir for processed data: ")
ap.add_argument("-s", "--size", type=int, default=70,
                help="Size of the image length and width of each image after being processed. Default is 70")

args = vars(ap.parse_args())


def preprocess_data(dataset, img_size):
    """
    Performs the following operations for each piece of data in the dataset :
    1. Transform the data into some numeric value
    2. Resize the data
    3. Each point is some numeric value. Is this case, since this is a color image, each value (i.e pixel) will be in the range of 0-255
    4. Divide each pixel by 255 to get it in any range between 0-1, and round to 2 decimal places. This is done to reduce the amount of data that must be processed.
    5. Add this data, along with it's corresponding label to a list
    :rtype: object
    :param dataset: directory where our data is stored
    :param img_size: integer that will be used for length and width of our images
    :return: [training_data_processed, unique_categories]
    """
    training_data_processed = []
    unique_categories = []

    for img_category in os.listdir(dataset):
      
        if img_category not in unique_categories:
            unique_categories.append(img_category)
            
        img_category_index = unique_categories.index(img_category)
        
        img_path = os.path.join(dataset, img_category)

        for img in os.listdir(img_path):
            try:
                img_processing = cv2.imread(os.path.join(img_path, img))

                # Explanation as to why this needs to occur here
                # https://stackoverflow.com/questions/54959387/rgb-image-display-in-matplotlib-plt-imshow-returns-a-blue-image
                img_processing = cv2.cvtColor(img_processing, cv2.COLOR_BGR2RGB)

                img_processing = cv2.resize(img_processing, (img_size, img_size))

                # Since we are using imagery data, we know that each pixel value on that image will be between a range of 0-255
                # thus we'll divide our data features by 255 in order to get each pixel in our images to be in the
                # range of 0-1 instead of 0-255 and round to 2 decimal places.
                img_processed = np.round(img_processing / 255., 2)

                # add processed data to the list
                training_data_processed.append([img_processed, img_category_index])
            except Exception as e:
                print('Unable to parse {}/{}'.format(category, img))

    return training_data_processed, unique_categories


if __name__ == "__main__":
    img_size = args["size"]
    training_data, categories = preprocess_data(args["data_in"], img_size)

    # shuffle our data
    random.shuffle(training_data)

    # Separate and package training data into the variables that will be fed to the neural network
    # (i.e an (x, y) pair, where x = features from your data, and y = labels)
    data = []
    labels = []
    print('[INFO] Separating and packaging training data')
    for features, label in training_data:
        data.append(features)
        labels.append(label)

    # Convert data features to a format which will be compatible with Keras. 
    print('[INFO] Reshaping features')
    X = np.array(data).reshape(-1, img_size, img_size, 3)

    # Package [y, categories] to a format which will be compatible with Cleverhans 
    print('[INFO] Reshaping labels')
    y = np.array(labels, dtype='int').ravel()
    tot_labels = y.shape[0]
    categorical = np.zeros((tot_labels, len(categories)))
    categorical[np.arange(tot_labels), y] = 1

    classes = np.array(categories)

    output = args["data_out"]
    if not os.path.exists(output):
        os.mkdir(output)

    # Save our preprocessed training dataset
    print('[INFO] Saving training sets')
    with open(os.path.join(output, 'x.p'), 'wb') as f:
        pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output, 'y.p'), 'wb') as f:
        pickle.dump(categorical, f)

    with open(os.path.join(output, 'categories.p'), 'wb') as f:
        pickle.dump(classes, f)
