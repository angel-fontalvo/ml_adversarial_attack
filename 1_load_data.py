import os
import cv2
import random
import pickle
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_dir", required=True,
	help="path to input dataset dir which contains our data. The data in this folder will be in the format: " +
		"$dataset_dir/LABEL_A/data1.jpg, " +
	       "$dataset_dir/LABEL_A/data2.jpg, " +
		"$dataset_dir/LABEL_B/data1.jpg," +
	       "$dataset_dir/LABEL_B/data2.jpg,")
ap.add_argument("-o", "--output_dir", required=True,
	help="path to output dir for processed data ")
ap.add_argument("-s", "--size", type=int, default=70,
	help="Size of the image. Default is 70")

args = vars(ap.parse_args())

def preprocess_data(dataset, img_size):
    training_data = []
    categories = []

    for category in os.listdir(dataset):

        if category not in categories:
            categories.append(category)

        path = os.path.join(dataset, category)
        category_num = categories.index(category)

        for img in os.listdir(path):
            try:
                img_raw = cv2.imread(os.path.join(path, img))

                # Explanation as to why this needs to occur here
                # https://stackoverflow.com/questions/54959387/rgb-image-display-in-matplotlib-plt-imshow-returns-a-blue-image
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                img_raw = cv2.resize(img_raw, (img_size, img_size))

                # Since we are using imagery data, we know that each pixel value on that image will be between a range of 0-255
                # thus we'll divide our data-features by 255 in order to get each bit in our images to be in the
                # range of 0-1 instead of 0-255. 
                img_raw = np.round(img_raw / 255., 2)

                training_data.append([img_raw, category_num])
            except Exception as e:
                print('Unable to parse {}/{}'.format(category, img))
    
    return training_data, categories


if __name__ == "__main__":
    img_size = args["size"]
    training_data, categories = preprocess_data(args["dataset"], img_size)

    # shuffle our data
    random.shuffle(training_data)

    # Separate and package training data into the variables that will be fed to the neural network
    # (i.e an (x, y) pair, where x = data-features, and y = labels)
    data = []
    labels = []
    print('[INFO] Separating and packaging data')
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

    output = args["output"]
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
