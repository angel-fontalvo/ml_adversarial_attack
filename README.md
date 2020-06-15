![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Machine Learning Adversarial Image Generator 

This tool generates poisonous inputs designed to cause deep neural networks to make decisions on behalf of the attacker. 

| __Benign__ | __Adversarial__ |
|-------------|------------|
| ![Preview](imgs/benign_1.png)         | ![Preview](imgs/adversarial_1.png)     |
| ![Preview](imgs/benign_2.png)         | ![Preview](imgs/adversarial_2.png) |
| ![Preview](imgs/benign_3.png)         | ![Preview](imgs/adversarial_3.png) |

Based from the following papers: 

* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
* [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/abs/1602.02697)

## Pre-requesites

The following dependencies are required: 
* Tensorflow >= v1.8
* [Cleverhans](https://github.com/tensorflow/cleverhans) 

## Getting Started
Clone the repo:
```bash
git clone https://github.com/angel-fontalvo/ml_adversarial_attack.git
```

Create a new virtual environment by choosing a Python interpreter and making a .\venv directory:
```bash
python -m venv .\venv
```

Activate the environment.

Windows:
```bash
.\venv\Scripts\activate
```

Ubuntu/mac OS:
```bash
source ./venv/bin/activate
```

Install the requirements:
```bash
pip install -r requirements.txt
```

## Binaries 
* 1_load_data.py
* 2_train_model.py
* 3_generate_adversarial_images.py

## End-to-end Attack Demo

### Fetch training data

```bash
git clone https://github.com/angel-fontalvo/Traffic-Sign-Dataset.git
```
About the dataset: The dataset contains a list of folders, each folder having the name of a traffic sign. Inside of each, there will be 150-250 images of that sign. The folder name will be your label, and its contents will be your training data. 

### (Optional) Augment your training data
Increases the image count in each folder to 6,500-8,000.
todo

###  Generate training labels and features 

```bash
$ DATA_PREPROCESSED=Traffic-Sign-Dataset/dataset
$ DATA_PROCESSED=dataset
$ python 1_load_data.py \
    --data_in $DATA_PREPROCESSED \
    --data_out $DATA_PROCESSED
```
3 files will be generated in `$DATA_PROCESSED`: `x.p`, `y.p`, and `categories.p`

###  Train traffic sign classifier model

```bash
$ MODEL_DIR=model
$ MODEL_NAME=saved_model.h5
$ python 2_train_model.py \
    --dataset $DATA_PROCESSED \
    --output $MODEL_DIR \
    --model-name $MODEL_NAME
```
Model will be generated in `$MODEL_DIR`

###  Generate the adversarial images

```bash
$ python 3_generate_adversarial_images.py \
    --dataset $DATA_PROCESSED \
    --model-dir $MODEL_DIR \
    --model-name $MODEL_NAME
```
Adversarial images will be generated under:
* `adv_imgs\adverarial` = adversarial images
* `adv_imgs\noise` = noise added to benign images
* `adv_imgs\benign` = benign images

###  Evaluate the images
todo

## Bring Your Own Model
todo

## Defenses
todo

