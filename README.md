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
This project was implemented and tested on Windows with Python 3.6, and TensorFlow 1.15. 

Clone the repo:
```bash
git clone https://github.com/angel-fontalvo/Traffic-Sign-Dataset
```

Install the requirements using `virtualenv`:
todo: create install script
```bash
# pip
source scripts/install.sh
```

## End-to-end Attack Demo

### Binaries 
* load_data.py
* train_model.py
* generate_adversarial_images.py

### Fetch data

```bash
git clone https://github.com/angel-fontalvo/Traffic-Sign-Dataset.git
```

###  Generate training labels and features 

```bash
$ DATA_DIR=/tmp/data
$ DATASET=Traffic-Sign-Dataset/dataset/
$ python load_data.py \
    --dataset $DATASET \
    --output $DATA_DIR
```
Pre-processed data will be generated in `$DATA_DIR`

###  Train traffic sign classifier model

```bash
$ MODEL_DIR=model
$ MODEL_NAME=saved_model.h5
$ python train_model.py \
    --dataset $DATA_DIR
    --outout $MODEL_DIR
    --model-name $MODEL_NAME
```
Model will be generated in `$MODEL_DIR`

###  Generate the adverarial images

```bash
$ python generate_adversarial_images.py \
    --dataset $DATA_DIR
    --model-dir $MODEL_DIR
    --model-name $MODEL_NAME
```
Adversarial images will be generated under:
* `adv_imgs\adverarial` = adversarial images
* `adv_imgs\noise` = noise added to benign image
* `adv_imgs\reg` = the benign image

###  Evaluate the images
todo

## Bring Your Own Model
todo

## Defenses
todo

