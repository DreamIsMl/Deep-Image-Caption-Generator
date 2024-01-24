# Deep Image Caption Generator

## Project Overview

This repository contains the code for a Deep Image Caption Generator. The model is trained on the Kaggle Flickr30k dataset, utilizing the Xception pre-trained model to extract features from images. The project involves preprocessing the caption data, creating encoder and decoder functions, building a model architecture with CNN, training the model, and developing a function to test images.

## Dataset

The dataset used for this project is the Kaggle Flickr30k dataset, which consists of images along with corresponding captions.

### Kaggle Flickr30k Dataset
- [Flickr30k Dataset on Kaggle](https://www.kaggle.com/hsankesara/flickr-image-dataset)

## Project Structure

The repository is structured as follows:

```plaintext
|-- data/
|   |-- flickr30k_images/          # Folder containing the Flickr30k images
|   |-- captions.txt               # Preprocessed captions file
|
|-- src/
|   |-- preprocess.py              # Script for preprocessing caption data
|   |-- model.py                   # Script defining the model architecture
|   |-- train.py                   # Script for training the model
|   |-- test.py                    # Script for testing the model on images
|
|-- main.py                        # Main script to run the entire pipeline
|-- requirements.txt               # Required dependencies for the project
|-- README.md                      # Project documentation
