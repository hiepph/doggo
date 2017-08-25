import argparse
import os
from glob import glob

import cv2
import numpy as np
from sklearn.datasets import load_files

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from model import build_model


## Variables
n_categories = 134


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), n_categories)

    return dog_files, dog_targets

def get_dog_breed(file_path):
    """Extract dog breed name from a file path"""
    return file_path.split('/')[-1].split('.')[-1]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, required=True,
            help='Directory contains train/valid data of dogs')
    argparser.add_argument('--bottleneck', type=str, required=True,
            help='Path to bottleneck features of pre-trained model')
    args = argparser.parse_args()

    # Load data
    print("INFO: Load data from %s" % args.data)
    _, train_targets = load_dataset(os.path.join(args.data, 'train'))
    _, valid_targets = load_dataset(os.path.join(args.data, 'valid'))

    # Load list of dog breeds
    # print("INFO: Load dog breeds")
    # dog_breeds = list(map(get_dog_breed, sorted(glob(os.path.join(args.data, 'train')))))

    # Load bottle-neck features from pre-trained Resnet50 model
    print("INFO: Load bottle-neck features from %s" % args.bottleneck)
    bottleneck_features = np.load(args.bottleneck)
    train_Resnet50 = bottleneck_features['train']
    valid_Resnet50 = bottleneck_features['valid']

    # Build model
    model = build_model(input_shape=train_Resnet50.shape[1:], n_output=n_categories)
    print("INFO: Model summary")
    model.summary()

    model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    checkpointer = ModelCheckpoint(
            filepath='models/doggo-resnet50.hdf5',
            save_best_only=True,
            verbose=1)

    if not os.path.exists('model'):
        os.mkdir('model')

    ## Train model
    print("INFO: Start training model")
    model.fit(
            train_Resnet50, train_targets,
            validation_data=(valid_Resnet50, valid_targets),
            epochs=20,
            batch_size=32,
            callbacks=[checkpointer],
            verbose=1)
