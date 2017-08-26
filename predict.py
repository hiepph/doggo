import argparse
from glob import glob

import numpy as np

from model import build_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
from keras.preprocessing import image


def extract_Resnet50(tensor):
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def predict_breed(model, img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = model.predict(bottleneck_feature)

    breed_id = np.argmax(predicted_vector)
    prob = np.max(predicted_vector)
    return breed_id, prob


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, required=True,
            help='Path to input image')
    args = argparser.parse_args()

    dog_breeds = open('dog_breeds.txt').read().split('\n')[:-1]

    model = load_model('model/doggo-resnet50.hdf5')
    breed_id, prob = predict_breed(model, args.input)

    print("%s (%.1f%%)" % (dog_breeds[breed_id], prob * 100))
