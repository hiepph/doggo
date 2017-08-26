# Doggo

Deep convolutional neural network to classify dog breeds.

Full list of dog breeds is in `dog_breeds.txt` (133 categories).

Transfer learning from pre-trained model by Udacity's dog-project.


## Requirements

+ Python >=3.4

+ Python libraries: `pip install -r requirements.txt`


## Train

+ Get data:

```sh
mkdir data
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip && rm dogImages.zip
mv dogImages data/dogs
```

+ Get bottleneck features:

```sh
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz
mv DogResnet50Data.npz data/
```

+ Train model:

```sh
python train.py --data data/dogs --bottleneck data/DogResnet50Data.npz
```

Model is saved in `model/doggo-resnet50.hdf5`


## Predict

```sh
python predict.py --input marutaro.jpg
```
