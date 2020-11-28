from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import ImageFile
from sklearn.datasets import load_files
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from dog_breed_classifier import paths

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the data given at path into source and target vectors using sklearn.datasets.load_files.
    :param path: Path to load data from.
    :return: the file names (representing the source vectors) and the target vector.
    """
    data = load_files(str(path))
    dog_files = np.array(data['filenames'])
    dog_targets = utils.to_categorical(np.array(data['target']), num_classes=133)
    return dog_files, dog_targets


def path_to_tensor(img_path: str) -> np.ndarray:
    """
    Converts the image given by img_path into a 4D-tensor with shape (1, 224, 224, 3).
    :param img_path: path as str to the image to convert in a tensor.
    :return: the 4D-tensor of the image.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths: Iterable[str]) -> np.ndarray:
    """
    Converts all images given by img_paths into 4D-tensor with shape (1, 224, 224, 3)
        and stacks them vertically.
    :param img_paths: paths to the images to convert into tensors.
    :return: a 4D-tensor with shape (num_samples, 224, 224, 3).
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def create_model() -> Sequential:
    """
    Creates a model from scratch, prints its summary and compiles it.
    Hint: If you want to experiment with different model architectures,
        this is the place to do it.
    :return: the defined and compiled model.
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(133, activation='softmax'))

    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def fit_model(model: Sequential,
              train_tensors: np.ndarray,
              train_targets: np.ndarray,
              valid_tensors: np.ndarray,
              valid_targets: np.ndarray,
              epochs: int,
              best_model_path: Path) -> None:
    """
    Fit the given model with the other parameters passed to this function.
    :param model: model to fit.
    :param train_tensors: source tensors used to train the model.
    :param train_targets: target vector used to train the model.
    :param valid_tensors: source tensors used to validate the model.
    :param valid_targets: target vector used to validate the model.
    :param epochs: number of epochs the model is trained for.
    :param best_model_path: path to store the best model to.
    :return: None
    """

    checkpointer = ModelCheckpoint(filepath=str(best_model_path), verbose=1, save_best_only=True)

    model.fit(train_tensors, train_targets,
              validation_data=(valid_tensors, valid_targets),
              epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


def evaluate_model(model: Sequential, test_tensors: np.ndarray, test_targets: np.ndarray) -> float:
    """
    Evaluate the given model on the given test_tensors with the given test_targets.
    :param model: model to evaluate.
    :param test_tensors: tensors the model makes predictions for.
    :param test_targets: target vector to compare the model's predictions with.
    :return: the computed test accuracy.
    """
    # get index of predicted dog breed for each image in test set
    dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # report test accuracy
    return 100 * np.sum(np.array(dog_breed_predictions) == np.argmax(test_targets, axis=1)) / len(dog_breed_predictions)


if __name__ == '__main__':

    # load train, test, and validation datasets
    train_files, train_targets = load_dataset(paths.DOG_IMAGES_TRAIN)
    valid_files, valid_targets = load_dataset(paths.DOG_IMAGES_VALID)
    test_files, test_targets = load_dataset(paths.DOG_IMAGES_TEST)

    # print statistics about the dataset
    print('There are %d total dog categories.' % len(list(paths.DOG_IMAGES_TRAIN.iterdir())))
    print('There are %s total dog images.' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.' % len(test_files))

    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32') / 255
    valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
    test_tensors = paths_to_tensor(test_files).astype('float32') / 255

    # create and compile the model
    model = create_model()

    # fit the model
    epochs = 5
    best_model_path: Path = paths.FROM_SCRATCH_MODEL_WEIGHTS
    fit_model(model, train_tensors, train_targets, valid_tensors, valid_targets, epochs, best_model_path)

    # load the best fitted model
    model.load_weights(str(best_model_path))

    # evaluate model
    test_accuracy = evaluate_model(model, test_tensors, test_targets)
    print('Test accuracy: %.4f%%' % test_accuracy)
