from pathlib import Path
from typing import Tuple, Iterable

from keras import utils
from keras.callbacks import History
from keras.models import Model
from keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_files
from tqdm import tqdm


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


def evaluate_model(model: Model, test_tensors: np.ndarray, test_targets: np.ndarray) -> float:
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


def plot_history(history: History, img_path: Path) -> None:
    """
    Saves the given model training history to a plot at the given img_path.
    :param history: model training history to plot.
    :param img_path: path to save the plot to (.png).
    :return: None
    """
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(img_path)
    plt.clf()
