from pathlib import Path

from keras.callbacks import History, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np
from PIL import ImageFile

from dog_breed_classifier import paths
from dog_breed_classifier.detection.training.utils import evaluate_model, load_dataset, paths_to_tensor, plot_history

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_model() -> Sequential:
    """
    Creates a model from scratch, prints its summary and compiles it.
    Hint: If you want to experiment with different model architectures,
        this is the place to do it.
    :return: the defined and compiled model.
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(133, activation='softmax'))

    model.summary()

    model.compile(optimizer=RMSprop(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def fit_model(model: Sequential,
              train_tensors: np.ndarray,
              train_targets: np.ndarray,
              valid_tensors: np.ndarray,
              valid_targets: np.ndarray,
              epochs: int,
              best_model_path: Path) -> History:
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

    train_history: History = model.fit(train_tensors, train_targets,
                                       validation_data=(valid_tensors, valid_targets),
                                       epochs=epochs,
                                       batch_size=32,
                                       callbacks=[checkpointer],
                                       verbose=1)

    return train_history


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
    epochs = 12
    best_model_path: Path = paths.FROM_SCRATCH_MODEL_WEIGHTS
    history = fit_model(model, train_tensors, train_targets, valid_tensors, valid_targets, epochs, best_model_path)

    # plot training history
    plot_history(history, paths.FROM_SCRATCH_MODEL_HISTORY)

    # load the best fitted model
    model.load_weights(str(best_model_path))

    # evaluate model
    test_accuracy = evaluate_model(model, test_tensors, test_targets)
    print('Test accuracy: %.4f%%' % test_accuracy)

    # save whole model
    model.save(str(paths.FROM_SCRATCH_MODEL))
