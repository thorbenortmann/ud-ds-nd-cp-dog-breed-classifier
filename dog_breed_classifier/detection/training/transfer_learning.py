from pathlib import Path

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import History, ModelCheckpoint
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
import numpy as np
from PIL import ImageFile

from dog_breed_classifier import paths
from dog_breed_classifier.detection.training.utils import evaluate_model, load_dataset, paths_to_tensor, plot_history

ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 32  # For keras ImageDataGenerators


def create_model() -> Model:
    """
    Creates a model to perform transfer learning.
    It consist of the already trained InceptionV3 network with weights learned on the imagenet dataset and
    a classification block specific for our dog breed classification problem on top.
    The InceptionV3 network's weights are frozen to prevent the destruction of already learned weights.
    :return: a model with the architecture described above.
    """
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output

    # Add our classification block
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(133, activation='softmax')(x)

    # Combine base model and our classification block into one
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done after setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_train_gen(train_tensors: np.ndarray, train_targets: np.ndarray) -> NumpyArrayIterator:
    """
    Creates a generator for training data. Besides returning the training data in batches,
    it also creates (random) variants of the training data (data augmentation).
    :param train_tensors: source tensors to learn from.
    :param train_targets: target tensor to learn from.
    :return: the created training data generator.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest')

    train_gen = train_datagen.flow(
        x=train_tensors,
        y=train_targets,
        batch_size=BATCH_SIZE)

    return train_gen


def create_valid_gen(valid_tensors: np.ndarray, valid_targets: np.ndarray) -> NumpyArrayIterator:
    """
    Creates a generator for validation data. It returns the given validation data in batches,
    it also creates (random) variants of the training data (data augmentation).
    :param valid_tensors: source tensors to validate with.
    :param valid_targets: target tensor to validate with.
    :return: the created validation data generator.
    """
    test_datagen = ImageDataGenerator()

    valid_gen = test_datagen.flow(
        x=valid_tensors,
        y=valid_targets,
        batch_size=BATCH_SIZE)

    return valid_gen


def fit_model(model: Model,
              train_gen: NumpyArrayIterator,
              train_len: int,
              valid_gen: NumpyArrayIterator,
              valid_len: int,
              epochs: int,
              checkpointer: ModelCheckpoint) -> History:
    """
    Fits the given model with the given data.
    :param model: model to fit.
    :param train_gen: generator providing the training data.
    :param train_len: length of one iteration of the train_gen.
    :param valid_gen: generator providing the validation data.
    :param valid_len: length of one iteration of the valid_gen.
    :param epochs: number of epochs to train.
    :param checkpointer: callback to store checkpoints of the model training.
    :return: the training history.
    """
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_len // BATCH_SIZE,
                                  epochs=epochs,
                                  validation_data=valid_gen,
                                  validation_steps=valid_len // BATCH_SIZE,
                                  callbacks=[checkpointer],
                                  verbose=1)

    return history


def fine_tune_model(model: Model,
                    train_gen: NumpyArrayIterator,
                    train_len: int,
                    valid_gen: NumpyArrayIterator,
                    valid_len: int,
                    epochs: int,
                    checkpointer: ModelCheckpoint) -> History:
    """
    Fine-tunes the given model with the given data by unfreezing parts of its architecture.
    :param model: model to fine-tune.
    :param train_gen: generator providing the training data.
    :param train_len: length of one iteration of the train_gen.
    :param valid_gen: generator providing the validation data.
    :param valid_len: length of one iteration of the valid_gen.
    :param epochs: number of epochs to train.
    :param checkpointer: callback to store checkpoints of the model training.
    :return: the training history.
    """
    # Unfreeze parts of the InceptionV3 network to fine-tune the model
    for layer in model.layers[:280]:
        layer.trainable = False
    for layer in model.layers[280:]:
        layer.trainable = True

    # fine-tune the model with a very small learning rate to avoid to destroy already learned weights
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_len // BATCH_SIZE,
                                  epochs=epochs,
                                  validation_data=valid_gen,
                                  validation_steps=valid_len // BATCH_SIZE,
                                  callbacks=[checkpointer],
                                  verbose=1)

    return history


if __name__ == '__main__':

    # load train, test, and validation datasets
    train_files, train_targets = load_dataset(paths.DOG_IMAGES_TRAIN)
    valid_files, valid_targets = load_dataset(paths.DOG_IMAGES_VALID)
    test_files, test_targets = load_dataset(paths.DOG_IMAGES_TEST)

    # print statistics about the dataset
    nb_train_samples = len(train_files)
    nb_valid_samples = len(valid_files)
    nb_test_samples = len(test_files)
    print('There are %d total dog categories.' % len(list(paths.DOG_IMAGES_TRAIN.iterdir())))
    print('There are %s total dog images.' % (nb_train_samples + nb_valid_samples + nb_test_samples))
    print('There are %d training dog images.' % nb_train_samples)
    print('There are %d validation dog images.' % nb_valid_samples)
    print('There are %d test dog images.' % nb_test_samples)

    # rescale for keras
    train_tensors = paths_to_tensor(train_files).astype('float32') / 255
    valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
    test_tensors = paths_to_tensor(test_files).astype('float32') / 255

    # pre-process for InceptionV3
    train_tensors = preprocess_input(train_tensors)
    valid_tensors = preprocess_input(valid_tensors)
    test_tensors = preprocess_input(test_tensors)

    # create ImageDataGenerators for data augmentation
    train_gen = create_train_gen(train_tensors, train_targets)
    valid_gen = create_valid_gen(valid_tensors, valid_targets)

    # create and compile the model
    model = create_model()

    # fit the model
    epochs = 12
    best_model_path: Path = paths.TRANSFER_LEARNING_MODEL_WEIGHTS
    checkpointer = ModelCheckpoint(filepath=str(best_model_path), verbose=1, save_best_only=True)
    history = fit_model(model,
                        train_gen, nb_train_samples,
                        valid_gen, nb_valid_samples,
                        epochs,
                        checkpointer)

    # plot training history
    plot_history(history, paths.TRANSFER_LEARNING_MODEL_HISTORY)

    # load the best fitted model
    model.load_weights(str(best_model_path))

    # evaluate model
    test_accuracy = evaluate_model(model, test_tensors, test_targets)
    print('Test accuracy: %.4f%%' % test_accuracy)

    # fine-tune model
    epochs = 12
    history = fine_tune_model(model,
                              train_gen, nb_train_samples,
                              valid_gen, nb_valid_samples,
                              epochs,
                              checkpointer)

    # plot training history
    plot_history(history, paths.FINE_TUNE_MODEL_HISTORY)

    # load the best fitted model
    model.load_weights(str(best_model_path))

    # evaluate model
    test_accuracy = evaluate_model(model, test_tensors, test_targets)
    print('Test accuracy: %.4f%%' % test_accuracy)

    # save whole model
    model.save(str(paths.TRANSFER_LEARNING_MODEL))
