from pathlib import Path
import random
from typing import List

import numpy as np
from keras import backend as K
from keras.models import load_model

from dog_breed_classifier.detection.training import utils
from dog_breed_classifier.detection.training.transfer_learning import preprocess_input
from dog_breed_classifier import paths

FROM_SCRATCH_MODEL_IS_PRESENT: bool = paths.FROM_SCRATCH_MODEL.is_file()
TRANSFER_LEARNING_MODEL_IS_PRESENT: bool = paths.TRANSFER_LEARNING_MODEL.is_file()
DOG_NAMES: List[str] = [dir_.name[4:] for dir_ in paths.DOG_IMAGES_TRAIN.iterdir()]


def detect_breed(img_path: Path) -> str:
    """
    Detects the breed of the dog in the given image.
    :param img_path: path to the image to classify.
    :return: the name of the predicted dog breed.
    """
    if FROM_SCRATCH_MODEL_IS_PRESENT or TRANSFER_LEARNING_MODEL_IS_PRESENT:
        tensor = utils.path_to_tensor(str(img_path)).astype('float32') / 255

        if TRANSFER_LEARNING_MODEL_IS_PRESENT:
            tensor = preprocess_input(tensor)
            model = load_model(str(paths.TRANSFER_LEARNING_MODEL))
        else:
            model = load_model(str(paths.FROM_SCRATCH_MODEL))

        predicted_vector = model.predict(tensor)
        K.clear_session()
        return DOG_NAMES[np.argmax(predicted_vector)]

    else:
        return f'Seems like you did not train a model yet. ' \
               f'So I will make a random guess: {DOG_NAMES[random.randint(0, 132)]} ?'
