from pathlib import Path

import numpy as np
from keras import backend as K
from keras.applications.xception import Xception, preprocess_input

from dog_breed_classifier.detection.training import utils


def detect_dog(img_path: Path) -> bool:
    """
    Checks whether the passed image contains a dog or not.
    :param img_path: path to the image.
    :return: whether the passed image contains a dog or not.
    """
    dog_detector = Xception(include_top=True, weights='imagenet')
    img = preprocess_input(utils.path_to_tensor(str(img_path)))
    prediction = np.argmax(dog_detector.predict(img))
    K.clear_session()
    return 151 <= prediction <= 268
