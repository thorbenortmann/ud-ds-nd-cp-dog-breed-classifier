from pathlib import Path
import random
from typing import List, Optional

import numpy as np
from tensorflow.keras.models import Sequential

from dog_breed_classifier import paths
from dog_breed_classifier.detection.training import train_from_scratch


def get_model() -> Optional[Sequential]:
    if paths.TRANSFER_LEARNING_MODEL_WEIGHTS.is_file():
        pass
    elif paths.FROM_SCRATCH_MODEL_WEIGHTS.is_file():
        custom_model = train_from_scratch.create_model()
        custom_model.load_weights(str(paths.FROM_SCRATCH_MODEL_WEIGHTS))
        return custom_model

    else:
        return None


classifier: Optional[Sequential] = get_model()
dog_names: List[str] = [dir_.name[4:] for dir_ in paths.DOG_IMAGES_TRAIN.iterdir()]


def detect_breed(img_path: Path) -> str:
    if classifier:
        tensor = train_from_scratch.path_to_tensor(str(img_path)).astype('float32') / 255
        predicted_vector = classifier.predict(tensor)
        return dog_names[np.argmax(predicted_vector)]
    else:
        return f'Seems like you did not train a model yet. ' \
               f'So I will make a random guess: {dog_names[random.randint(0, 132)]} ?'
