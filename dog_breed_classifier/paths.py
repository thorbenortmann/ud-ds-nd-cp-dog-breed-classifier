from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).parent

UPLOAD_FOLDER: Path = PACKAGE_ROOT / 'app' / 'static' / 'uploads'

TRAINING_FOLDER: Path = PACKAGE_ROOT / 'detection' / 'training'

TRAINING_DATA: Path = TRAINING_FOLDER / 'data'
DOG_IMAGES: Path = TRAINING_DATA / 'dog_images'
DOG_IMAGES_TRAIN: Path = DOG_IMAGES / 'train'
DOG_IMAGES_VALID: Path = DOG_IMAGES / 'valid'
DOG_IMAGES_TEST: Path = DOG_IMAGES / 'test'

MODELS: Path = TRAINING_FOLDER / 'models'
FROM_SCRATCH_MODEL_WEIGHTS: Path = MODELS / 'weights.best.from_scratch.hdf5'
TRANSFER_LEARNING_MODEL_WEIGHTS: Path = MODELS / 'weights.best.transfer_learning.hdf5'
