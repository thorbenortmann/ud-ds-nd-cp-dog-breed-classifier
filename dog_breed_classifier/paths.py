from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).parent

UPLOAD_FOLDER: Path = PACKAGE_ROOT / 'app' / 'static' / 'uploads'

# TRAINING
TRAINING_FOLDER: Path = PACKAGE_ROOT / 'detection' / 'training'

# TRAINING - DATA
TRAINING_DATA: Path = TRAINING_FOLDER / 'data'

DOG_IMAGES: Path = TRAINING_DATA / 'dog_images'
DOG_IMAGES_TRAIN: Path = DOG_IMAGES / 'train'
DOG_IMAGES_VALID: Path = DOG_IMAGES / 'valid'
DOG_IMAGES_TEST: Path = DOG_IMAGES / 'test'

# TRAINING - MODELS
MODELS: Path = TRAINING_FOLDER / 'models'

# TRAINING - MODELS - FROM SCRATCH
FROM_SCRATCH: Path = MODELS / 'from_scratch'
FROM_SCRATCH_MODEL_WEIGHTS: Path = FROM_SCRATCH / 'weights.best.from_scratch_model.hdf5'
FROM_SCRATCH_MODEL_HISTORY: Path = FROM_SCRATCH / 'from_scratch_model_history.png'

# TRAINING - MODELS - FROM SCRATCH - FINAL MODEL
FROM_SCRATCH_MODEL: Path = FROM_SCRATCH / 'from_scratch_model.hdf5'

# TRAINING - MODELS - TRANSFER LEARNING
TRANSFER_LEARNING: Path = MODELS / 'transfer_learning'
TRANSFER_LEARNING_MODEL_WEIGHTS: Path = TRANSFER_LEARNING / 'weights.best.transfer_learning_model.hdf5'
TRANSFER_LEARNING_MODEL_HISTORY: Path = TRANSFER_LEARNING / 'transfer_learning_model_history.png'
FINE_TUNE_MODEL_HISTORY: Path = TRANSFER_LEARNING / 'fine_tune_model_history.png'

# TRAINING - MODELS - TRANSFER LEARNING - FINAL MODEL
TRANSFER_LEARNING_MODEL: Path = TRANSFER_LEARNING / 'transfer_learning_model.hdf5'
