from yacs.config import CfgNode as ConfigurationNode
from src.data.load_datasets import update_cfg_using_dotenv
from pathlib import Path

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

__C = ConfigurationNode()

# importing default as a global singleton
cfg = __C
__C.DESCRIPTION = 'Default config from the Singleton'
__C.DATASET = ConfigurationNode()
__C.DATASET.NAME = 'bengali_kaggle'
__C.DATASET.DEFAULT_SIZE = (137, 236)
__C.DATASET.RESIZE_SHAPE = (128, 128)
__C.DATASET.CONCENTRATE_CROP = True
__C.DATASET.GRAPHEME_SIZE = 168
__C.DATASET.VOWEL_SIZE = 11
__C.DATASET.CONSONANT_SIZE = 7
__C.DATASET.TRAIN_DATA_PATH = 'C:/Users/mingy/Documents/ml_data/bengali/train_data.p'
__C.DATASET.VAL_DATA_PATH = 'C:/Users/mingy/Documents/ml_data/bengali/val_data.p'

__C.DATASET.AUGMENTATION = ConfigurationNode()
__C.DATASET.AUGMENTATION.BLURRING_PROB = 0.25
__C.DATASET.AUGMENTATION.GAUSS_NOISE_PROB = 0.25
__C.DATASET.AUGMENTATION.BRIGHTNESS_CONTRAST_PROB = 1
__C.DATASET.AUGMENTATION.GRID_DISTORTION_PROB = 1
__C.DATASET.AUGMENTATION.ROTATION_PROB = 1
__C.DATASET.AUGMENTATION.ROTATION_DEGREE = 20

__C.DATASET.BATCH_SIZE = 32
__C.DATASET.CPU_NUM = 1
__C.DATASET.TO_RGB = True
__C.DATASET.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
__C.DATASET.NORMALIZE_STD = [0.229, 0.224, 0.225]
__C.DATASET.FOCUS_CLASS = []

__C.MODEL = ConfigurationNode()

__C.MODEL.PARALLEL = False
__C.MODEL.META_ARCHITECTURE = 'baseline'
__C.MODEL.NORMALIZATION_FN = 'BN'

__C.MODEL.BACKBONE = ConfigurationNode()
__C.MODEL.BACKBONE.NAME = 'mobilenet_v2'
__C.MODEL.BACKBONE.RGB = True
__C.MODEL.BACKBONE.PRETRAINED_PATH = r'C:\Git\bengali.ai\models\mobilenet_v2-b0353104.pth'

__C.MODEL.HEAD = ConfigurationNode()
__C.MODEL.HEAD.NAME = 'simple_head'
__C.MODEL.HEAD.ACTIVATION = 'leaky_relu'
__C.MODEL.HEAD.OUTPUT_DIMS = [168, 11, 7]
__C.MODEL.HEAD.INPUT_DIM = 1280  # mobilenet V2
__C.MODEL.HEAD.HIDDEN_DIMS = [512, 256]
__C.MODEL.HEAD.BN = True
__C.MODEL.HEAD.DROPOUT = -1

__C.MODEL.SOLVER = ConfigurationNode()
__C.MODEL.SOLVER.OPTIMIZER = 'adam'
__C.MODEL.SOLVER.BASE_LR = 0.001
__C.MODEL.SOLVER.TOTAL_EPOCHS = 40
__C.MODEL.SOLVER.AMP = False

__C.MODEL.SOLVER.LOSS = ConfigurationNode()
__C.MODEL.SOLVER.LOSS.OHEM_RATE = 1.0
__C.MODEL.SOLVER.LOSS.NAME = 'xentropy'
__C.MODEL.SOLVER.LOSS.LABELS_WEIGHTS_PATH = 'C:/Users/mingy/Documents/ml_data/bengali/labels_weights.p'

# focal loss related
__C.MODEL.SOLVER.LOSS.FOCAL_LOSS = ConfigurationNode()
__C.MODEL.SOLVER.LOSS.FOCAL_LOSS.GAMMA = 1
__C.MODEL.SOLVER.LOSS.FOCAL_LOSS.ALPHA = -1

__C.OUTPUT_PATH = ''
__C.RESUME_PATH = ''


def get_cfg_defaults():
  """
  Get a yacs CfgNode object with default values for my_project.
  """
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern recommended by the YACS repo.
  # It will be subsequently overwritten with local YAML.
  return __C.clone()


def combine_cfgs(path_cfg_data: Path=None, path_cfg_override: Path=None):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_output: path to output files
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :return: cfg_base incorporating the overwrite.
    """

    # Path order of precedence is:
    # Priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 4:
    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    # Priority 3:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 2:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # Merge from .env
    # Priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg_base.merge_from_list(list_cfg)

    return cfg_base