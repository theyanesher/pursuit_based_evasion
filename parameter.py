# FILE MANAGEMENT 
FOLDER_NAME = 'teamwork_encoder_vistations_recenccy' # 3D_OUTDOOR_MAAC_DIST_SCALING_ALL_FEATURES
LOAD_FOLDER_NAME = 'teamwork_encoder_vistations-1' # 3D_OUTDOOR_MAAC_DIST_SCALING_ALL_FEATURES 
# debug is with np.max dist list, 3_17 is without modifying check done, 3_18 is the updated map data commit
# 3_20_2 is with distance
# 3_26 is new node generation
model_path = f'model/{FOLDER_NAME}'
load_path = f'load_model/{LOAD_FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
USE_WANDB = False
LOAD_MODEL = False  # do you want to load the model trained before
TEST_SET = False

# DATA COLLECTION
SUMMARY_WINDOW = 32
SAVE_IMG_GAP = 500
SIM_STEP = 5

# ROBOT PARAMETERS
N_AGENTS = 6
MIN_FLIGHT_HGT = 0    # in meter
MAX_FLIGHT_HGT = 60   # in meter
MIN_BUILDING_HGT_CELL = 0 # in pixels
FLT_SAFETY_BUFFER = 0
TRAJECTORY_LEN = 5

# SENSOR PARAMETERS
CELL_SIZE = 1  # meter 0.85
SENSOR_RANGE = 40  # meter, 35
UTILITY_RANGE = SENSOR_RANGE * 0.8
MIN_UTILITY = 1
FRONTIER_CELL_SIZE = 4 * CELL_SIZE
SAFETY_RANGE = 0.8 * SENSOR_RANGE

# MAP PARAMETERS
QUAD_TREE_SIZE = 1000
NODE_RESOLUTION = 10.0 # meter, 16
DOWNSAMPLE_SIZE = NODE_RESOLUTION // CELL_SIZE
LOCAL_MAP_SIZE = 60  # meter
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION
NUM_NODE_NEIGHBORS = 5
K_SIZE = NUM_NODE_NEIGHBORS**2   # the number of neighboring nodes
NUM_BORDER_POINTS = 3
PERCENTAGE_BUILDING_HEIGHT = 60  # poercentage of building height to be considered as lower than the drone
SOUTH_OPENING_SIZE = 0    # metres
MIN_BUILDING_SIZE = 35     # cells
MAX_CLUSTER_SIZE = 12

# TRAINING PARAMETERS
MAX_EPISODE_STEP = 100
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 1000
BATCH_SIZE = 512
LR = 1e-5
GAMMA = 1

# MODEL PARAMETERS
NODE_INPUT_DIM = 14
FRONTIER_CLUSTER_INPUT_DIM = 6
EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 380  # the number of nodes will be padded to this value
CLUSTERS_PADDING_SIZE = 55

# HARDWARE PARAMETERS
USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 2
NUM_META_AGENT = 28

# Training algo
TRAIN_ALGO = 1 # 0: normal RL, 1: MAAC, 2: GT, 3: GT and MAAC
EPISODE_BUFFER_SIZE = 26   #  30 for GT