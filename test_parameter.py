# FILE MANAGEMENT 
FOLDER_NAME = 'test' # 3D_OUTDOOR_MAAC_DIST_SCALING_ALL_FEATURES
LOAD_FOLDER_NAME = 'teamwork_encoder_vistations-1' # 3D_OUTDOOR_MAAC_DIST_SCALING_ALL_FEATURES 
# debug is with np.max dist list, 3_17 is without modifying check done, 3_18 is the updated map data commit
model_path = f'model/{FOLDER_NAME}'
load_path = f'load_model/{LOAD_FOLDER_NAME}'
gifs_path = f'results/gifs/{LOAD_FOLDER_NAME}'

# MODEL PARAMETERS
NODE_INPUT_DIM = 14
EMBEDDING_DIM = 128
FRONTIER_CLUSTER_INPUT_DIM = 6

# HARDWARE PARAMETERS
USE_GPU_TEST = True  # Test using GPU
NUM_GPU = 1
NUM_META_AGENT = 1

# Test parameter
NUM_TEST = 100
NUM_RUN = 1
SAVE_CSV = True
SAVE_GIFS = True
N_AGENTS = 10
