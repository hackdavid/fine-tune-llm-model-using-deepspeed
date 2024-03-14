import glob
# dataset path having 2 keys prompt and response
DATA_FILES = '/path/dataset.jsonl'
# model path 
MODEL_PATH = '/PATH/MODEL'
SUB_RUN = 1
# path to save checkpoints 
OUTPUT_DIR=f'/path/save_checkpoint/folder'
GRAD_ACCUM=2
BATCH_SIZE=1
EPOCHS=5
MAX_SEQ_LEN=2048
SAVE_STEPS=2000
WEIGHT_DECAY=0.0001
LR_SHCEDULER_TYPE="cosine"
OPTIM="paged_adamw_32bit"
# TARGET_MODULES=["Wqkv", "out_proj", "up_proj", "down_proj"]
RANK=8
LORA_ALPHA=16
LEARNING_RATE=5e-5
RESPONSE_TEMPLATE = '\n\n### Response:'
LOAD_MODEL_IN_PEFT = False
DEEPSPPED_CONFIG_PATH = 'ds_configs/ds_zero2_config.json'