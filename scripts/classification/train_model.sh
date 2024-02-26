#! /bin/bash

BASE_PATH=${1-"/dtu/p1/johlau/Thesis-Synthex"}
MODEL_NAME=${2-"pd"}


MODEL_DIR="${BASE_PATH}/classification/models/"
DATA_DIR="${BASE_PATH}/data/"


EPOCHS=${3-50}
LEARNING_RATE=${4-0.00001}
BATCH_SIZE=${5-128}
GAMMA=${6-0.8}


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-dir ${MODEL_DIR}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --epochs ${CKPT_NAME}"
OPTS+=" --learning-rate ${LEARNING_RATE}"
OPTS+=" --batch-size ${BATCH_SIZE}"


CARBON_LOG_DIR="${BASE_PATH}/carbon_logs/"

CARBONTRACKER_OPTS=""
# carbontracker (currently only 1 but in case more are added)
CARBONTRACKER_OPTS+=" --log_dir ${CARBON_LOG_DIR}"

CMD="python3 -m carbontracker.cli ${BASE_PATH}/classification/train_model.py ${OPTS} ${CARBONTRACKER_OPTS} $@"

echo ${CMD}
${CMD}
