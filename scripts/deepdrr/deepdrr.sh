#! /bin/bash

BASE_PATH=${1-"/dtu/p1/johlau/Thesis-Synthex"}
COLLECTTION=${3-"LIDC-IDRA"}
OUTPUT_DIR=${2-"/home/data_shares/purrlab_students/LIDC-IDRI"}

OPTS=""
# nbia 
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --output-dir ${MODEL_DIR}"
OPTS+=" --collection-name ${COLLECTION}"



CARBON_LOG_DIR="${BASE_PATH}/carbon_logs/"

CARBONTRACKER_OPTS=""
# carbontracker (currently only 1 but in case more are added)
CARBONTRACKER_OPTS+=" --log_dir ${CARBON_LOG_DIR}"

CMD="python3 -m carbontracker.cli ${BASE_PATH}/data/nbia_downloader.py ${OPTS} ${CARBONTRACKER_OPTS} $@"

echo ${CMD}
${CMD}
