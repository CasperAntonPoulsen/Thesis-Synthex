#!/bin/bash

BASE_PATH=${1-"/home/caap/Thesis-Synthex"}
OUTPUT_DIR=${2-"/home/data_shares/purrlab_students/LIDC-IDRI"}
COLLECTION=${3-"LIDC-IDRI"}



OPTS=""
# nbia 
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --collection-name ${COLLECTION}"



CARBON_LOG_DIR="${BASE_PATH}/carbon_logs/"

CARBONTRACKER_OPTS=""
# carbontracker (currently only 1 but in case more are added)
CARBONTRACKER_OPTS+=" --log_dir ${CARBON_LOG_DIR}"

CMD="python3 -m carbontracker.cli python3 ${BASE_PATH}/nbia_downloader.py ${OPTS} ${CARBONTRACKER_OPTS} $@"

echo ${CMD}
${CMD}
