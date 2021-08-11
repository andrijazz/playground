#!/bin/bash

# check if storage path is specified
if [ "$#" -ne 3 ]; then
  echo "Illegal number of arguments. Please specify project name, storage drive path and wandb api key"
  exit 1
fi

# args
PROJECT=$1
echo "PROJECT=${PROJECT}"
STORAGE=$2
echo "STORAGE=${STORAGE}"
WANDB_API_KEY=$3
echo "WANDB_API_KEY=${WANDB_API_KEY}"

# generate vars
PROJECT_STORAGE_ROOT="${STORAGE}/${PROJECT}"
echo "PROJECT_STORAGE_ROOT=${PROJECT_STORAGE_ROOT}"
SRC_ROOT=${PWD}
echo "SRC_ROOT=${SRC_ROOT}"
DATASETS="${PROJECT_STORAGE_ROOT}/datasets"
echo "DATASETS=${DATASETS}"
MODELS="${PROJECT_STORAGE_ROOT}/models"
echo "MODELS=${MODELS}"
LOG="${PROJECT_STORAGE_ROOT}/log"
echo "LOG=${LOG}"
RAY_LOG="${LOG}/ray"
echo "RAY_LOG=${RAY_LOG}"

# create storage path in case it doesn't exists
mkdir -p "${PROJECT_STORAGE_ROOT}"
mkdir -p "${DATASETS}"
mkdir -p "${MODELS}"
mkdir -p "${LOG}"
mkdir -p "${RAY_LOG}"

# create .env file in project root and add vars to it
ENV="${SRC_ROOT}/.env"
echo "SRC_ROOT=${PWD}
PROJECT=${PROJECT}
STORAGE=${STORAGE}
PROJECT_STORAGE_ROOT=${PROJECT_STORAGE_ROOT}
DATASETS=${DATASETS}
MODELS=${MODELS}
LOG=${LOG}
RAY_LOG=${RAY_LOG}
WANDB_API_KEY=${WANDB_API_KEY}
PYTHONPATH=\${SRC_ROOT}:\${PYTHONPATH}" > "${ENV}"

# LD_LIBRARY_PATH=/home/${USER}/.mujoco/mujoco200/bin:${LD_LIBRARY}" > "${ENV}"

# export - needed for ray cluster
# echo "export SRC_ROOT=${PWD}" >> ~/.bashrc
# echo "export PROJECT=${PROJECT}" >> ~/.bashrc
# echo "export STORAGE=${STORAGE}" >> ~/.bashrc
# echo "export PROJECT_STORAGE_ROOT=${PROJECT_STORAGE_ROOT}" >> ~/.bashrc
# echo "export DATASETS=${DATASETS}" >> ~/.bashrc
# echo "export MODELS=${MODELS}" >> ~/.bashrc
# echo "export LOG=${LOG}" >> ~/.bashrc
# echo "export RAY_LOG=${RAY_LOG}" >> ~/.bashrc
# echo "export WANDB_API_KEY=${WANDB_API_KEY}" >> ~/.bashrc
# echo "export PYTHONPATH=\${SRC_ROOT}:\${PYTHONPATH}" >> ~/.bashrc
# echo "export LD_LIBRARY_PATH=/home/${USER}/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}" >> ~/.bashrc
