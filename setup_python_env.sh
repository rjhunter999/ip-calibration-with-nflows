#!/bin/bash
PROJECT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
echo $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}/python"
