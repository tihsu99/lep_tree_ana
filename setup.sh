#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export TREE_ANA_DIR=${SCRIPT_DIR}

export PATH=${PATH}:${TREE_ANA_DIR}/bin
export PYTHONPATH=/afs/cern.ch/user/c/cmo/.local/lib/python3.11/site-packages/:$PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:${TREE_ANA_DIR}:${TREE_ANA_DIR}/processor/:${TREE_ANA_DIR}/python

if [[ -f "${TREE_ANA_DIR}/RooUnfold/build/setup.sh" ]]; then
    source "${TREE_ANA_DIR}/RooUnfold/build/setup.sh"
else
    echo "RooUnfold is present but not built yet. Run: "
    echo "cd ${TREE_ANA_DIR}/RooUnfold"
    echo "mkdir -p build && cd build && cmake ../"
    echo "make -j4"
    echo "source setup.sh"
fi
