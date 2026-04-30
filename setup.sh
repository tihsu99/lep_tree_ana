#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase

echo "=== running setupATLAS ==="
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh -q


lsetup git
lsetup cmake
kernel_info=$(uname -a)
if [[ $kernel_info =~ el9 ]] && [[ $kernel_info =~ x86_64 ]]; then
    lsetup "views LCG_106 x86_64-el9-gcc13-opt"
else
    echo "Please define the LCG version for your system in setup.sh"
    echo "Your kernel_info is: $kernel_info"
fi

# source /eos/home-c/cmo/bbtautau/bbttml/run3mltoolkit/build_local_env.sh
export TREE_ANA_DIR=${SCRIPT_DIR}

export PATH=${PATH}:${TREE_ANA_DIR}/bin
# build your own vector 1.6.1 if not accessible to the path below
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
