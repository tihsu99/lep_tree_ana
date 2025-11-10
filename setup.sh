#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase

echo "=== running setupATLAS ==="
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh -q


lsetup git
lsetup cmake
kernel_info=$(uname -a)
if [[ $kernel_info =~ el7 ]] && [[ $kernel_info =~ x86_64 ]]; then
    lsetup "views LCG_105 x86_64-centos7-gcc11-opt"
elif [[ $kernel_info =~ el9 ]] && [[ $kernel_info =~ x86_64 ]]; then
    lsetup "views LCG_105 x86_64-el9-gcc11-opt"
else
    echo "Please define the LCG version for your system in setup.sh"
    echo "Your kernel_info is: $kernel_info"
fi

# source /eos/home-c/cmo/bbtautau/bbttml/run3mltoolkit/build_local_env.sh
export TREE_ANA_DIR=${SCRIPT_DIR}

export PATH=${PATH}:${TREE_ANA_DIR}/bin
export PYTHONPATH=${PYTHONPATH}:${TREE_ANA_DIR}:${TREE_ANA_DIR}/processor/:${TREE_ANA_DIR}/python
