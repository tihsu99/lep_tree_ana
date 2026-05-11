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

source ${SCRIPT_DIR}/setup.sh