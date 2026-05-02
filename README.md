# LEP_tree_ana

This package filters and analyzes LEP data (*_ttree.root) from the parton-level.

## Environment

You can set up the analysis environment in either of two ways.

### Option 1: build a Python environment from `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
source setup.sh
```

The setup script adds this package to `PATH` and `PYTHONPATH`, and sources the local RooUnfold build if it is available. If RooUnfold has not been built yet, follow the commands printed by `setup.sh`.

### Option 2: use the existing CVMFS environment

On CERN-style systems with CVMFS available, source the setup script directly:

```bash
cd /home/cmo/projects/LEP/tree_ana/tree_ana
source setup_cvmfs.sh
```

This uses `/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase` and the configured LCG view to provide ROOT/PyROOT and the common Python analysis stack.

## Usage
After sourcing the setup script, modify `bin/tree_ana` to set the input and output file names, then run:
```bash
tree_ana
```
