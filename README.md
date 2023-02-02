# Requirements (additional packages not required in the netCDF tutorial)
# these two links are specifications for using the deformetrica package:
conda create -n deformetrica python=3.8 numpy && source activate deformetrica
pip install deformetrica
# python package requirements (without deformetrica)
pip install xarray numpy pandas matplotlib notebook jupytext netcdf4 dill

# Data

## Inputs

The original netCDF files must be put in a folder in the parent folder of where this script is located:
`../InitialData`

# Usage

## 1) create pkl files
`python transform.py`

## 2)
`python main_flow.py`
This will create outputs in `../bulk_image` (that folder will be created by the script).


## soon, pkl to nc . py ; will convert to .nc with new name
