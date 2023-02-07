# Requirements
## python package requirements
pip install xarray numpy pandas matplotlib notebook jupytext netcdf4 scikit-image dill

## NB, for the deformetrica package:
conda create -n deformetrica python=3.8 numpy && source activate deformetrica
pip install deformetrica


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

## 3) create the .nc files
Use pickel2nc.py, and use an index from 0 to 19.

