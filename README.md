# MGPMS
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.6](https://img.shields.io/badge/python-%3E%3D3.4.5-yellow)


matplotlib==3.3.3
numpy==1.19.2
pandas==1.2.4
scikit_learn==1.1.1
scipy==1.6.3
torch==1.10.0
tqdm==4.60.0
python==3.7




MGPMS implements a deep learning framework using Python and PyTorch: [Real-time Prediction for Mechanical Ventilation in COVID-19 Patients using A Multi-task Gaussian Process Multi-objective Self-attention Network](https://arxiv.org/abs/2102.01147).

MGPMS has an advantage over traditional Machine Learning Models (Logistic Regression, Cox Regression, etc.) in that it can provide in-time robust risk score trajectory.

MGPMS can be used in numerous time series prediction applications in addition to the Risk Score prediction of performing Mechanical Ventilation in COVID-19 patients. 

For more details, see full paper: *Real-time Prediction for Mechanical Ventilation in COVID-19 Patients using A Multi-task Gaussian Process Multi-objective Self-attention Network*

### General
* Python >= 3.5
* [PyTorch](https://pytorch.org/get-started/locally/) (tested with PyTorch 1.7.1)
* Optional: CUDA (tested with CUDA 11.0) 

## Input Dataset Format
Training DeepSurv can be done in a few lines. 
First, all you need to do is prepare the datasets to have the following keys:

A dictionary of the following keys: values, each value is a list of N patients' data
- 'covs': list of N covariates (length-n_covs vector), n_covs: number of demographic dimensions (dtype = float64), 
- 'labels': list of N binaries (0/1) (dtype = int8),
- 'times': list of N observational time points (vector of various lengths), usually vector length is not the same for all patients (dtype = float64),
- 'values': list of N observational observational values (sampling values at the above time points as a vector, usually vector length is not the same for all patients (dtype = float64),
- 'ind_lvs': list of N observational value indices (length-num_obs_values vector), index of a observational feature in the 'values' vector (dtype = int64),
- 'ind_times': list of N observational time indices (length-num_obs_values vector), index of a observational time point in the 'times' vector (dtype = int64),
- 'meds_on_grid': list of N drug prescription history time indices (matrix of rnn_grid_times-by-n_meds), (dtype = float64),
- 'num_obs_values': list of N integers (dtype = int32),
- 'rnn_grid_times': list of N vectors, vector of time points at which the values need to be imputed (dtype = float64),
- 'num_rnn_grid_times': length of rnn_grid_times (dtype = int32),
Check out the Jupyter Notebook `fastPC Demo` to see a demonstration of the functionality. 


The folder 'data' contains one small dataset for demonstration purposes.

### Running

Run `mgp-ms.py` to run MGP-MS, you can custermize parameters and your own dataset(s).
