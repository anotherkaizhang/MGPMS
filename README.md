# MGPMS

MGPMS implements a deep learning framework using Python and PyTorch. 

MGPMS has an advantage over traditional Machine Learning Models (Logistic Regression, Cox Regression, etc.) in that it can provide in-time robust risk score trajectory.

MGPMS can be used in numerous time series prediction applications in addition to the Risk Score prediction of performing Mechanical Ventilation in COVID-19 patients. 

For more details, see full paper: *Real-time Prediction for Mechanical Ventilation in COVID-19 Patients using A Multi-task Gaussian Process Multi-objective Self-attention Network*

### Dependencies

PyTorch, pandas, matplotlib (for visualization) and all of their respective dependencies. 

## Input Dataset Format

Training DeepSurv can be done in a few lines. 
First, all you need to do is prepare the datasets to have the following keys:

One patient: a dictionary of the following key:value pairs
- 'covs': covariates, length-n_covs vectors, N: patient number, n_covs: number of dimensions (dtype = float32), 
- 'labels': 0/1 (dtype = int8),
- 'times': observation time points, vector of sampling time points, not necessarily same length for each patient (dtype = float32),
- 'values': observation value, sampling values at the above time points as a vector, not necessarily same length for each patient (dtype = float32),
- 'ind_lvs': index of a observational feature in the feature vector (dtype = int32),
- 'ind_times': index of a observational time point in the time vector (dtype = int32),
- 'meds_on_grid': vector of medicine prescriptions (dtype = float32),
- 'num_obs_values': vector of observational time vector (dtype = int32),
- 'rnn_grid_times': vector of time points at which the values need to be imputed (dtype = float32),
- 'num_rnn_grid_times': length of rnn_grid_times (dtype = int32),
