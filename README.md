# MGPMS
![](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)
![](https://img.shields.io/badge/python-%3E%3D3.7-green)
![](https://img.shields.io/badge/torch-%3E%3D1.10-blue)
![](https://img.shields.io/badge/numpy-%3E%3D1.19-yellow)
![](https://img.shields.io/badge/matplotlib-%3E%3D3.3-brightgreen)
![](https://img.shields.io/badge/pandas-%3E%3D1.2-green)
![](https://img.shields.io/badge/scikit__learn-%3E%3D1.1-yellowgreen)
![](https://img.shields.io/badge/scipy-%3E%3D1.6-orange)
![](https://img.shields.io/badge/tqdm-%3E%3D4.60-lightgrey)

![](images/Figure 1-1.jpg?raw=true)


MGPMS implements a deep learning framwork that incorporates Multivariant Gaussian Process to the Transformer to do classification (and easily be tailored to do regression task) when the input time series data has missing values. 

The idea is to use Multivariant Gaussian Process for missing value imputation. The key note is that the Gaussian Process parameters (mean and covariance matrix) are integrated as parameters of the prediction neural network, therefore, missing data imputation and classification are performed together and jointly trained via back-propagation, which siginificantly boost prediction performance.

We applied it on a disease prediction task and achieved state-of-the-art performance on the same task compared to various deep learning and conventional machine learning models. [Real-time Prediction for Mechanical Ventilation in COVID-19 Patients using A Multi-task Gaussian Process Multi-objective Self-attention Network](https://pubmed.ncbi.nlm.nih.gov/35489596/).

In this paper, MGPMS was applied not only to classify (0/1) a future event happens or not, but also generate patient's entire risk score trajectory from the prediction time to the time event happens. We show the risk trajectory has siginificant robustness and consistency compared to traditional Machine Learning Models (Logistic Regression, Cox Regression, etc.) in that it uses target replication to further boost performance.

MGPMS can be used in numerous time series prediction applications when missing values exists, besides the COVID-19 patients Risk Score prediction in this paper.

### Requirements
* matplotlib>=3.3.3
* numpy>=1.19.2
* pandas>=1.2.4
* scikit_learn>=1.1.1
* scipy>=1.6.3
* torch>=1.10.0
* tqdm>=4.60.0


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
