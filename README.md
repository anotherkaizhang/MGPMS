# MGPMS

MGPMS implements a deep learning framework using Python and PyTorch. 

MGPMS has an advantage over traditional Machine Learning Models (Logistic Regression, Cox Regression, etc.) in that it can provide in-time robust risk score trajectory.

MGPMS can be used in numerous time series prediction applications in addition to the Risk Score prediction of performing Mechanical Ventilation in COVID-19 patients. 

For more details, see full paper: 

### Dependencies

PyTorch, pandas, matplotlib (for visualization) and all of their respective dependencies. 

### Running the tests

After installing, you can optionally run the test suite with

	py.test

from the command line while in the repo's main directory. 

## Running Experiments

Experiments are run using Docker containers built off of the [floydhub](https://github.com/floydhub/dl-docker) deep learning Docker images. DeepSurv can be run on either the CPU or the GPU with nvidia-docker. 

All experiments are in the `DeepSurv/experiments/` directory. 

To run an experiment, define the experiment name as an environmental variable `EXPRIMENT`and run the docker-compose file. Further details are in the `DeepSurv/experiments/` directory. 

## Input Dataset Format

Training DeepSurv can be done in a few lines. 
First, all you need to do is prepare the datasets to have the following keys:

One patient:
{
		'covs': covariates, length-n_covs vectors, N: patient number, n_covs: number of dimensions (dtype = float32), 
	 	'labels': 0/1 (dtype = int8),
	 	'times': observation time points, vector of sampling time points, not necessarily same length for each patient (dtype = float32),
    'values': observation value, sampling values at the above time points as a vector, not necessarily same length for each patient (dtype = float32),
    }
