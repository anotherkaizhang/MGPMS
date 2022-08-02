
import numpy as np
import pickle
import random
def OU_kernel_np(length,x):
    """ just a correlation function, for identifiability
    """
    x1 = np.reshape(x,[-1,1]) #colvec
    x2 = np.reshape(x,[1,-1]) #rowvec
    K_xx = np.exp(-np.abs(x1-x2)/length)
    return K_xx

def sim_multitask_GP(times, length, noise_vars, K_f, trainfrac):
    """
    draw from a multitask GP.

    we continue to assume for now that the dim of the input space is 1, ie just time

    M: number of tasks (labs/vitals/time series)

    train_frac: proportion of full N * M data matrix Y to include

    K_f is M x M matrix.
    """
    M = np.shape(K_f)[0]  # No. of variables
    N = len(times)  # No. of observations for this patient
    n = N * M  # total amount of data
    K_t = OU_kernel_np(length, times)  # correlation function if N by N matrix
    Sigma = np.diag(noise_vars)  # matrix of M x M

    K = np.kron(K_f, K_t) + np.kron(Sigma, np.eye(N)) + 1e-6 * np.eye(n)  # matrix of n x n
    L_K = np.linalg.cholesky(K)  # matrix of n by n

    y = np.dot(L_K, np.random.normal(0, 1, n))  # a vector of n numbers, Draw normal


    # get indices of which time series and which time point, for each element in y
    ind_kf = np.tile(np.arange(M), (N, 1)).flatten('F')  # vec by column
    ind_kx = np.tile(np.arange(N), (M, 1)).flatten()

    # randomly dropout some fraction of fully observed time series
    perm = np.random.permutation(n)  # n=M*N 
    n_train = int(trainfrac * n)  # 80% discard
    train_inds = perm[:n_train]

    y_ = y[train_inds]
    ind_kf_ = ind_kf[train_inds]
    ind_kx_ = ind_kx[train_inds]

    return y_, ind_kf_, ind_kx_

def gen_MGP_params(M):
    """
    Generate some MGP params for each class.
    Assume MGP is stationary and zero mean, so hyperparams are just:
        Kf: covariance across time series: (2, M, M)  # 2 classes, each is M x M covariance matrix
        length: length scale for shared kernel across time series (2, M)  # 2 classes, each M time stamp
        noise: noise level for each time series: (2, )  # 2 classes, noise level at each time stamp
    """
    true_Kfs = []
    true_noises = []
    true_lengths = []

    # Class 0
    tmp = rs.normal(0, .2, (M, M))  # RandomState.normal(loc=0.0, scale=1.0, size=None)
    true_Kfs.append(np.dot(tmp, tmp.T))
    true_lengths.append(6.0)
    true_noises.append(np.linspace(.02, .08, M))

    # Class 1
    tmp = rs.normal(0, .8, (M, M))
    true_Kfs.append(np.dot(tmp, tmp.T))  # matrix: M x M
    true_lengths.append(2.0)  # one value
    true_noises.append(np.linspace(.1, .12, M))  # vector of length: M

    return true_Kfs, true_noises, true_lengths


def sim_dataset(num_encs, stay_length, M, n_covs, n_meds, pos_class_rate=0.5, trainfrac=0.05):
    true_Kfs, true_noises, true_lengths = gen_MGP_params(M)

    # end_times = np.random.randint(low=0, high=stay_length-1, size=num_encs)  # end observational time of 17 patients
    # print(end_times)

    num_obs_times = np.random.randint(low=10, high=stay_length-1, size=num_encs)   # No. of observational times of 17 patients
    # print(num_obs_times)
    num_obs_values = np.array(num_obs_times * M * trainfrac,
                              dtype="int")  # No. of observed labs of 17 patients, Should be M, trainfrac is how much fraction is observed,
    # trainfrac = 0.2 means only 20% of the total M lab tests are observed
    # number of inputs to RNN. will be a grid on integers, starting at 0 and ending at next integer after end_time
    num_rnn_grid_times = stay_length * np.ones((num_encs,), # array(np.floor(end_times) + 1,
                                  dtype="int")  # Grid (int) time stamps of 17 patients, because the above obs_time is float
    # print(num_rnn_grid_times)

    rnn_grid_times = []
    labels = rs.binomial(n=1, p=pos_class_rate, size=num_encs)  # labels of 17 patients, 0/1

    T = []  # actual observation times
    Y = []
    ind_kf = []
    ind_kt = []  # actual data; indices pointing to which lab, which time
    baseline_covs = np.zeros((num_encs, n_covs))
    # each contains an array of size num_rnn_grid_times x n_meds
    #   simulates a matrix of indicators, where each tells which meds have been given between the
    #   previous grid time and the current.  in practice you will have actual medication administration
    #   times and will need to convert to this form, for feeding into the RNN
    meds_on_grid = []

    print('Simming data...')
    for i in range(num_encs):  # each patient of the 17
        if i % 500 == 0:
            print('%d/%d' % (i, num_encs))

        obs_times = np.sort(random.sample(range(stay_length), num_obs_times[i]))

        T.append(obs_times)
        l = labels[i]

        y_i, ind_kf_i, ind_kt_i = sim_multitask_GP(obs_times, true_lengths[l], true_noises[l], true_Kfs[l], trainfrac)

        # Append this patient to result
        Y.append(y_i)
        ind_kf.append(ind_kf_i)
        ind_kt.append(ind_kt_i)

        # build rnn grid for this patient, append to result
        rnn_grid_times.append(np.arange(num_rnn_grid_times[i]))

        if l == 0:  # sim some different baseline covs; meds for 2 classes
            baseline_covs[i, : int(n_covs / 2)] = rs.normal(0.1, 0.2, int(n_covs / 2))
            baseline_covs[i, int(n_covs / 2):] = rs.binomial(1, 0.2, int(n_covs / 2))
            meds = rs.binomial(1, .02, (num_rnn_grid_times[i], n_meds))  # This is a 0/1 matrix, row: time, col: 5 medicine
        else:
            baseline_covs[i, :int(n_covs / 2)] = rs.normal(0.8, 0.8, int(n_covs / 2))
            baseline_covs[i, int(n_covs / 2):] = rs.binomial(1, 0.6, int(n_covs / 2))
            meds = rs.binomial(n=1, p=.24, size=(num_rnn_grid_times[i], n_meds))

        meds_on_grid.append(meds)


    return (num_obs_times, num_obs_values, num_rnn_grid_times, rnn_grid_times,
            labels, T, Y, ind_kf, ind_kt, meds_on_grid, baseline_covs)

if __name__ == '__main__':
    seed = 8675309
    rs = np.random.RandomState(seed) #fixed seed in np

    num_encs = 5000
    M = 25  # number of features in the lab test table （matrix)
    n_covs = 22  # number of features in the demographics (vector)
    n_meds = 10  # number of features in the medication table (matrix)
    stay_length = 42  # number of features in the medication table (matrix)

    (num_obs_times, num_obs_values, num_rnn_grid_times, rnn_grid_times, labels, times,
       values, ind_lvs, ind_times, meds_on_grid, covs) = sim_dataset(num_encs,stay_length, M, n_covs, n_meds)

    ### Normalization
    values_all = []
    for v in values:
        values_all.extend(v)
    values_mean = np.mean(values_all)
    values_std = np.std(values_all)
    for i in range(len(values)):
        values[i] = (values[i] - values_mean)/values_std
#     print(values_mean, values_std)

    covs_all = []
    for v in covs:
        covs_all.extend(v)
    covs_mean = np.mean(covs_all)
    covs_std = np.std(covs_all)
    for i in range(len(covs)):
        covs[i] = (covs[i] - covs_mean)/covs_std
#     print(covs_mean, covs_std)



    # print('num_obs_times', num_obs_times)
    # print('num_obs_values',num_obs_values)
    # print('num_rnn_grid_times',num_rnn_grid_times)
    # print('rnn_grid_times',rnn_grid_times)
    # print('labels',labels)
    # print('times',times)
    # print('values',values)
    # print('ind_lvs',ind_lvs)
    # print('ind_times',ind_times)
    # print('meds_on_grid',meds_on_grid)
    # print('covs',covs)
    # print(rnn_grid_times)
    rnn_grid_times = [np.int64(x) for x in rnn_grid_times]
    times = [np.float64(x) for x in times]
    values = [np.float64(x) for x in values]
    ind_lvs = [np.int64(x) for x in ind_lvs]
    ind_times = [np.int64(x) for x in ind_times]
    meds_on_grid = [np.float64(x) for x in meds_on_grid]
    covs = [np.float64(x) for x in covs]

    # print('num_obs_times', num_obs_times[0].dtype)
    # print('num_obs_values', num_obs_values[0].dtype)
    # print('num_rnn_grid_times', num_rnn_grid_times[0].dtype)
    print('rnn_grid_times', rnn_grid_times[0].dtype)
    # print('labels', labels[0].dtype)
    print('times', times[0].dtype)
    print('values', values[0].dtype)
    print('ind_lvs', ind_lvs[0].dtype)
    print('ind_times', ind_times[0].dtype)
    print('meds_on_grid', meds_on_grid[0].dtype)
    print('covs', covs[0].dtype)


    result = dict()
    result['num_obs_times'] = num_obs_times
    result['num_obs_values'] = num_obs_values
    result['num_rnn_grid_times'] = num_rnn_grid_times
    result['rnn_grid_times'] = rnn_grid_times
    result['labels'] = labels
    result['times'] = times
    result['values'] = values
    result['ind_lvs'] = ind_lvs
    result['ind_times'] = ind_times
    result['meds_on_grid'] = meds_on_grid
    result['covs'] = covs

    f = open("data/input_for_MGPMS-simulation_M"+str(M)+"_cov"+str(n_covs)+"_med"+str(n_meds)+".pickle", 'wb')
    pickle.dump(result, f, protocol=3)
    f.close()

    input('Finish simulation!')
