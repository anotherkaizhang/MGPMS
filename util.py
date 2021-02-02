import torch
import torch.nn.functional as F
import numpy as np
import globals

def pad_rawdata(T, Y, ind_kf, ind_kt, X, meds_on_grid):
    N = len(T)
    num_meds = meds_on_grid[0].shape[1]

    T_lens = np.array([len(t) for t in T])
    T_maxlen = np.max(T_lens)
    T_pad = np.zeros((N, T_maxlen))

    Y_lens = np.array([len(y) for y in Y])
    Y_maxlen = np.max(Y_lens)
    Y_pad = np.zeros((N, Y_maxlen))
    ind_kf_pad = np.zeros((N, Y_maxlen))
    ind_kt_pad = np.zeros((N, Y_maxlen))

    grid_lens = np.array([np.shape(m)[0] for m in meds_on_grid])
    grid_maxlen = np.max(grid_lens)
    meds_pad = np.zeros((N, grid_maxlen, num_meds))
    X_pad = np.zeros((N, grid_maxlen))

    for i in range(N):
        T_pad[i, :T_lens[i]] = T[i]
        Y_pad[i, :Y_lens[i]] = Y[i]
        ind_kf_pad[i, :Y_lens[i]] = ind_kf[i]
        ind_kt_pad[i, :Y_lens[i]] = ind_kt[i]
        X_pad[i, :grid_lens[i]] = X[i]
        meds_pad[i, :grid_lens[i], :] = meds_on_grid[i]

    return T_pad, Y_pad, \
           ind_kf_pad, ind_kt_pad, \
           X_pad, meds_pad


def OU_kernel(length, x1, x2):
    x1 = torch.reshape(x1, (-1, 1))  # colvec
    x2 = torch.reshape(x2, (1, -1))  # rowvec
    K = torch.exp(-torch.abs(x1 - x2) / length)
    return K


def SE_kernel(length, x1, x2):
    x1 = torch.reshape(x1, (-1, 1))  # colvec
    x2 = torch.reshape(x2, (1, -1))  # rowvec
    K = torch.exp(-torch.pow(x1 - x2, 2.0) / length)
    return K

def gather_nd(K, ind):
    ind = ind.type(torch.long)
    return K[list(ind.T)].T

def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))

        return m + torch.log(sum_exp)


def CG(A, b):
    b = torch.reshape(b, (-1,))
    n = A.shape[0]
    x = torch.zeros((n,), device=globals.device)
    r = b
    p = r

    CG_EPS = n / 1000.0
    MAX_ITER = n / 250 + 3

    i = 0

    while i < MAX_ITER and torch.norm(r) > CG_EPS:
        p_vec = torch.reshape(p, (-1, 1))
        Ap = torch.reshape(torch.mm(A, p_vec), (-1,))  # make a vector
        alpha = torch.dot(r, r) / torch.dot(p, Ap)
        x = x + alpha * p
        r2 = r - alpha * Ap
        beta = torch.dot(r2, r2) / torch.dot(r, r)
        r = r2
        p = r + beta * p
        i += 1

    return torch.reshape(x, (-1, 1))


def block_CG(A_, B_):
    n = B_.shape[0]
    m = B_.shape[1]

    X = torch.zeros((n, m), device=globals.device)
    V_ = torch.zeros((n, m), device=globals.device)
    R = B_
    R_ = B_

    CG_EPS = n / 1000.0
    MAX_ITER = n / 250 + 3

    i = 0
    while i < MAX_ITER and torch.norm(R) > CG_EPS:
        S = torch.solve(torch.mm(torch.transpose(R, 0, 1), R),
                        torch.mm(torch.transpose(R_, 0, 1), R_))[0]
        V = R + torch.mm(V_, S)
        T = torch.solve(torch.mm(torch.transpose(R, 0, 1), R),
                        torch.mm(torch.transpose(V, 0, 1), torch.mm(A_, V)))[0]
        X = X + torch.mm(V, T)
        V_ = V
        R_ = R
        R = R - torch.mm(A_, torch.mm(V, T))
        i += 1

    return X


def Lanczos(Sigma_func, b):
    n = b.shape[0]
    k = n / 500 + 3

    betas = torch.zeros(1, device=globals.device)
    alphas = torch.zeros(0, device=globals.device)
    D = torch.zeros((n, 1), device=globals.device)

    b_norm = torch.norm(b)
    D = torch.cat((D, torch.reshape(b / b_norm, (-1, 1))), 1)

    j = 1
    while j < k + 1:
        d_j = D[:, j:j + 1]
        d = Sigma_func(d_j) - betas[-1] * D[:, j - 1:j]
        alphas = torch.cat((alphas, [torch.dot(d_j, d)]), 0)
        d = d - alphas[-1] * d_j
        betas = torch.cat((betas, [torch.norm(d)]), 0)
        D = torch.cat((D, d / betes[j:j + 1]), 1)
        j += 1

    betas_ = torch.diag(betas[1:k])
    D_ = D[:, 1:k + 1]

    H = torch.diag(alphas) + F.pad(betas_, (0, 1, 1, 0)) + F.pad(betas_, (1, 0, 0, 1))

    e, v = torch.symeig(H, eigenvectors=True)
    e_pos = torch.max(0.0, e) + 1e-6
    e_sqrt = torch.diag(torch.sqrt(e_pos))
    sq_H = torch.mm(v, torch.mm(e_sqrt, torch.transpose(v, 0, 1)))

    out = b_norm * torch.mm(D_, sq_H)
    return out[:, 0:1]


def block_Lanczos(Sigma_func, B_, n_mc_smps):
    n = B_.shape[0]
    s = n_mc_smps
    k = int(n / 500 + 3)

    betas = torch.zeros((1, s), device=globals.device)
    alphas = torch.zeros((0, s), device=globals.device)
    D = torch.zeros((s, n, 1), device=globals.device)

    B_norms = torch.norm(B_, dim=0)
    D = torch.cat((D, torch.unsqueeze(torch.transpose(B_ / B_norms, 0, 1), 2)), 2)

    j = 1
    while j < k + 1:
        d_j = torch.squeeze(D[:, :, j:j + 1])
        d = Sigma_func(torch.transpose(d_j, 0, 1)) - betas[j - 1:j, :] * \
            torch.transpose(torch.squeeze(D[:, :, j - 1:j]), 0, 1)
        alphas = torch.cat((alphas, torch.diagonal(torch.mm(d_j, d)).unsqueeze(0)), 0)
        d = d - alphas[j - 1:j, :] * torch.transpose(d_j, 0, 1)
        betas = torch.cat((betas, torch.norm(d, dim=0).unsqueeze(0)), 0)
        D = torch.cat((D, torch.transpose(d / betas[j:j + 1, :], 0, 1).unsqueeze(2)), 2)
        j += 1

    D_ = D[:, :, 1:1 + k]

    H = torch.zeros((0, k, k), device=globals.device)

    for ss in range(s):
        this_beta = torch.diag(torch.squeeze(betas[1:k, ss:ss + 1]))
        this_H = (torch.diag(torch.squeeze(alphas[:, ss:ss + 1])) +
                  F.pad(this_beta, (0, 1, 1, 0)) +
                  F.pad(this_beta, (1, 0, 0, 1)))
        H = torch.cat((H, this_H.unsqueeze(0)), 0)

    E, V = torch.symeig(H, eigenvectors=True)  # !!!different from 'torch.eig'
    E_sqrt = torch.zeros((0, k, k), device=globals.device)

    for ss in range(s):
        E_sqrt = torch.cat((E_sqrt, torch.diag(torch.squeeze(
            torch.sqrt(torch.max(E[ss:ss + 1, :], 1e-6 * torch.ones_like(E[ss:ss + 1, :], device=globals.device))))).unsqueeze(
            0)), 0)

    sq_H = torch.matmul(V, torch.matmul(E_sqrt, V.permute(0, 2, 1)))

    e1 = torch.transpose(torch.eye(k, device=globals.device)[:, 0:1].repeat(1, s), 0, 1).unsqueeze(
        2)

    out = B_norms * torch.transpose(torch.squeeze(torch.matmul(D_, torch.matmul(sq_H, e1))), 0, 1)
    return out


def get_GP_samples(Y, T, X, ind_kf, ind_kt, num_obs_times, num_obs_values,
                   num_rnn_grid_times, med_grid,
                   length, noises, Kf,
                   n_mc_smps, M, n_meds, sequence_len):
    Z = torch.zeros((0, sequence_len, M+n_meds), device=globals.device)
    N = T.shape[0]
    ind_kf = ind_kf.type(torch.long)
    ind_kt = ind_kt.type(torch.long)

    i = 0
    while i < N:
        Yi = torch.reshape(Y[i:i + 1, 0:num_obs_values[i]], (-1,))
        Ti = torch.reshape(T[i:i + 1, 0:num_obs_times[i]], (-1,))
        ind_kfi = torch.reshape(ind_kf[i:i + 1, 0:num_obs_values[i]], (-1,))
        ind_kti = torch.reshape(ind_kt[i:i + 1, 0:num_obs_values[i]], (-1,))
        Xi = torch.reshape(X[i:i + 1, 0:num_rnn_grid_times[i]], (-1,))
        X_len = num_rnn_grid_times[i]

        GP_draws = draw_GP(Yi, Ti, Xi, ind_kfi, ind_kti, length, noises, Kf, n_mc_smps, M)

        pad_len = sequence_len - X_len
        cur_GP_draw = torch.zeros((n_mc_smps, pad_len, GP_draws.shape[2]), dtype=torch.float32, device=globals.device)
        padded_GP_draws = torch.cat((GP_draws, cur_GP_draw), 1)

        meds = med_grid[i:i + 1]
        pad_len = sequence_len - meds.shape[1]
        meds = torch.cat([meds, torch.zeros((1, pad_len, meds.shape[2])).to(device=globals.device)], 1)
        tiled_meds = meds.repeat(n_mc_smps, 1, 1)
        padded_GPdraws_medcovs = torch.cat((padded_GP_draws, tiled_meds), 2)
        Z = torch.cat((Z, padded_GPdraws_medcovs), 0)
        i += 1

    return Z

def draw_GP(Yi, Ti, Xi, ind_kfi, ind_kti, length, noises, Kf, n_mc_smps, M):
    ny = Yi.shape[0]
    K_tt = OU_kernel(length, Ti, Ti)

    D = torch.diag(noises)
    grid_f = torch.meshgrid(ind_kfi,
                            ind_kfi)
    grid_f = (grid_f[0].T, grid_f[1].T)

    Kf_big = gather_nd(Kf, torch.stack((grid_f[0], grid_f[1]), -1))

    grid_t = torch.meshgrid(ind_kti, ind_kti)  # indexing=xy,
    grid_t = (grid_t[0].T, grid_t[1].T)
    Kt_big = gather_nd(K_tt, torch.stack((grid_t[0], grid_t[1]), -1))

    Kf_Ktt = torch.mul(Kf_big, Kt_big)

    DI_big = gather_nd(D, torch.stack((grid_f[0], grid_f[1]), -1))
    DI = torch.diag(torch.diagonal(DI_big, dim1=-2, dim2=-1))

    Ky = Kf_Ktt + DI + 1e-6 * torch.eye(ny, device=globals.device)

    nx = Xi.shape[0]

    K_xx = OU_kernel(length, Xi, Xi)
    K_xt = OU_kernel(length, Xi, Ti)

    ind = torch.cat([torch.tensor([i], device=globals.device).repeat([nx]) for i in range(M)], 0)
    grid = torch.meshgrid(ind, ind)
    grid = (grid[0].T, grid[1].T)
    Kf_big = gather_nd(Kf, torch.stack((grid[0], grid[1]), -1))
    ind2 = torch.arange(0, nx, device=globals.device).repeat([M])
    grid2 = torch.meshgrid(ind2, ind2)
    grid2 = (grid2[0].T, grid2[1].T)
    Kxx_big = gather_nd(K_xx, torch.stack((grid2[0], grid2[1]), -1))

    K_ff = torch.mul(Kf_big, Kxx_big)

    full_f = torch.cat([torch.tensor([i], device=globals.device).repeat([nx]) for i in range(M)], 0)
    grid_1 = torch.meshgrid(full_f, ind_kfi)
    Kf_big = gather_nd(Kf, torch.stack((grid_1[0], grid_1[1]), -1))
    full_x = torch.arange(0, nx, device=globals.device).repeat([M]).type(torch.long)

    grid_2 = torch.meshgrid(full_x, ind_kti)
    Kxt_big = gather_nd(K_xt, torch.stack((grid_2[0], grid_2[1]), -1))

    K_fy = torch.mul(Kf_big, Kxt_big)


    y_ = torch.reshape(Yi, (-1, 1))

    Mu = torch.matmul(K_fy, CG(Ky, y_))
    Ly = torch.cholesky(Ky)

    xi = torch.normal(mean=0, std=1.0, size=(nx * M, n_mc_smps), device=globals.device)
    Sigma = K_ff - torch.mm(K_fy, torch.cholesky_solve(torch.transpose(K_fy, 0, 1), Ly)) + 1e-6 * torch.eye(
        K_ff.shape[0], device=globals.device)

    draw = Mu + torch.mm(torch.cholesky(Sigma), xi)
    draw_reshape = (torch.reshape(torch.transpose(draw, 0, 1), (n_mc_smps, M, nx))).permute(0, 2, 1)
    return draw_reshape
