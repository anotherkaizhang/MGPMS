import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
import time
import globals
import matplotlib.pyplot as plt
    
from util import pad_rawdata,SE_kernel,OU_kernel,gather_nd,log_sum_exp,CG,Lanczos,block_CG,block_Lanczos
from model import TransformerModel

def get_probs_and_accuracy(preds, O):
    all_probs = torch.exp(
        preds[:, 1] - log_sum_exp(preds, dim=1))
    N = preds.shape[0] / n_mc_smps
    probs = torch.zeros([0], device=globals.device)
    i = 0
    while i < N:
        probs = torch.cat(
            [probs, torch.tensor([torch.mean(all_probs[i * n_mc_smps: i * n_mc_smps + n_mc_smps])], device=globals.device)], 0)
        i += 1

    correct_pred = torch.eq(torch.gt(probs, 0.5).type(torch.uint8), O)
    accuracy = torch.mean((correct_pred.type(torch.float32)))
    return probs, accuracy

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()

    perm = rs.permutation(Ntr)

    batch = 0
    for s, e in zip(starts_tr, ends_tr):
        batch_start = time.time()
        inds = perm[s:e]

        T_pad, Y_pad, ind_kf_pad, \
        ind_kt_pad, X_pad, meds_pad \
            = pad_rawdata(T=[times_tr[i] for i in inds],
                          Y=[values_tr[i] for i in inds],
                          ind_kf=[ind_lvs_tr[i] for i in inds],
                          ind_kt=[ind_times_tr[i] for i in inds],
                          X=[rnn_grid_times_tr[i] for i in inds],
                          meds_on_grid=[meds_on_grid_tr[i] for i in inds])

        T_pad = torch.tensor(T_pad, dtype=torch.float32)
        Y_pad = torch.tensor(Y_pad, dtype=torch.float32)
        X_pad = torch.tensor(X_pad, dtype=torch.float32)
        ind_kf_pad = torch.tensor(ind_kf_pad, dtype=torch.int32)
        ind_kt_pad = torch.tensor(ind_kt_pad, dtype=torch.int32)
        meds_pad = torch.tensor(meds_pad, dtype=torch.float32)
        covs = torch.tensor([covs_tr[i] for i in inds], dtype=torch.float32)
        num_obs_times = torch.tensor([num_obs_times_tr[i] for i in inds], dtype=torch.int32)
        num_obs_values = torch.tensor([num_obs_values_tr[i] for i in inds], dtype=torch.int32)
        num_rnn_grid_times = torch.tensor([num_rnn_grid_times_tr[i] for i in inds], dtype=torch.int32)
        O = torch.tensor([labels_tr[i] for i in inds], dtype=torch.float32)
        O_dup = torch.reshape(O.unsqueeze(1).repeat(1, sequence_len), (-1,))

        T_pad = T_pad.to(globals.device)
        Y_pad = Y_pad.to(globals.device)
        X_pad = X_pad.to(globals.device)
        ind_kf_pad = ind_kf_pad.to(globals.device)
        ind_kt_pad = ind_kt_pad.to(globals.device)
        meds_pad = meds_pad.to(globals.device)
        covs = covs.to(globals.device)
        num_obs_times = num_obs_times.to(globals.device)
        num_obs_values = num_obs_values.to(globals.device)
        num_rnn_grid_times = num_rnn_grid_times.to(globals.device)
        O = O.to(globals.device)
        O_dup = torch.reshape(O.unsqueeze(1).repeat(1, sequence_len), (-1,))

        optimizer.zero_grad()
        output = model(Y_pad, T_pad, X_pad, ind_kf_pad, ind_kt_pad, num_obs_times, num_obs_values,
                       num_rnn_grid_times, meds_pad, covs)  # [batch_size, sequence_len]
        loss = criterion(output.view(-1), O_dup)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 1
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                epoch, batch, len(starts_tr), scheduler.get_last_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss))
            total_loss = 0
            start_time = time.time()

        batch += 1


def evaluate(eval_model):
    eval_model.eval()
    total_loss = 0.

    perm = rs.permutation(Nte)

    output_all = []
    target_all = []

    with torch.no_grad():
        for s, e in tqdm(zip(starts_te, ends_te)):
            batch_start = time.time()
            inds = perm[s:e]

            T_pad, Y_pad, ind_kf_pad, \
            ind_kt_pad, X_pad, meds_pad \
                = pad_rawdata(T=[times_te[i] for i in inds],
                              Y=[values_te[i] for i in inds],
                              ind_kf=[ind_lvs_te[i] for i in inds],
                              ind_kt=[ind_times_te[i] for i in inds],
                              X=[rnn_grid_times_te[i] for i in inds],
                              meds_on_grid=[meds_on_grid_te[i] for i in inds])

            T_pad = torch.tensor(T_pad, dtype=torch.float32)
            Y_pad = torch.tensor(Y_pad, dtype=torch.float32)
            X_pad = torch.tensor(X_pad, dtype=torch.float32)
            ind_kf_pad = torch.tensor(ind_kf_pad, dtype=torch.int32)
            ind_kt_pad = torch.tensor(ind_kt_pad, dtype=torch.int32)
            meds_pad = torch.tensor(meds_pad, dtype=torch.float32)
            covs = torch.tensor([covs_te[i] for i in inds], dtype=torch.float32)
            num_obs_times = torch.tensor([num_obs_times_te[i] for i in inds], dtype=torch.int32)
            num_obs_values = torch.tensor([num_obs_values_te[i] for i in inds], dtype=torch.int32)
            num_rnn_grid_times = torch.tensor([num_rnn_grid_times_te[i] for i in inds], dtype=torch.int32)
            O = torch.tensor([labels_te[i] for i in inds], dtype=torch.float32)
            O_dup = torch.reshape(O.unsqueeze(1).repeat(1, sequence_len), (-1,))

            T_pad = T_pad.to(globals.device)
            Y_pad = Y_pad.to(globals.device)
            X_pad = X_pad.to(globals.device)
            ind_kf_pad = ind_kf_pad.to(globals.device)
            ind_kt_pad = ind_kt_pad.to(globals.device)
            meds_pad = meds_pad.to(globals.device)
            covs = covs.to(globals.device)
            num_obs_times = num_obs_times.to(globals.device)
            num_obs_values = num_obs_values.to(globals.device)
            num_rnn_grid_times = num_rnn_grid_times.to(globals.device)
            O = O.to(globals.device)
            O_dup = torch.reshape(O.unsqueeze(1).repeat(1, sequence_len), (-1,))

            output = model(Y_pad, T_pad, X_pad, ind_kf_pad, ind_kt_pad, num_obs_times, num_obs_values,
                           num_rnn_grid_times, meds_pad, covs)

            output_all.extend(list(output.cpu().numpy()))
            target_all.extend(list(O.cpu().numpy()))
            total_loss += criterion(output.view(-1), O_dup).item()

    return total_loss, output_all, target_all

def plot_trend(output_all, target_all, sequence_len):
    negatives = [ind for ind in range(len(target_all)) if target_all[ind] == 0]
    positives = [ind for ind in range(len(target_all)) if target_all[ind] == 1]

    n_row = 20

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_row):
        ax.plot(output_all[negatives[i]][:sequence_len], color='olive', marker="o", linestyle="dotted")
        if i == n_row - 1:
            ax.plot(output_all[negatives[i]][:sequence_len], color='olive', marker="o", linestyle="dotted",
                    label="Patients Not Need MV")

    for i in range(n_row):
        ax.plot(output_all[positives[i]][:sequence_len], color='teal', marker="o", linestyle="dotted")
        if i == n_row - 1:
            ax.plot(output_all[negatives[i]][:sequence_len], color='teal', marker="o", linestyle="dotted", label="Patients Need MV")

    # ax.set_xticks(np.arange(0, 19, 4), tuple([str(4*i)+'h' for i in range(0, 19, 4)]))
    ax.set_xticks(np.arange(0, sequence_len))
    ax.set_xticklabels([str(4 * i) + 'h' for i in range(0, sequence_len)])

    ax.set_xlabel("Hours after admission")
    ax.set_ylabel("Risk Score")
    ax.set_title("Patients' Risk Score Trend of Intubation (3 days after admission)")
    ax.legend()
    plt.show()
    # fig.savefig('Robustness'+'.tiff', dpi=1200, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

def plot_auc_aupr(output_all, target_all, sequence_len):
    from sklearn import metrics
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc

    ### AUC
    x = range(sequence_len)
    aucs = []
    for i in range(sequence_len):
        fpr, tpr, thresholds = metrics.roc_curve(np.array(target_all).astype(int), [output_all[j][i] for j in range(
            len(target_all))])
        roc_auc = metrics.auc(fpr, tpr)
        #     print("Area under the ROC curve : %f" % roc_auc)
        aucs.append(roc_auc)

    print(np.max(aucs))

    fig, ax = plt.subplots()
    ax.plot(x, aucs)
    ax.set_xticks(np.arange(0, sequence_len))
    ax.set_xticklabels([str(4 * i) + 'h' for i in range(0, sequence_len)])
    ax.set_xlabel("Hours after admission")
    ax.set_ylabel("AUC")
    ax.set_title("AUC Intubation Precition - 7 Days After Admission. Maximum")


    ### AUPR
    prcs = []
    for i in range(sequence_len):
        lr_precision, lr_recall, _ = precision_recall_curve(np.array(target_all).astype(int), [output_all[j][i] for j in
                                                                                               range(len(
                                                                                                   target_all))])  # , np.array(target_all).astype(int))
        prcs.append(auc(lr_recall, lr_precision))
    print(np.max(prcs))
    fig, ax = plt.subplots()
    ax.plot(x, prcs)
    ax.set_xticks(np.arange(0, sequence_len, 4))
    ax.set_xticklabels([str(4 * i) + 'h' for i in range(0, sequence_len, 4)])
    ax.set_xlabel("Hours after admission")
    ax.set_ylabel("AUPR")
    ax.set_title("AUPR Intubation Precition - 7 Days After Admission")


if __name__ == '__main__':
    ### Device
    print("GPU available: ", torch.cuda.is_available())
    print("GPU count: ", torch.cuda.device_count())

    if torch.cuda.is_available():
        device = torch.device(10)  # custermize your own GPU device
    else:
        device = torch.device('cpu')


    globals.device=device
    torch.set_default_dtype(torch.float32)

    ### data
    f = open("data/input_for_GPRNN-simulation_M25_cov22_med10.pickle", 'rb')

    input_for_GPRNN = pickle.load(f, encoding="latin1")
    f.close()

    num_obs_times = input_for_GPRNN['num_obs_times']
    num_obs_values = input_for_GPRNN['num_obs_values']
    num_rnn_grid_times = input_for_GPRNN['num_rnn_grid_times']
    rnn_grid_times = input_for_GPRNN['rnn_grid_times']
    labels = input_for_GPRNN['labels']
    times = input_for_GPRNN['times']
    values = input_for_GPRNN['values']
    ind_lvs = input_for_GPRNN['ind_lvs']
    ind_times = input_for_GPRNN['ind_times']
    meds_on_grid = input_for_GPRNN['meds_on_grid']
    covs = input_for_GPRNN['covs']

    print("That's all!")

    N_tot = len(labels)

    seed = 8675309
    rs = np.random.RandomState(seed)

    train_test_perm = rs.permutation(N_tot)
    val_frac = 0.1  # fraction of full data to set aside for testing

    te_ind = train_test_perm[: int(val_frac * N_tot)]
    tr_ind = train_test_perm[int(val_frac * N_tot):]

    Nte = len(te_ind)
    Ntr = len(tr_ind)

    batch_size = 50
    eval_batch_size = 10

    starts_tr = np.arange(0, Ntr, batch_size)
    ends_tr = np.arange(batch_size, Ntr + 1, batch_size)

    if len(starts_tr) > len(ends_tr):
        starts_tr = starts_tr[:-1]

    starts_te = np.arange(0, Nte, eval_batch_size)
    ends_te = np.arange(eval_batch_size, Nte + 1, eval_batch_size)

    if len(starts_te) > len(ends_te):
        starts_te = starts_te[:-1]

    # Break everything out into train/test
    for varname in ['covs', 'labels', 'times', 'values', 'ind_lvs', 'ind_times', 'meds_on_grid', \
                    'num_obs_times', 'num_obs_values', 'rnn_grid_times', 'num_rnn_grid_times']:
        print(varname + '_tr = [' + varname + '[i] for i in tr_ind]')
        exec(varname + '_tr = [' + varname + '[i] for i in tr_ind]')

    for varname in ['covs', 'labels', 'times', 'values', 'ind_lvs', 'ind_times', 'meds_on_grid', \
                    'num_obs_times', 'num_obs_values', 'rnn_grid_times', 'num_rnn_grid_times']:
        print(varname + '_te = [' + varname + '[i] for i in te_ind]')
        exec(varname + '_te = [' + varname + '[i] for i in te_ind]')


    ### Parameter setting
    M = 25
    n_covs = 22
    n_meds = 10

    ninput = M + n_meds
    emsize = 512  # embedding dimension
    nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8  # the number of heads in the multiheadattention models
    dropout = 0.3

    sequence_len = 42  # 1 week, 4-hr average
    n_mc_smps = 20

    model = TransformerModel(M=M,
                             n_meds=n_meds,
                             n_covs=n_covs,
                             sequence_len=sequence_len,
                             emsize=emsize,
                             nhead=nhead,
                             nhid=nhid,
                             nlayers=nlayers,
                             n_mc_smps=n_mc_smps,
                             dropout=dropout).to(globals.device)

    print("data fully setup!")

    ### Training parameters
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    lr = 0.03
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


    ### Training
    best_val_loss = float("inf")
    epochs = 100
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss, _, _ = evaluate(model)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
              .format(epoch, (time.time() - epoch_start_time),
                      val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    ### Validation
    val_loss, output_all, target_all = evaluate(best_model)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
          .format(epoch, (time.time() - epoch_start_time),
                  val_loss))
    
    ### Plot trends
    # plot_trend(output_all, target_all, sequence_len)
    # plot_auc_aupr(output_all, target_all, sequence_len)