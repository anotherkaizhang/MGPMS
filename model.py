import torch
import torch.nn as nn
import math
import globals
from util import get_GP_samples

class TransformerModel(nn.Module):
    def __init__(self, M, n_meds, n_covs, sequence_len, emsize, nhead, nhid, nlayers, n_mc_smps,
                 dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.encoder = nn.Linear(M+n_meds, emsize)
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.emsize = emsize
        self.final = torch.rand(size=(sequence_len * (emsize + n_covs),))

        # GP parameters
        self.M = M
        self.n_meds=n_meds
        self.n_mc_smps = n_mc_smps
        self.sequence_len = sequence_len
        self.n_covs = n_covs
        self.emsize = emsize

        self.log_length = torch.normal(size=[1], mean=1, std=0.1)
        self.log_noises = torch.normal(size=[self.M], mean=-2, std=0.1)
        self.L_f_init = torch.eye(self.M)

        self.log_length = torch.nn.Parameter(self.log_length)
        self.log_noises = torch.nn.Parameter(self.log_noises)
        self.L_f_init = torch.nn.Parameter(self.L_f_init)
        self.final = torch.nn.Parameter(self.final)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, Y, T, X, ind_kf, ind_kt, num_obs_times, num_obs_values,
                num_rnn_grid_times, med_grid, covs):  # [batch_size, sequence_len, M]

        length = torch.exp(self.log_length)
        noises = torch.exp(self.log_noises)
        Lf = torch.tril(self.L_f_init)
        Kf = torch.mm(Lf, torch.transpose(Lf, 0, 1))
        final_reshape = torch.zeros(size=(self.sequence_len * (self.emsize + self.n_covs), self.sequence_len)).to(globals.device)
        for i in range(self.sequence_len):
            final_reshape[:2 * (i + 1), i] = self.final[:2 * (i + 1)]

        Z = get_GP_samples(Y=Y,
                           T=T,
                           X=X,
                           ind_kf=ind_kf,
                           ind_kt=ind_kt,
                           num_obs_times=num_obs_times,
                           num_obs_values=num_obs_values,
                           num_rnn_grid_times=num_rnn_grid_times,
                           med_grid=med_grid,
                           length=length,
                           noises=noises,
                           Kf=Kf,
                           n_mc_smps=self.n_mc_smps,
                           M=self.M,
                           n_meds=self.n_meds,
                           sequence_len=self.sequence_len)  # M = M + n_meds  # [batch_size * n_mc_smps, sequence_len, M]


        batch_size_MonteCarlo, _, _ = Z.shape
        batch_size = batch_size_MonteCarlo // self.n_mc_smps

        # Enbedding
        src = self.encoder(Z)  # [batch_size_MC, sequence_len, embed_size]
        src = src.permute(1, 0, 2)  # [sequence_len, batch_size_MC, embed_size]

        # Position-Encoder
        src = self.pos_encoder(src)  # [sequence_len, batch_size_MC, embed_size]

        # Transformer
        output = self.transformer_encoder(src, self.src_mask)  # [sequence_len, batch_size_MC, embed_size]

        # Append covs
        output = output.permute(1, 0, 2)  # [batch_size_MC, sequence_len, embed_size]
        covs = covs.repeat((1, self.n_mc_smps)).repeat((1, self.sequence_len)).view(
            (batch_size * self.n_mc_smps, self.sequence_len, self.n_covs))  # [batch_size_MC, sequence_len, ncovs]

        output = torch.cat((output, covs), axis=2)  # [batch_size_MC, sequence_len, embed_size+ncovs]

        output = torch.reshape(output,
                               (batch_size * self.n_mc_smps, -1))  # [batch_size_MC, sequence_len *(embed_size+ncovs)]

        output = torch.reshape(output, (batch_size * self.n_mc_smps, self.sequence_len * (
                    self.emsize + self.n_covs)))  # [batch_size, sequence_len*(embed_size+ncovs)] row-by-row
        output = torch.mm(output, final_reshape)  # [batch_size, sequenceLen]
        output = torch.mean(output.reshape(-1, self.n_mc_smps, self.sequence_len),
                            dim=1)  # [real_batch_size, n_mc_smps, sequenceLen]


        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)