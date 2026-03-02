import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return torch.clamp(self.act(out), min=-1000.0, max=1000.0)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, args):
        super(Model, self).__init__()
        if args.dataset == 'questions':
            activation = nn.ReLU()
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
        self.read_mode = readout
        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.fc6 = nn.Linear(n_h, n_h, bias=False)
        self.fc5 = nn.Linear(n_h, n_in, bias=False)
        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)
        self.to(self.device)

    def forward(self, seq1, adj, normal_for_generation_idx, normal_for_train_idx, train_flag, args, sparse=False):
        h_1 = self.gcn1(seq1, adj, sparse)
        # emb = h_1
        emb = self.gcn2(h_1, adj, sparse)


        outlier_emb = None
        emb_combine = None
        normal_for_generation_emb = emb[:, normal_for_generation_idx, :]

        noise = torch.randn(normal_for_generation_emb.size(), device=self.device) * args.var + args.mean
        noised_normal_for_generation_emb = normal_for_generation_emb + noise
        # Add noise into the attribute of sampled abnormal nodes
        # degree = torch.sum(raw_adj[0, :, :], 0)[sample_abnormal_idx]
        # neigh_adj = raw_adj[0, sample_abnormal_idx, :] / torch.unsqueeze(degree, 1)

        neigh_adj = adj[0, normal_for_generation_idx, :]
        # emb[0, sample_abnormal_idx, :] =self.act(torch.mm(neigh_adj, emb[0, :, :]))
        # emb[0, sample_abnormal_idx, :] = self.fc4(emb[0, sample_abnormal_idx, :])

        outlier_emb = torch.mm(neigh_adj, emb[0, :, :])
        outlier_emb = self.act(self.fc4(outlier_emb))
        # emb_con = self.act(self.fc6(emb_con))

        emb_combine = torch.cat((emb[:, normal_for_train_idx, :], torch.unsqueeze(outlier_emb, 0)), 1)

        # TODO ablation study add noise on the selected nodes

        # std = 0.01
        # mean = 0.02
        # noise = torch.randn(emb[:, sample_abnormal_idx, :].size()) * std + mean
        # emb_combine = torch.cat((emb[:, normal_idx, :], emb[:, sample_abnormal_idx, :] + noise), 1)

        # TODO ablation study generate outlier from random noise
        # std = 0.01
        # mean = 0.02
        # emb_con = torch.mm(neigh_adj, emb[0, :, :])
        # noise = torch.randn(emb_con.size()) * std + mean
        # emb_con = self.act(self.fc4(noise))
        # emb_combine = torch.cat((emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)

        if train_flag:
            f_1 = self.fc1(emb_combine)
        else:
            f_1 = self.fc1(emb)
        
        f_1 = self.act(f_1)
        f_2 = self.fc2(f_1)
        f_2 = self.act(f_2)
        logits = self.fc3(f_2)
        emb[:, normal_for_generation_idx, :] = outlier_emb

        agg_attention_weights = 0
        con_loss = torch.tensor(0.0, device=emb.device)
        gna_loss = torch.tensor(0.0, device=emb.device)
        reconstruction_loss = torch.tensor(0.0, device=emb.device)

        return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, agg_attention_weights, con_loss, gna_loss, reconstruction_loss
