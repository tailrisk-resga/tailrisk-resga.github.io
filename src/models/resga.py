"""ReSGA model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReSGA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.T = args.sequence_length
        self.K = args.K
        self.feature_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        
        self.dropout = nn.Dropout(args.dropout)

        self.rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.dropout
        )

    
        
        # SGA-like layers
        self.hidden_concept_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_hs = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(args.dropout)
        )
        self.fc_hs_fore = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(args.dropout)
        )
        self.fc_hs_back = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_indi = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(args.dropout)
        )
        self.fc_ret = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(args.dropout)
        )
        self.fc_ret_back = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_ret_fore = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(args.dropout)
        )
        
        # Output head for fusion
        self.fc_out = nn.Linear(self.hidden_dim, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def cal_cos_similarity(self, x, y):
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x*x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y*y, dim=1)).reshape(-1, 1)
        cos_similarity = xy / x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity

    def forward(self, x):
        x = x.squeeze(0)  # [N, 5*T, D]
        N, total_T, D = x.shape
        T = self.T
        H = self.hidden_dim

    

        rnn_output, _ = self.rnn(x)
        h_shared_info = rnn_output[:, -1, :]

        similarity_mat = self.cal_cos_similarity(h_shared_info, h_shared_info)  # [N, N]
        diag = similarity_mat.diagonal(0).clone()
        similarity_mat.fill_diagonal_(0)

        mask_row = torch.arange(N, device=x.device).unsqueeze(1).repeat(1, self.K).reshape(1, -1)  # [1, N*K]
        mask_column = torch.topk(torch.abs(similarity_mat), self.K, dim=1)[1].reshape(1, -1)       # [1, N*K]
        mask = torch.zeros(N, N, device=x.device)
        mask[mask_row, mask_column] = 1
        topK_similarity_mat = similarity_mat * mask
        topK_similarity_mat = topK_similarity_mat + torch.diag_embed((topK_similarity_mat.sum(0)!=0).float() * diag)

        # ---- Concept aggregation (dense) ----
        concept_feature = torch.t(h_shared_info).mm(topK_similarity_mat).t()  # [*, H]
        concept_feature = concept_feature[concept_feature.sum(1) != 0]
        concept_feature = F.leaky_relu(self.hidden_concept_fc(concept_feature), negative_slope=0.2)
        concept_feature = self.dropout(concept_feature)

        concept_similarity_mat = self.cal_cos_similarity(h_shared_info, concept_feature)
        concept_attention_mat = F.softmax(concept_similarity_mat, dim=1)

        h_shared_info = concept_attention_mat.mm(concept_feature)
        h_shared_info = self.fc_hs(h_shared_info)
        h_shared_back = self.fc_hs_back(h_shared_info)
        output_hs = self.fc_hs_fore(h_shared_info)

        # ---- End SGA block ----

        neighbors_others = torch.topk(torch.abs(similarity_mat), k=self.K, dim=1).indices
        self_idx = torch.arange(N, device=x.device).unsqueeze(1)                             # [N, 1]
        neighbors_idx = torch.cat([self_idx, neighbors_others], dim=1)  

        num_segments = total_T // T  -1
        keys_all_list = []    
        values_all_list = []  
        for s in range(num_segments):
            keys_all_list.append(rnn_output[:, (s+1)*T - 1, :])  
            values_all_list.append(rnn_output[:, (s+1)*T, :])  
        
        q = rnn_output[:, -1, :]
    
        keys_all = torch.stack(keys_all_list, dim=1)
        values_all = torch.stack(values_all_list, dim=1)

        keys   = keys_all[neighbors_idx, :, :]    # [N, K+1, S, H]
        values = values_all[neighbors_idx, :, :]  # [N, K+1, S, H]

        eps = 1e-8
        q_norm = q / (q.norm(dim=1, keepdim=True) + eps)
        keys_norm = keys / (keys.norm(dim=3, keepdim=True) + eps)

        sim = torch.einsum('nh,nkjh->nkj', q_norm, keys_norm)

        w_time = F.softmax(sim, dim=2)                     # [N, K+1, L-1]
        retrieved_per_nb = torch.einsum('nks,nksh->nkh', w_time, values)  # [N, K+1, H]

        nb_score = sim.mean(dim=2)                                 # [N, K+1]
        w_nb = F.softmax(nb_score, dim=1)                  # [N, K+1]

        retrieved_info = torch.einsum('nk,nkh->nh', w_nb, retrieved_per_nb)    # [N, H]

        retrieved_info = self.fc_ret(retrieved_info)
        retrieved_back = self.fc_ret_back(retrieved_info)
        output_retrieved = self.fc_ret_fore(retrieved_info)

        individual_info = q - h_shared_back - retrieved_back
        output_indi = self.fc_indi(individual_info)

        all_info = output_hs + output_indi + output_retrieved  # [N, H]


        out = self.fc_out(all_info)  # [N, 2]
        v = -F.softplus(out[:, 0])
        e = v - F.softplus(out[:, 1])
        return v, e



__all__ = ["ReSGA"]
