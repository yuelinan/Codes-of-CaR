import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset
import torch.nn as nn
from conv import GNN_node, GNN_node_Virtualnode
import numpy as np
import random
import copy
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
nn_act = torch.nn.ReLU()
F_act = F.relu


class CaR(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', drop_ratio = 0.5, gamma = 0.4, use_linear_predictor=False):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(CaR, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gamma  = gamma

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split('-')[0]
        emb_dim_rat = emb_dim
        if 'virtual' in gnn_type: 
            rationale_gnn_node = GNN_node_Virtualnode(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
            self.graph_encoder = GNN_node_Virtualnode(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
        else:
            rationale_gnn_node = GNN_node(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
            self.graph_encoder = GNN_node(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
        self.separator = separator_gum(
            rationale_gnn_node=rationale_gnn_node, 
            gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim_rat, 2*emb_dim_rat), torch.nn.BatchNorm1d(2*emb_dim_rat), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim_rat, 2)),
            nn=None
            )
        rep_dim = emb_dim
        if use_linear_predictor:
            self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        else:
            self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))


    def shuffle_batch(self, xc):
        num = xc.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        x = xc[random_idx]
        return x
    def forward(self, batched_data):
        # print(batched_data)
        # print(batched_data.size())
        h_node,h_list = self.graph_encoder(batched_data)

        ##  rationale
        h_r, h_env, r_node_num, env_node_num, gate = self.separator(batched_data, h_node)
        multi_layer_env_shuffle_list = []
        multi_layer_env = self.get_multi_layer_env(batched_data,h_list,gate)
        for h_layer in multi_layer_env:
            shuffle_h_layer = self.shuffle_batch(h_layer)
            multi_layer_env_shuffle_list.append(shuffle_h_layer)
        ## pred
        pred_rem = self.predictor(h_r)

        shuffle_env = self.shuffle_batch(h_env)
        combine_rationale = h_r + shuffle_env
        pred_rep = self.predictor(combine_rationale)

        ##
        pred_multi_layer_list = []
        for shuffle in  multi_layer_env_shuffle_list:
            combine_rationale = h_r + shuffle
            pred_rep_shuffle = self.predictor(combine_rationale)
            pred_multi_layer_list.append(pred_rep_shuffle)
        pred_multi_layer_list.append(pred_rep)

        loss_contrastive = self.get_contrastive_loss( h_r, h_r+h_env, h_env )

        loss_reg =  torch.abs(r_node_num / (r_node_num + env_node_num) - self.gamma  * torch.ones_like(r_node_num)).mean()

        output = {'pred_rem': pred_rem, 'loss_reg':loss_reg, 'pred_rep':pred_multi_layer_list, 'loss_contrastive':loss_contrastive}
        return output

    def get_multi_layer_env(self,batched_data,h_list,gate):
        c_out_list = []
        batch = batched_data.batch
        for i in range(1,len(h_list)-1):
            h_node = h_list[i]
            c_out = global_mean_pool((1 - gate) * h_node, batch)
            c_out_list.append(c_out)
        return c_out_list


    def get_contrastive_loss(self,x,x_aug,x_cp,T=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        x_cp_abs = x_cp.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs) + 1e-8)
        sim_matrix = torch.exp(sim_matrix / T)

        sim_matrix_cp = torch.einsum('ik,jk->ij', x, x_cp) / (torch.einsum('i,j->ij', x_abs, x_cp_abs)+1e-8)
        sim_matrix_cp = torch.exp(sim_matrix_cp / T)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        
        loss2 = pos_sim / (sim_matrix_cp.sum(dim=1) + pos_sim)
        loss = - torch.log(loss2).mean()
        return loss




    def eval_forward(self, batched_data):
        h_node,_ = self.graph_encoder(batched_data)

        batch = batched_data.batch
        size = batch[-1].item() + 1 
        
        h_r, h_env, _, _,_ = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r)
        # pred_gnn = self.predictor(h_out)
        return pred_rem # , pred_gnn


class separator_gum(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(separator_gum, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, batched_data, h_node, size=None):
        # print(batched_data.batch)
        # print(h_node.size())
        x,_ = self.rationale_gnn_node(batched_data)
        
        batch = batched_data.batch
        # print(batch.size())
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        # print(size)
        gate = self.gate_nn(x)

        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = F.gumbel_softmax(gate,hard=False,dim=-1)

        gate = gate[:,-1].unsqueeze(-1)


        h_out = global_mean_pool(gate * h_node, batch)

        c_out = global_mean_pool((1 - gate) * h_node, batch)
        # c_out = scatter_add((1 - gate) * h_node, batch, dim=0, dim_size=size)

        r_node_num = scatter_add(gate, batch, dim=0, dim_size=size)
        env_node_num = scatter_add((1 - gate), batch, dim=0, dim_size=size)
        
        return h_out, c_out, r_node_num + 1e-8 , env_node_num + 1e-8 ,gate

    def eval_forward(self, batched_data, h_node, size=None):
        x,_ = self.rationale_gnn_node(batched_data)
        
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        
        gate = self.gate_nn(x).view(-1, 1)
        
        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)

        return gate





