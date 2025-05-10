import torch
from torch import nn
from model.model_blocks import *

class GCMP(nn.Module):
    def __init__(self, n_in, n_out):
        super(GCMP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.mlp_msg_Gcc = MLP(n_in * 3, n_out * 3, n_out * 4 + 1)
        self.mlp_reduce_Gcc = MLP(n_out * 4 + n_in, n_out)

        self.mlp_ve_down = MLP(n_out * 4, n_out)
    
    def msg_cc(self, edges):
        x = torch.cat([edges.src['in_nf'], edges.dst['in_nf'], edges.data['in_ef']], dim=1)
        x = self.mlp_msg_Gcc(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_out, self.n_out, self.n_out, self.n_out], dim=1)
        k = torch.sigmoid(k)
        return {'efcco1': f1 * k, 'efcco2': f2 * k, 'efcco3': f3 * k, 'efcco4': f4 * k}

    def reduce_cc(self, nodes):
        x = torch.cat([nodes.data['in_nf'], nodes.data['nfcco1'], nodes.data['nfcco2'], nodes.data['nfcco3'], nodes.data['nfcco4']], dim=1)
        x = self.mlp_reduce_Gcc(x)
        return {'new_ccnx': x}

    def forward(self, g, in_vc, in_ve):
        with g.local_scope():
            g.nodes['gc'].data['in_nf'] = in_vc
            g.edges['e_cc'].data['in_ef'] = in_ve
            g.apply_edges(self.msg_cc, etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco1', 'efcco1'), fn.sum('efcco1', 'nfcco1'), etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco2', 'efcco2'), fn.max('efcco2', 'nfcco2'), etype=('gc',    'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco3', 'efcco3'), fn.min('efcco3', 'nfcco3'), etype=('gc',    'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco4', 'efcco4'), fn.mean('efcco4', 'nfcco4'), etype=('gc',   'e_cc', 'gc'))
            g.apply_nodes(self.reduce_cc, ntype='gc')
    
            out_vc = g.nodes['gc'].data['new_ccnx']
            out_ve = self.mlp_ve_down(torch.cat([g.edges['e_cc'].data['efcco1'], g.edges['e_cc'].data['efcco2'], g.edges['e_cc'].data['efcco3'], g.edges['e_cc'].data['efcco4']], dim=1))

            return out_vc, out_ve

class ToEdge(nn.Module):
    def __init__(self, n_in_fc, n_in_fe1, n_in_fe2, n_out_fe):
        super(ToEdge, self).__init__()
        self.n_in_fc = n_in_fc
        self.n_in_fe1 = n_in_fe1
        self.n_in_fe2 = n_in_fe2
        self.n_out_fe = n_out_fe
        self.mlp_msg_Gcc1 = MLP(n_in_fc * 2 + n_in_fe1 + n_in_fe2, n_out_fe * 3, n_out_fe * 4 + 1)
        self.mlp_msg_Gcc2 = nn.Linear(n_out_fe * 4, n_out_fe)

    def msg_cc(self, edges):
        x = torch.cat([edges.src['in_nf'], edges.dst['in_nf'], edges.data['in_ef']], dim=1)
        x = self.mlp_msg_Gcc1(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_out_fe, self.n_out_fe, self.n_out_fe, self.n_out_fe], dim=1)
        k = torch.sigmoid(k)
        out_ve = self.mlp_msg_Gcc2(torch.cat([f1 * k, f2 * k, f3 * k, f4 * k], dim=1))
        return {'out_ve': out_ve}

    def forward(self, g, in_vc, in_ve1, in_ve2):
        with g.local_scope():
            g.nodes['gc'].data['in_nf'] = in_vc
            g.edges['e_cc'].data['in_ef'] = torch.cat([in_ve1, in_ve2], dim=1)
            g.apply_edges(self.msg_cc, etype=('gc', 'e_cc', 'gc'))
            out_ve = g.edges['e_cc'].data['out_ve']
        return out_ve

class STGSG_V3(nn.Module):
    def __init__(self, n_gc_nf, n_gc_ef):
        super(STGSG_V3, self).__init__()
        self.n_gc_nf = n_gc_nf
        self.n_gc_ef = n_gc_ef

        self.fc_gc = nn.Linear(n_gc_nf, 32)
        self.fc_ef = nn.Linear(n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        # self.gcmp4 = GCMP(64, 32)
        self.toedge = ToEdge(64, 64, 1, 1)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            # out_vc4, out_ve4 = self.gcmp4(g, out_vc3, out_ve3)
            out_ve = self.toedge(g, out_vc3, out_ve3, g.edges['e_cc'].data['ef'])


            return out_ve








