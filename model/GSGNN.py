import torch
from torch import nn
from model.model_blocks import *
import torch.nn.functional as F


class GCMP(nn.Module):
    def __init__(self, n_in, n_out):
        super(GCMP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.mlp_msg_Gcc = MLP(n_in * 3, n_out * 3, n_out * 4 + 1)
        self.mlp_reduce_Gcc = MLP(n_out * 4 + n_in, n_out)

        self.mlp_ve_down = MLP(n_out * 4 + n_in, n_out)
        
        self.bn_gc = nn.BatchNorm1d(n_out)
        self.bn_ef = nn.BatchNorm1d(n_out)
    
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

    def forward(self, g, in_vc1, in_ve1, in_vc2=None, in_ve2=None):
        with g.local_scope():
            if in_vc2 is not None:
                g.nodes['gc'].data['in_nf'] = torch.cat([in_vc1, in_vc2], dim=1)
            else:
                g.nodes['gc'].data['in_nf'] = in_vc1
            if in_ve2 is not None:
                g.edges['e_cc'].data['in_ef'] = torch.cat([in_ve1, in_ve2], dim=1)
            else:
                g.edges['e_cc'].data['in_ef'] = in_ve1
            g.apply_edges(self.msg_cc, etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco1', 'efcco1'), fn.sum('efcco1', 'nfcco1'), etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco2', 'efcco2'), fn.max('efcco2', 'nfcco2'), etype=('gc',    'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco3', 'efcco3'), fn.min('efcco3', 'nfcco3'), etype=('gc',    'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco4', 'efcco4'), fn.mean('efcco4', 'nfcco4'), etype=('gc',   'e_cc', 'gc'))
            g.apply_nodes(self.reduce_cc, ntype='gc')
    
            out_vc = g.nodes['gc'].data['new_ccnx']
            out_ve = self.mlp_ve_down(torch.cat([g.edges['e_cc'].data['efcco1'], g.edges['e_cc'].data['efcco2'], g.edges['e_cc'].data['efcco3'], g.edges['e_cc'].data['efcco4'], g.edges['e_cc'].data['in_ef']], dim=1))

            out_vc = self.bn_gc(out_vc)
            out_ve = self.bn_ef(out_ve)

            return out_vc, out_ve

class GSMP(nn.Module):
    def __init__(self, n_in, n_out):
        super(GSMP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.mlp_msg_Gss = MLP(n_in * 3, n_out * 3, n_out * 4 + 1)
        self.mlp_reduce_Gss = MLP(n_out * 4 + n_in, n_out)

        self.mlp_ve_down = MLP(n_out * 4 + n_in, n_out)

        self.bn_nf = nn.BatchNorm1d(n_out)
        self.bn_ef = nn.BatchNorm1d(n_out)
    
    def msg_ss(self, edges):
        x = torch.cat([edges.src['in_nf'], edges.dst['in_nf'], edges.data['in_ef']], dim=1)
        x = self.mlp_msg_Gss(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_out, self.n_out, self.n_out, self.n_out], dim=1)
        k = torch.sigmoid(k)
        return {'efss1': f1 * k, 'efss2': f2 * k, 'efss3': f3 * k, 'efss4': f4 * k}

    def reduce_ss(self, nodes):
        x = torch.cat([nodes.data['in_nf'], nodes.data['nfss1'], nodes.data['nfss2'], nodes.data['nfss3'], nodes.data['nfss4']], dim=1)
        x = self.mlp_reduce_Gss(x)
        return {'new_ssnx': x}

    def forward(self, g, in_vc1, in_ve1, in_vc2=None, in_ve2=None):
        with g.local_scope():
            if in_vc2 is not None:
                g.nodes['gc'].data['in_nf'] = torch.cat([in_vc1, in_vc2], dim=1)
            else:
                g.nodes['gc'].data['in_nf'] = in_vc1
            if in_ve2 is not None:
                g.edges['e_ss'].data['in_ef'] = torch.cat([in_ve1, in_ve2], dim=1)
            else:
                g.edges['e_ss'].data['in_ef'] = in_ve1
            g.apply_edges(self.msg_ss, etype=('gc', 'e_ss', 'gc'))
            g.update_all(fn.copy_e('efss1', 'efss1'), fn.sum('efss1', 'nfss1'), etype=('gc', 'e_ss', 'gc'))
            g.update_all(fn.copy_e('efss2', 'efss2'), fn.max('efss2', 'nfss2'), etype=('gc',    'e_ss', 'gc'))
            g.update_all(fn.copy_e('efss3', 'efss3'), fn.min('efss3', 'nfss3'), etype=('gc',    'e_ss', 'gc'))
            g.update_all(fn.copy_e('efss4', 'efss4'), fn.mean('efss4', 'nfss4'), etype=('gc',   'e_ss', 'gc'))
            g.apply_nodes(self.reduce_ss, ntype='gc')
    
            out_vc = g.nodes['gc'].data['new_ssnx']
            out_ve = self.mlp_ve_down(torch.cat([g.edges['e_ss'].data['efss1'], g.edges['e_ss'].data['efss2'], g.edges['e_ss'].data['efss3'], g.edges['e_ss'].data['efss4'], g.edges['e_ss'].data['in_ef']], dim=1))

            out_vc = self.bn_nf(out_vc)
            out_ve = self.bn_ef(out_ve)

            return out_vc, out_ve

class ToEdge(nn.Module):
    def __init__(self, n_in_fc, n_in_fe, n_out_fe):
        super(ToEdge, self).__init__()
        self.n_in_fc = n_in_fc
        self.n_in_fe = n_in_fe
        self.n_out_fe = n_out_fe
        self.mlp_msg_Gcc1 = MLP(n_in_fc * 2 + n_in_fe, 128, 96, 16 * 4 + 1)
        self.mlp_msg_Gcc2 = MLP(16 * 4, 16 * 2, self.n_out_fe)

    def msg_cc(self, edges):
        x = torch.cat([edges.src['in_nf'], edges.dst['in_nf'], edges.data['in_ef']], dim=1)
        x = self.mlp_msg_Gcc1(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, 16, 16, 16, 16], dim=1)
        k = torch.sigmoid(k)
        out_ve = self.mlp_msg_Gcc2(torch.cat([f1 * k, f2 * k, f3 * k, f4 * k], dim=1))
        return {'out_ve': out_ve}

    def forward(self, g, in_vc, in_ve):
        with g.local_scope():
            g.nodes['gc'].data['in_nf'] = in_vc
            g.edges['e_cc'].data['in_ef'] = in_ve
            g.apply_edges(self.msg_cc, etype=('gc', 'e_cc', 'gc'))
            out_ve = g.edges['e_cc'].data['out_ve']
        return out_ve

class GSGNN_Sub0(nn.Module):
    def __init__(self):
        super(GSGNN_Sub0, self).__init__()
        self.n_gc_nf = 6
        self.n_cc_ef = 1
        self.n_ss_ef = 2

        self.n_mp_out = 32

        self.fc_ss_nf = nn.Linear(self.n_gc_nf, 32)
        self.fc_ss_ef = nn.Linear(self.n_ss_ef, 32)
        self.gsmp1 = GSMP(32, 64)
        self.gsmp2 = GSMP(64, 128)
        self.gsmp3 = GSMP(64 + 128, 64)
        self.mlp_ss_nf = MLP(64 + 32, 64, self.n_mp_out)
        self.mlp_ss_ef = MLP(64 + 32, 64, self.n_mp_out)

        self.fc_cc_nf = nn.Linear(self.n_gc_nf + self.n_mp_out, 32)
        self.fc_cc_ef = nn.Linear(self.n_cc_ef, 32)
        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(64 + 128, 64)
        self.toedge = ToEdge(64 + 32, 64 + 32 + self.n_cc_ef, 1)

    def forward(self, g):
        with g.local_scope():
            in_gc_nf = g.nodes['gc'].data['nf']
            in_cc_ef = g.edges['e_cc'].data['ef']
            in_ss_ef = g.edges['e_ss'].data['ef']

            # modify
            in_gc_nf[:, 1:3] = 0
            in_cc_ef[:, :] = 0


            # ss
            out_ss_vc0 = self.fc_ss_nf(in_gc_nf)
            out_ss_ve0 = self.fc_ss_ef(in_ss_ef)
            out_ss_vc1, out_ss_ve1 = self.gsmp1(g, out_ss_vc0, out_ss_ve0)
            out_ss_vc2, out_ss_ve2 = self.gsmp2(g, out_ss_vc1, out_ss_ve1)
            out_ss_vc3, out_ss_ve3 = self.gsmp3(g, out_ss_vc2, out_ss_ve2, out_ss_vc1, out_ss_ve1)
            out_ss_vc4 = self.mlp_ss_nf(torch.cat([out_ss_vc3, out_ss_vc0], dim=1))
            out_ss_ve4 = self.mlp_ss_ef(torch.cat([out_ss_ve3, out_ss_ve0], dim=1))

            # cc
            out_cc_vc0 = self.fc_cc_nf(torch.cat([in_gc_nf, out_ss_vc4], dim=1))
            out_cc_ve0 = self.fc_cc_ef(in_cc_ef)
            out_cc_vc1, out_cc_ve1 = self.gcmp1(g, out_cc_vc0, out_cc_ve0)
            out_cc_vc2, out_cc_ve2 = self.gcmp2(g, out_cc_vc1, out_cc_ve1)
            out_cc_vc3, out_cc_ve3 = self.gcmp3(g, out_cc_vc2, out_cc_ve2, out_cc_vc1, out_cc_ve1)
            out_ve = self.toedge(g, torch.cat([out_cc_vc3, out_cc_vc0], dim=1), torch.cat([out_cc_ve3, out_cc_ve0, in_cc_ef], dim=1))

            # resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            # out_resource = self.mlp_resource(resource)
            # out_ve3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            
            # out_ve = self.toedge(g, out_vc3, out_ve3d6, g.edges['e_cc'].data['ef'])
            
            return out_ve
        
class GSGNN_Sub1(nn.Module):
    def __init__(self):
        super(GSGNN_Sub1, self).__init__()
        self.n_gc_nf = 6
        self.n_cc_ef = 1
        self.n_ss_ef = 2

        self.n_mp_out = 32

        self.fc_ss_nf = nn.Linear(self.n_gc_nf, 32)
        self.fc_ss_ef = nn.Linear(self.n_ss_ef, 32)
        self.gsmp1 = GSMP(32, 64)
        self.gsmp2 = GSMP(64, 128)
        self.gsmp3 = GSMP(64 + 128, 64)
        self.mlp_ss_nf = MLP(64 + 32, 64, self.n_mp_out)
        self.mlp_ss_ef = MLP(64 + 32, 64, self.n_mp_out)

        self.fc_cc_nf = nn.Linear(self.n_gc_nf + self.n_mp_out, 32)
        self.fc_cc_ef = nn.Linear(self.n_cc_ef, 32)
        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(64 + 128, 64)
        self.toedge = ToEdge(64 + 32, 64 + 32 + self.n_cc_ef, 1)

    def forward(self, g):
        with g.local_scope():
            in_gc_nf = g.nodes['gc'].data['nf']
            in_cc_ef = g.edges['e_cc'].data['ef']
            in_ss_ef = g.edges['e_ss'].data['ef']

            # modify
            in_gc_nf[:, 0:1] = 0
            in_gc_nf[:, 3:6] = 0
            in_cc_ef[:, :] = 0
            in_ss_ef[:, :] = 0

            # ss
            out_ss_vc0 = self.fc_ss_nf(in_gc_nf)
            out_ss_ve0 = self.fc_ss_ef(in_ss_ef)
            out_ss_vc1, out_ss_ve1 = self.gsmp1(g, out_ss_vc0, out_ss_ve0)
            out_ss_vc2, out_ss_ve2 = self.gsmp2(g, out_ss_vc1, out_ss_ve1)
            out_ss_vc3, out_ss_ve3 = self.gsmp3(g, out_ss_vc2, out_ss_ve2, out_ss_vc1, out_ss_ve1)
            out_ss_vc4 = self.mlp_ss_nf(torch.cat([out_ss_vc3, out_ss_vc0], dim=1))
            out_ss_ve4 = self.mlp_ss_ef(torch.cat([out_ss_ve3, out_ss_ve0], dim=1))

            # cc
            out_cc_vc0 = self.fc_cc_nf(torch.cat([in_gc_nf, out_ss_vc4], dim=1))
            out_cc_ve0 = self.fc_cc_ef(in_cc_ef)
            out_cc_vc1, out_cc_ve1 = self.gcmp1(g, out_cc_vc0, out_cc_ve0)
            out_cc_vc2, out_cc_ve2 = self.gcmp2(g, out_cc_vc1, out_cc_ve1)
            out_cc_vc3, out_cc_ve3 = self.gcmp3(g, out_cc_vc2, out_cc_ve2, out_cc_vc1, out_cc_ve1)
            out_ve = self.toedge(g, torch.cat([out_cc_vc3, out_cc_vc0], dim=1), torch.cat([out_cc_ve3, out_cc_ve0, in_cc_ef], dim=1))

            # resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            # out_resource = self.mlp_resource(resource)
            # out_ve3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            
            # out_ve = self.toedge(g, out_vc3, out_ve3d6, g.edges['e_cc'].data['ef'])
            
            return out_ve

class GSGNN_Sub2(nn.Module):
    def __init__(self):
        super(GSGNN_Sub2, self).__init__()
        self.n_gc_nf = 6
        self.n_cc_ef = 1
        self.n_ss_ef = 2

        self.n_mp_out = 32

        self.fc_ss_nf = nn.Linear(self.n_gc_nf, 32)
        self.fc_ss_ef = nn.Linear(self.n_ss_ef, 32)
        self.gsmp1 = GSMP(32, 64)
        self.gsmp2 = GSMP(64, 128)
        self.gsmp3 = GSMP(64 + 128, 64)
        self.mlp_ss_nf = MLP(64 + 32, 64, self.n_mp_out)
        self.mlp_ss_ef = MLP(64 + 32, 64, self.n_mp_out)

        self.fc_cc_nf = nn.Linear(self.n_gc_nf + self.n_mp_out, 32)
        self.fc_cc_ef = nn.Linear(self.n_cc_ef, 32)
        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(64 + 128, 64)
        self.toedge = ToEdge(64 + 32, 64 + 32 + self.n_cc_ef, 1)

    def forward(self, g):
        with g.local_scope():
            in_gc_nf = g.nodes['gc'].data['nf']
            in_cc_ef = g.edges['e_cc'].data['ef']
            in_ss_ef = g.edges['e_ss'].data['ef']

            # modify
            # in_gc_nf[:, 0:1] = 0
            # in_gc_nf[:, 3:6] = 0
            # in_cc_ef[:, :] = 0
            # in_ss_ef[:, :] = 0
            
            # ss
            out_ss_vc0 = self.fc_ss_nf(in_gc_nf)
            out_ss_ve0 = self.fc_ss_ef(in_ss_ef)
            out_ss_vc1, out_ss_ve1 = self.gsmp1(g, out_ss_vc0, out_ss_ve0)
            out_ss_vc2, out_ss_ve2 = self.gsmp2(g, out_ss_vc1, out_ss_ve1)
            out_ss_vc3, out_ss_ve3 = self.gsmp3(g, out_ss_vc2, out_ss_ve2, out_ss_vc1, out_ss_ve1)
            out_ss_vc4 = self.mlp_ss_nf(torch.cat([out_ss_vc3, out_ss_vc0], dim=1))
            out_ss_ve4 = self.mlp_ss_ef(torch.cat([out_ss_ve3, out_ss_ve0], dim=1))

            # cc
            out_cc_vc0 = self.fc_cc_nf(torch.cat([in_gc_nf, out_ss_vc4], dim=1))
            out_cc_ve0 = self.fc_cc_ef(in_cc_ef)
            out_cc_vc1, out_cc_ve1 = self.gcmp1(g, out_cc_vc0, out_cc_ve0)
            out_cc_vc2, out_cc_ve2 = self.gcmp2(g, out_cc_vc1, out_cc_ve1)
            out_cc_vc3, out_cc_ve3 = self.gcmp3(g, out_cc_vc2, out_cc_ve2, out_cc_vc1, out_cc_ve1)
            out_ve = self.toedge(g, torch.cat([out_cc_vc3, out_cc_vc0], dim=1), torch.cat([out_cc_ve3, out_cc_ve0, in_cc_ef], dim=1))

            # resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            # out_resource = self.mlp_resource(resource)
            # out_ve3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            
            # out_ve = self.toedge(g, out_vc3, out_ve3d6, g.edges['e_cc'].data['ef'])
            
            return out_ve

class GSGNN_Sub3(nn.Module):
    def __init__(self):
        super(GSGNN_Sub3, self).__init__()
        self.n_gc_nf = 4
        self.n_cc_ef = 2
        self.n_ss_ef = 2

        self.n_mp_out = 32

        self.fc_ss_nf = nn.Linear(self.n_gc_nf, 32)
        self.fc_ss_ef = nn.Linear(self.n_ss_ef, 32)
        self.gsmp1 = GSMP(32, 64)
        self.gsmp2 = GSMP(64, 128)
        self.gsmp3 = GSMP(64 + 128, 64)
        self.mlp_ss_nf = MLP(64 + 32, 64, self.n_mp_out)
        self.mlp_ss_ef = MLP(64 + 32, 64, self.n_mp_out)

        self.fc_cc_nf = nn.Linear(self.n_gc_nf + self.n_mp_out, 32)
        self.fc_cc_ef = nn.Linear(self.n_cc_ef, 32)
        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(64 + 128, 64)
        self.toedge = ToEdge(64 + 32, 64 + 32 + self.n_cc_ef, 1)

    def forward(self, g):
        with g.local_scope():
            in_gc_nf = g.nodes['gc'].data['nf']
            in_cc_ef = g.edges['e_cc'].data['ef']
            in_ss_ef = g.edges['e_ss'].data['ef']
            
            # ss
            out_ss_vc0 = self.fc_ss_nf(in_gc_nf)
            out_ss_ve0 = self.fc_ss_ef(in_ss_ef)
            out_ss_vc1, out_ss_ve1 = self.gsmp1(g, out_ss_vc0, out_ss_ve0)
            out_ss_vc2, out_ss_ve2 = self.gsmp2(g, out_ss_vc1, out_ss_ve1)
            out_ss_vc3, out_ss_ve3 = self.gsmp3(g, out_ss_vc2, out_ss_ve2, out_ss_vc1, out_ss_ve1)
            out_ss_vc4 = self.mlp_ss_nf(torch.cat([out_ss_vc3, out_ss_vc0], dim=1))
            out_ss_ve4 = self.mlp_ss_ef(torch.cat([out_ss_ve3, out_ss_ve0], dim=1))

            # cc
            out_cc_vc0 = self.fc_cc_nf(torch.cat([in_gc_nf, out_ss_vc4], dim=1))
            out_cc_ve0 = self.fc_cc_ef(in_cc_ef)
            out_cc_vc1, out_cc_ve1 = self.gcmp1(g, out_cc_vc0, out_cc_ve0)
            out_cc_vc2, out_cc_ve2 = self.gcmp2(g, out_cc_vc1, out_cc_ve1)
            out_cc_vc3, out_cc_ve3 = self.gcmp3(g, out_cc_vc2, out_cc_ve2, out_cc_vc1, out_cc_ve1)
            out_ve = self.toedge(g, torch.cat([out_cc_vc3, out_cc_vc0], dim=1), torch.cat([out_cc_ve3, out_cc_ve0, in_cc_ef], dim=1))
            
            return out_ve


class GSGNN_Sub4(nn.Module):
    def __init__(self):
        super(GSGNN_Sub4, self).__init__()
        self.n_gc_nf = 4
        self.n_cc_ef = 3
        self.n_ss_ef = 2

        self.n_mp_out = 32

        self.fc_ss_nf = nn.Linear(self.n_gc_nf, 32)
        self.fc_ss_ef = nn.Linear(self.n_ss_ef, 32)
        self.gsmp1 = GSMP(32, 64)
        self.gsmp2 = GSMP(64, 128)
        self.gsmp3 = GSMP(64 + 128, 64)
        self.mlp_ss_nf = MLP(64 + 32, 64, self.n_mp_out)
        self.mlp_ss_ef = MLP(64 + 32, 64, self.n_mp_out)

        self.fc_cc_nf = nn.Linear(self.n_gc_nf + self.n_mp_out, 32)
        self.fc_cc_ef = nn.Linear(self.n_cc_ef, 32)
        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(64 + 128, 64)
        self.toedge = ToEdge(64 + 32, 64 + 32 + self.n_cc_ef, 1)

    def forward(self, g):
        with g.local_scope():
            in_gc_nf = g.nodes['gc'].data['nf']
            in_cc_ef = g.edges['e_cc'].data['ef']
            in_ss_ef = g.edges['e_ss'].data['ef']
            
            # ss
            out_ss_vc0 = self.fc_ss_nf(in_gc_nf)
            out_ss_ve0 = self.fc_ss_ef(in_ss_ef)
            out_ss_vc1, out_ss_ve1 = self.gsmp1(g, out_ss_vc0, out_ss_ve0)
            out_ss_vc2, out_ss_ve2 = self.gsmp2(g, out_ss_vc1, out_ss_ve1)
            out_ss_vc3, out_ss_ve3 = self.gsmp3(g, out_ss_vc2, out_ss_ve2, out_ss_vc1, out_ss_ve1)
            out_ss_vc4 = self.mlp_ss_nf(torch.cat([out_ss_vc3, out_ss_vc0], dim=1))
            out_ss_ve4 = self.mlp_ss_ef(torch.cat([out_ss_ve3, out_ss_ve0], dim=1))

            # cc
            out_cc_vc0 = self.fc_cc_nf(torch.cat([in_gc_nf, out_ss_vc4], dim=1))
            out_cc_ve0 = self.fc_cc_ef(in_cc_ef)
            out_cc_vc1, out_cc_ve1 = self.gcmp1(g, out_cc_vc0, out_cc_ve0)
            out_cc_vc2, out_cc_ve2 = self.gcmp2(g, out_cc_vc1, out_cc_ve1)
            out_cc_vc3, out_cc_ve3 = self.gcmp3(g, out_cc_vc2, out_cc_ve2, out_cc_vc1, out_cc_ve1)
            out_ve = self.toedge(g, torch.cat([out_cc_vc3, out_cc_vc0], dim=1), torch.cat([out_cc_ve3, out_cc_ve0, in_cc_ef], dim=1))
            
            return out_ve

class GSGNN_Sub5(nn.Module):
    def __init__(self):
        super(GSGNN_Sub5, self).__init__()
        self.n_gc_nf = 4
        self.n_cc_ef = 3

        self.n_mp_out = 32

        self.fc_cc_nf = nn.Linear(self.n_gc_nf, 32)
        self.fc_cc_ef = nn.Linear(self.n_cc_ef, 32)
        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(64 + 128, 64)
        self.toedge = ToEdge(64 + 32, 64 + 32 + self.n_cc_ef, 1)

    def forward(self, g):
        with g.local_scope():
            in_gc_nf = g.nodes['gc'].data['nf']
            in_cc_ef = g.edges['e_cc'].data['ef']

            # cc
            out_cc_vc0 = self.fc_cc_nf(in_gc_nf)
            out_cc_ve0 = self.fc_cc_ef(in_cc_ef)
            out_cc_vc1, out_cc_ve1 = self.gcmp1(g, out_cc_vc0, out_cc_ve0)
            out_cc_vc2, out_cc_ve2 = self.gcmp2(g, out_cc_vc1, out_cc_ve1)
            out_cc_vc3, out_cc_ve3 = self.gcmp3(g, out_cc_vc2, out_cc_ve2, out_cc_vc1, out_cc_ve1)
            out_ve = self.toedge(g, torch.cat([out_cc_vc3, out_cc_vc0], dim=1), torch.cat([out_cc_ve3, out_cc_ve0, in_cc_ef], dim=1))
            
            return out_ve


class GSGNN(nn.Module):
    def __init__(self):
        super(GSGNN, self).__init__()
        self.n_gc_nf = 4
        self.n_cc_ef = 3
        self.n_ss_ef = 2

        self.n_mp_out = 32

        self.n_resource_in = 32
        self.n_resource_out = 1

        self.fc_ss_nf = nn.Linear(self.n_gc_nf, 32)
        self.fc_ss_ef = nn.Linear(self.n_ss_ef, 32)
        self.gsmp1 = GSMP(32, 64)
        self.gsmp2 = GSMP(64, 128)
        self.gsmp3 = GSMP(64 + 128, 64)
        self.mlp_ss_nf = MLP(64 + 32, 64, self.n_mp_out)
        self.mlp_ss_ef = MLP(64 + 32, 64, self.n_mp_out)

        self.fc_cc_nf = nn.Linear(self.n_gc_nf + self.n_mp_out, 32)
        self.fc_cc_ef = nn.Linear(self.n_cc_ef, 32)
        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(64 + 128, 64 + self.n_resource_in)
        self.mlp_resource = MLP(self.n_resource_in, 16, 1)
        self.toedge = ToEdge(64 + 32 + self.n_resource_in, 64 + 32 + self.n_cc_ef + self.n_resource_out, 1)

    def forward(self, g):
        with g.local_scope():
            in_gc_nf = g.nodes['gc'].data['nf']
            in_cc_ef = g.edges['e_cc'].data['ef']
            in_ss_ef = g.edges['e_ss'].data['ef']
            
            # ss
            out_ss_vc0 = self.fc_ss_nf(in_gc_nf)
            out_ss_ve0 = self.fc_ss_ef(in_ss_ef)
            out_ss_vc1, out_ss_ve1 = self.gsmp1(g, out_ss_vc0, out_ss_ve0)
            out_ss_vc2, out_ss_ve2 = self.gsmp2(g, out_ss_vc1, out_ss_ve1)
            out_ss_vc3, out_ss_ve3 = self.gsmp3(g, out_ss_vc2, out_ss_ve2, out_ss_vc1, out_ss_ve1)
            out_ss_vc4 = self.mlp_ss_nf(torch.cat([out_ss_vc3, out_ss_vc0], dim=1))
            out_ss_ve4 = self.mlp_ss_ef(torch.cat([out_ss_ve3, out_ss_ve0], dim=1))

            # cc
            out_cc_vc0 = self.fc_cc_nf(torch.cat([in_gc_nf, out_ss_vc4], dim=1))
            out_cc_ve0 = self.fc_cc_ef(in_cc_ef)
            out_cc_vc1, out_cc_ve1 = self.gcmp1(g, out_cc_vc0, out_cc_ve0)
            out_cc_vc2, out_cc_ve2 = self.gcmp2(g, out_cc_vc1, out_cc_ve1)
            out_cc_vc3, out_cc_ve3 = self.gcmp3(g, out_cc_vc2, out_cc_ve2, out_cc_vc1, out_cc_ve1)

            ve1_resource_in, ve_other = torch.split(out_cc_ve3, [self.n_resource_in, 64], dim=1)
            out_resource = self.mlp_resource(ve1_resource_in)

            out_ve = self.toedge(g, torch.cat([out_cc_vc3, out_cc_vc0], dim=1), torch.cat([ve_other, out_cc_ve0, in_cc_ef, out_resource], dim=1))
            
            return out_resource, out_ve