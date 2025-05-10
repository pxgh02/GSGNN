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
            out_ve = self.mlp_ve_down(torch.cat([g.edges['e_cc'].data['efcco1'], g.edges['e_cc'].data['efcco2'], g.edges['e_cc'].data['efcco3'], g.edges['e_cc'].data['efcco4'], g.edges['e_cc'].data['in_ef']], dim=1))

            out_vc = self.bn_gc(out_vc)
            out_ve = self.bn_ef(out_ve)

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

    def forward(self, g, in_vc, in_ve1, in_ve2=None):
        with g.local_scope():
            g.nodes['gc'].data['in_nf'] = in_vc
            if in_ve2 is not None:
                g.edges['e_cc'].data['in_ef'] = torch.cat([in_ve1, in_ve2], dim=1)
            else:
                g.edges['e_cc'].data['in_ef'] = in_ve1
            g.apply_edges(self.msg_cc, etype=('gc', 'e_cc', 'gc'))
            out_ve = g.edges['e_cc'].data['out_ve']
        return out_ve

class GSMP(nn.Module):
    def __init__(self, n_in, n_out):
        super(GSMP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.mlp_msg_Gss = MLP(n_in * 3, n_out * 3, n_out * 4 + 1)
        self.mlp_reduce_Gss = MLP(n_out * 4 + n_in, n_out)

        self.mlp_ve_down = MLP(n_out * 4 + n_in, n_out)
    
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

    def forward(self, g, in_vc, in_ve):
        with g.local_scope():
            g.nodes['gc'].data['in_nf'] = in_vc
            g.edges['e_ss'].data['in_ef'] = in_ve
            g.apply_edges(self.msg_ss, etype=('gc', 'e_ss', 'gc'))
            g.update_all(fn.copy_e('efss1', 'efss1'), fn.sum('efss1', 'nfss1'), etype=('gc', 'e_ss', 'gc'))
            g.update_all(fn.copy_e('efss2', 'efss2'), fn.max('efss2', 'nfss2'), etype=('gc',    'e_ss', 'gc'))
            g.update_all(fn.copy_e('efss3', 'efss3'), fn.min('efss3', 'nfss3'), etype=('gc',    'e_ss', 'gc'))
            g.update_all(fn.copy_e('efss4', 'efss4'), fn.mean('efss4', 'nfss4'), etype=('gc',   'e_ss', 'gc'))
            g.apply_nodes(self.reduce_ss, ntype='gc')
    
            out_vc = g.nodes['gc'].data['new_ssnx']
            out_ve = self.mlp_ve_down(torch.cat([g.edges['e_ss'].data['efss1'], g.edges['e_ss'].data['efss2'], g.edges['e_ss'].data['efss3'], g.edges['e_ss'].data['efss4'], g.edges['e_ss'].data['in_ef']], dim=1))

            return out_vc, out_ve

class GC_Conv(nn.Module):
    def __init__(self, n_gc_nf, n_gc_ef, n_edge_out=1):
        super(GC_Conv, self).__init__()
        self.n_gc_nf = n_gc_nf
        self.n_gc_ef = n_gc_ef

        self.fc_gc = nn.Linear(n_gc_nf, 32)
        self.fc_ef = nn.Linear(n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        # self.gcmp4 = GCMP(64, 32)
        self.toedge = ToEdge(64, 64, 1, n_edge_out)

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

class GS_Conv(nn.Module):
    def __init__(self, n_gc_nf, n_gc_ef):
        super(GS_Conv, self).__init__()
        self.n_gc_nf = n_gc_nf
        self.n_gc_ef = n_gc_ef

        self.fc_gc = nn.Linear(n_gc_nf, 32)
        self.fc_ef = nn.Linear(n_gc_ef, 32)

        self.gsmp1 = GSMP(32, 64)
        self.gsmp2 = GSMP(64, 128)
        self.gsmp3 = GSMP(128, 64)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_ss'].data['ef'])

            out_vc1, out_ve1 = self.gsmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gsmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gsmp3(g, out_vc2, out_ve2)
            # out_vc4, out_ve4 = self.gcmp4(g, out_vc3, out_ve3)
            # out_ve = self.toedge(g, out_vc3, out_ve3, g.edges['e_cc'].data['ef'])
            return out_vc3

# only GC_Conv
class GSCGNN_V1(nn.Module):
    def __init__(self):
        super(GSCGNN_V1, self).__init__()
        self.gc_conv = GC_Conv(6, 1)

    def forward(self, g):
        return self.gc_conv(g)

# only GS_Conv
class GSCGNN_V2(nn.Module):
    def __init__(self):
        super(GSCGNN_V2, self).__init__()
        self.gs_conv = GS_Conv(6, 2)
        self.toedge = ToEdge(64, 1, 0, 1)

    def forward(self, g):
        out_vc1 =self.gs_conv(g)
        out_ve = self.toedge(g, out_vc1, g.edges['e_cc'].data['ef'])
        return out_ve

# combine separate
## experiment show that this model is not good
class GSCGNN_V3(nn.Module):
    def __init__(self):
        super(GSCGNN_V3, self).__init__()
        self.gc_conv = GC_Conv(6, 1)
        self.gs_conv = GS_Conv(6, 2)
        self.toedge = ToEdge(64, 1, 1, 1)

    def forward(self, g):
        out_ve1 = self.gc_conv(g)
        out_vc1 = self.gs_conv(g)
        out_ve2 = self.toedge(g, out_vc1, out_ve1, g.edges['e_cc'].data['ef'])
        return out_ve2

# V1 + Resource Auxiliary task
class GSCGNN_V4(nn.Module):
    def __init__(self):
        super(GSCGNN_V4, self).__init__()
        self.n_gc_nf = 6
        self.n_gc_ef = 1

        self.fc_gc = nn.Linear(self.n_gc_nf, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        # self.gcmp4 = GCMP(64, 32)
        self.toedge = ToEdge(64, 32+1, 1, 1)
        self.mlp_resource = MLP(32, 32, 1)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_vc3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            out_ve = self.toedge(g, out_vc3, out_vc3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve

# V4 + change netdensity, change dataset
class GSCGNN_V5(nn.Module):
    def __init__(self):
        super(GSCGNN_V5, self).__init__()
        self.n_gc_nf = 4
        self.n_gc_ef = 2

        self.fc_gc = nn.Linear(self.n_gc_nf, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        self.toedge = ToEdge(64, 32+1, self.n_gc_ef, 1)
        self.mlp_resource = MLP(32, 32, 1)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_vc3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            out_ve = self.toedge(g, out_vc3, out_vc3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve

#change dataset, add ef isHorizontal, Lweight = 2
class GSCGNN_V6(nn.Module):
    def __init__(self):
        super(GSCGNN_V6, self).__init__()
        self.n_gc_nf = 4
        self.n_gc_ef = 3

        self.fc_gc = nn.Linear(self.n_gc_nf, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        self.toedge = ToEdge(64, 32+1, self.n_gc_ef, 1)
        self.mlp_resource = MLP(32, 32, 1)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_vc3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            out_ve = self.toedge(g, out_vc3, out_vc3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve

#V6-based change dataset, modify netdensity
class GSCGNN_V7(nn.Module):
    def __init__(self):
        super(GSCGNN_V7, self).__init__()
        self.n_gc_nf = 4
        self.n_gc_ef = 3

        self.fc_gc = nn.Linear(self.n_gc_nf, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        self.toedge = ToEdge(64, 32+1, self.n_gc_ef, 1)
        self.mlp_resource = MLP(32, 32, 1)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_vc3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            out_ve = self.toedge(g, out_vc3, out_vc3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve

#V7 + change dataset, netdensity_init + netdensity_congestion_aware
class GSCGNN_V8(nn.Module):
    def __init__(self):
        super(GSCGNN_V8, self).__init__()
        self.n_gc_nf = 4
        self.n_gc_ef = 4

        self.fc_gc = nn.Linear(self.n_gc_nf, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        self.toedge = ToEdge(64, 32+1, self.n_gc_ef, 1)
        self.mlp_resource = MLP(32, 32, 1)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_vc3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            out_ve = self.toedge(g, out_vc3, out_vc3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve

#new change dataset, half-side congestion aware, use V6 dataset
## achieve good result
class GSCGNN_V9(nn.Module):
    def __init__(self):
        super(GSCGNN_V9, self).__init__()
        self.n_gc_nf = 4
        self.n_gc_ef = 3

        self.fc_gc = nn.Linear(self.n_gc_nf, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        self.toedge = ToEdge(64, 32+1, self.n_gc_ef, 1)
        self.mlp_resource = MLP(32, 32, 1)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_vc3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            out_ve = self.toedge(g, out_vc3, out_vc3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve

# V9-based add e_ss
# modify ef_ss, scalar to vector
class GSCGNN_V10(nn.Module):
    def __init__(self):
        super(GSCGNN_V10, self).__init__()
        self.n_gc_nf = 4
        self.n_gc_ef = 3

        self.fc_gc = nn.Linear(self.n_gc_nf, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)

        self.gs_conv = GS_Conv(4, 2)

        self.toedge = ToEdge(64 + 64, 32+1, self.n_gc_ef, 1)
        self.mlp_resource = MLP(32, 32, 1)
        
    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_ve3d6 = torch.cat([out_resource, out_ve3d5], dim=1)

            out_vc_ess = self.gs_conv(g)    
            comb_vc = torch.cat([out_vc_ess, out_vc3], dim=1)
            out_ve = self.toedge(g, comb_vc, out_ve3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve

#refine struction，V9-based
class GSCGNN_V11(nn.Module):
    def __init__(self):
        super(GSCGNN_V11, self).__init__()
        self.n_gc_nf = 4
        self.n_gc_ef = 3

        self.fc_gc = nn.Linear(self.n_gc_nf, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)
        self.toedge = ToEdge(64, 32+1, self.n_gc_ef, 1)
        self.mlp_resource = MLP(32, 32, 1)

    def forward(self, g):
        with g.local_scope():
            in_vc = self.fc_gc(g.nodes['gc'].data['nf'])
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_vc3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            out_ve = self.toedge(g, out_vc3, out_vc3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve

#refine struction，V10-based
class GSCGNN_V12(nn.Module):
    def __init__(self):
        super(GSCGNN_V12, self).__init__()
        self.n_gc_nf = 4
        self.n_gc_ef = 3

        self.fc_gc = nn.Linear(self.n_gc_nf + 64, 32)
        self.fc_ef = nn.Linear(self.n_gc_ef, 32)

        self.gcmp1 = GCMP(32, 64)
        self.gcmp2 = GCMP(64, 128)
        self.gcmp3 = GCMP(128, 64)

        self.gs_conv = GS_Conv(4, 2)

        self.toedge = ToEdge(64, 32+1, self.n_gc_ef, 1)
        self.mlp_resource = MLP(32, 32, 1)

    def forward(self, g):
        with g.local_scope():
            out_vc_ess = self.gs_conv(g)

            in_vc = self.fc_gc(torch.cat([out_vc_ess, g.nodes['gc'].data['nf']], dim=1))
            in_ve = self.fc_ef(g.edges['e_cc'].data['ef'])

            out_vc1, out_ve1 = self.gcmp1(g, in_vc, in_ve)
            out_vc2, out_ve2 = self.gcmp2(g, out_vc1, out_ve1)
            out_vc3, out_ve3 = self.gcmp3(g, out_vc2, out_ve2)
            resource, out_ve3d5 = torch.split(out_ve3, [32, 32], dim=1)
            out_resource = self.mlp_resource(resource)
            out_ve3d6 = torch.cat([out_resource, out_ve3d5], dim=1)
            
            out_ve = self.toedge(g, out_vc3, out_ve3d6, g.edges['e_cc'].data['ef'])
            
            return out_resource, out_ve




