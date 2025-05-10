import torch
from torch import nn
from model.model_blocks import *

class Embedding_Model(nn.Module):
    def __init__(self, n_gc, n_gs, n_hid):
        super(Embedding_Model, self).__init__()
        self.n_gc = n_gc
        self.n_gs = n_gs
        self.n_hid = n_hid

        self.lin_gc = nn.Linear(n_gc, n_hid)
        self.lin_gs = nn.Linear(n_gs, n_hid)

        self.mlp_msg_Gs2c = MLP(n_hid * 2, n_hid * 2, n_hid * 4 + 1)
        self.mlp_reduce_Gs2c = MLP(n_hid * 5, n_hid)

        self.mlp_msg_Gc2s = MLP(n_hid * 2, n_hid * 2, n_hid * 4 + 1)
        self.mlp_reduce_Gc2s = MLP(n_hid * 5, n_hid)
        
        self.mlp_gc = MLP(2 * n_hid, n_hid, n_hid)
        self.mlp_gs = MLP(2 * n_hid, n_hid, n_hid)
        
    def msg_s2c(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_Gs2c(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_hid, self.n_hid, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efco1': f1 * k, 'efco2': f2 * k, 'efco3': f3 * k, 'efco4': f4 * k}
    
    def reduce_s2c(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfco1'], nodes.data['nfco2'], nodes.data['nfco3'], nodes.data['nfco4']], dim=1)
        x = self.mlp_reduce_Gs2c(x)
        return {'new_cx': x}
    
    def msg_c2s(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_Gc2s(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_hid, self.n_hid, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efso1': f1 * k, 'efso2': f2 * k, 'efso3': f3 * k, 'efso4': f4 * k}
    
    def reduce_c2s(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfso1'], nodes.data['nfso2'], nodes.data['nfso3'], nodes.data['nfso4']], dim=1)
        x = self.mlp_reduce_Gc2s(x)
        return {'new_sx': x}

    def forward(self, g, nf_gc, nf_gs):
        with (g.local_scope()):
            gc_embed = self.lin_gc(nf_gc)
            gs_embed = self.lin_gs(nf_gs)

            g.nodes['gc'].data['x'] = gc_embed
            g.nodes['gs'].data['x'] = gs_embed

            g.apply_edges(self.msg_s2c, etype=('gs', 'e_s2c', 'gc'))
            g.update_all(fn.copy_e('efco1', 'efco1'), fn.sum('efco1', 'nfco1'), etype=('gs', 'e_s2c', 'gc'))
            g.update_all(fn.copy_e('efco2', 'efco2'), fn.max('efco2', 'nfco2'), etype=('gs', 'e_s2c', 'gc'))
            g.update_all(fn.copy_e('efco3', 'efco3'), fn.min('efco3', 'nfco3'), etype=('gs', 'e_s2c', 'gc'))
            g.update_all(fn.copy_e('efco4', 'efco4'), fn.sum('efco4', 'nfco4'), etype=('gs', 'e_s2c', 'gc'))
            g.apply_nodes(self.reduce_s2c, ntype='gc')

            g.apply_edges(self.msg_c2s, etype=('gc', 'e_c2s', 'gs'))
            g.update_all(fn.copy_e('efso1', 'efso1'), fn.sum('efso1', 'nfso1'), etype=('gc', 'e_c2s', 'gs'))
            g.update_all(fn.copy_e('efso2', 'efso2'), fn.max('efso2', 'nfso2'), etype=('gc', 'e_c2s', 'gs'))
            g.update_all(fn.copy_e('efso3', 'efso3'), fn.min('efso3', 'nfso3'), etype=('gc', 'e_c2s', 'gs'))
            g.update_all(fn.copy_e('efso4', 'efso4'), fn.sum('efso4', 'nfso4'), etype=('gc', 'e_c2s', 'gs'))
            g.apply_nodes(self.reduce_c2s, ntype='gs')
            
            out_fc = self.mlp_gc(torch.cat([g.nodes['gc'].data['x'],  g.nodes['gc'].data['new_cx']], dim=1))
            out_fs = self.mlp_gs(torch.cat([g.nodes['gs'].data['x'],  g.nodes['gs'].data['new_sx']], dim=1))
            return out_fc, out_fs

class GSegmentDown_Model(nn.Module):
    def __init__(self, n_hid):
        super(GSegmentDown_Model, self).__init__()
        self.n_hid = n_hid

        self.mlp_msg_Gc2s = MLP(n_hid * 3, n_hid * 2, n_hid * 4 + 1)
        self.mlp_reduce_Gc2s = MLP(n_hid * 5, n_hid)
        
        self.mlp_msg_Gss = MLP(n_hid * 2, n_hid * 2, n_hid * 4 + 1)
        self.mlp_reduce_Gss = MLP(n_hid * 5, n_hid)
        
        self.lin_gc = nn.Linear(n_hid * 2, n_hid)
        self.lin_gs = nn.Linear(n_hid, n_hid)

        
    def msg_ss(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_Gss(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_hid, self.n_hid, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efsso1': f1 * k, 'efsso2': f2 * k, 'efsso3': f3 * k, 'efsso4': f4 * k}
    
    def reduce_ss(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfsso1'], nodes.data['nfsso2'], nodes.data['nfsso3'], nodes.data['nfsso4']], dim=1)
        x = self.mlp_reduce_Gss(x)
        return {'new_ssx': x}
    
    def msg_c2s(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_Gc2s(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_hid, self.n_hid, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efso1': f1 * k, 'efso2': f2 * k, 'efso3': f3 * k, 'efso4': f4 * k}
    
    def reduce_c2s(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfso1'], nodes.data['nfso2'], nodes.data['nfso3'], nodes.data['nfso4']], dim=1)
        x = self.mlp_reduce_Gc2s(x)
        return {'new_sx': x}

    def forward(self, g, nf_gc0, nf_gc1, nf_gs):
        with (g.local_scope()):
            g.nodes['gc'].data['x'] = torch.cat([nf_gc0, nf_gc1], dim=1)
            g.nodes['gs'].data['x'] = nf_gs
            
            g.apply_edges(self.msg_ss, etype=('gs', 'e_ss1', 'gs'))
            g.update_all(fn.copy_e('efsso1', 'efsso1'), fn.sum('efsso1', 'nfsso1'), etype=('gs', 'e_ss1', 'gs'))
            g.update_all(fn.copy_e('efsso2', 'efsso2'), fn.max('efsso2', 'nfsso2'), etype=('gs', 'e_ss1', 'gs'))
            g.update_all(fn.copy_e('efsso3', 'efsso3'), fn.min('efsso3', 'nfsso3'), etype=('gs', 'e_ss1', 'gs'))
            g.update_all(fn.copy_e('efsso4', 'efsso4'), fn.mean('efsso4', 'nfsso4'), etype=('gs', 'e_ss1', 'gs'))
            g.apply_nodes(self.reduce_ss, ntype='gs')

            g.apply_edges(self.msg_c2s, etype=('gc', 'e_c2s', 'gs'))
            g.update_all(fn.copy_e('efso1', 'efso1'), fn.sum('efso1', 'nfso1'), etype=('gc', 'e_c2s', 'gs'))
            g.update_all(fn.copy_e('efso2', 'efso2'), fn.max('efso2', 'nfso2'), etype=('gc', 'e_c2s', 'gs'))
            g.update_all(fn.copy_e('efso3', 'efso3'), fn.min('efso3', 'nfso3'), etype=('gc', 'e_c2s', 'gs'))
            g.update_all(fn.copy_e('efso4', 'efso4'), fn.mean('efso4', 'nfso4'), etype=('gc', 'e_c2s', 'gs'))
            g.apply_nodes(self.reduce_c2s, ntype='gs')
            
            out_fc = self.lin_gc(g.nodes['gc'].data['x'])
            out_fs = self.lin_gs(g.nodes['gs'].data['new_ssx'] + g.nodes['gs'].data['new_sx'])
            
            return out_fc, out_fs

class GCellUp_Model(nn.Module):
    def __init__(self, n_hid):
        super(GCellUp_Model, self).__init__()
        self.n_hid = n_hid

        self.mlp_msg_Gs2c = MLP(n_hid * 3, n_hid * 2, n_hid * 4 + 1)
        self.mlp_reduce_Gs2c = MLP(n_hid * 5, n_hid)
        
        self.mlp_msg_Gcc = MLP(n_hid * 2, n_hid * 2, n_hid * 4 + 1)
        self.mlp_reduce_Gcc = MLP(n_hid * 5, n_hid)
        
        self.lin_gc = nn.Linear(n_hid, n_hid)
        self.lin_gs = nn.Linear(n_hid * 2, n_hid)
        
    def msg_cc(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_Gcc(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_hid, self.n_hid, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efcco1': f1 * k, 'efcco2': f2 * k, 'efcco3': f3 * k, 'efcco4': f4 * k}
    
    def reduce_cc(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfcco1'], nodes.data['nfcco2'], nodes.data['nfcco3'], nodes.data['nfcco4']], dim=1)
        x = self.mlp_reduce_Gcc(x)
        return {'new_ccx': x}
    
    def msg_s2c(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_Gs2c(x)
        k, f1, f2, f3, f4 = torch.split(x, [1, self.n_hid, self.n_hid, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efco1': f1 * k, 'efco2': f2 * k, 'efco3': f3 * k, 'efco4': f4 * k}
    
    def reduce_s2c(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfco1'], nodes.data['nfco2'], nodes.data['nfco3'], nodes.data['nfco4']], dim=1)
        x = self.mlp_reduce_Gs2c(x)
        return {'new_cx': x}

    def forward(self, g, nf_gc, nf_gs0, nf_gs1):
        with (g.local_scope()):
            g.nodes['gc'].data['x'] = nf_gc
            g.nodes['gs'].data['x'] = torch.cat([nf_gs0, nf_gs1], dim=1)
            
            g.apply_edges(self.msg_cc, etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco1', 'efcco1'), fn.sum('efcco1', 'nfcco1'), etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco2', 'efcco2'), fn.max('efcco2', 'nfcco2'), etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco3', 'efcco3'), fn.min('efcco3', 'nfcco3'), etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco4', 'efcco4'), fn.mean('efcco4', 'nfcco4'), etype=('gc', 'e_cc', 'gc'))
            g.apply_nodes(self.reduce_cc, ntype='gc')

            g.apply_edges(self.msg_s2c, etype=('gs', 'e_s2c', 'gc'))
            g.update_all(fn.copy_e('efco1', 'efco1'), fn.sum('efco1', 'nfco1'), etype=('gs', 'e_s2c', 'gc'))
            g.update_all(fn.copy_e('efco2', 'efco2'), fn.max('efco2', 'nfco2'), etype=('gs', 'e_s2c', 'gc'))
            g.update_all(fn.copy_e('efco3', 'efco3'), fn.min('efco3', 'nfco3'), etype=('gs', 'e_s2c', 'gc'))
            g.update_all(fn.copy_e('efco4', 'efco4'), fn.mean('efco4', 'nfco4'), etype=('gs', 'e_s2c', 'gc'))
            g.apply_nodes(self.reduce_s2c, ntype='gc')
            
            out_fc = self.lin_gc(g.nodes['gc'].data['new_ccx'] + g.nodes['gc'].data['new_cx'])
            out_fs = self.lin_gs(g.nodes['gs'].data['x'])
            
            return out_fc, out_fs

class MyModel_C(nn.Module):
    def __init__(self, n_gc, n_gs, n_hid, n_out_congestion, n_out_WL):
        super(MyModel_C, self).__init__()
        
        self.aux_fgc0 = nn.Linear(n_gc, n_hid)
        
        self.em = Embedding_Model(n_gc, n_gs, n_hid)
        
        self.gsdm1 = GSegmentDown_Model(n_hid)
        self.gsdm2 = GSegmentDown_Model(n_hid)
        self.gsdm3 = GSegmentDown_Model(n_hid)
        
        self.gcum1 = GCellUp_Model(n_hid)
        self.gcum2 = GCellUp_Model(n_hid)
        self.gcum3 = GCellUp_Model(n_hid)
        
        self.mlp_congestion = MLP(n_hid, n_hid, n_hid)
        self.lin_congestion = nn.Linear(n_hid, n_out_congestion)
        
        self.mlp_WL = MLP(n_hid + n_gs, n_hid, n_hid)
        self.lin_WL = nn.Linear(n_hid, n_out_WL)
        
    def forward(self, g):
        nf_gc = g.nodes['gc'].data['nf']
        nf_gs = g.nodes['gs'].data['nf']

        fgc0 = self.aux_fgc0(nf_gc)
        
        fgc1, fgs1 = self.em(g, nf_gc, nf_gs)
        fgc2, fgs2 = self.gsdm1(g, fgc1, fgc0, fgs1)
        fgc3, fgs3 = self.gsdm2(g, fgc2, fgc1, fgs2)
        fgc4, fgs4 = self.gsdm3(g, fgc3, fgc2, fgs3)
        
        fgc5, fgs5 = self.gcum1(g, fgc4, fgs4, fgs3)
        fgc6, fgs6 = self.gcum2(g, fgc5, fgs5, fgs2)
        fgc7, fgs7 = self.gcum3(g, fgc6, fgs6, fgs1)
        
        congestion_out = self.mlp_congestion(fgc7)
        congestion_out = self.lin_congestion(congestion_out)

        WL_out = self.mlp_WL(torch.cat([fgs7, nf_gs], dim=1))
        WL_out = torch.sum(self.lin_WL(WL_out))
        
        return congestion_out, WL_out

class MyModel_C_notWL(nn.Module):
    def __init__(self, n_gc, n_gs, n_hid, n_out_congestion):
        super(MyModel_C_notWL, self).__init__()
        
        self.aux_fgc0 = nn.Linear(n_gc, n_hid)
        
        self.em = Embedding_Model(n_gc, n_gs, n_hid)
        
        self.gsdm1 = GSegmentDown_Model(n_hid)
        self.gsdm2 = GSegmentDown_Model(n_hid)
        self.gsdm3 = GSegmentDown_Model(n_hid)
        
        self.gcum1 = GCellUp_Model(n_hid)
        self.gcum2 = GCellUp_Model(n_hid)
        self.gcum3 = GCellUp_Model(n_hid)
        
        self.mlp_congestion = MLP(n_hid, n_hid, n_hid)
        self.lin_congestion = nn.Linear(n_hid, n_out_congestion)


    def forward(self, g):
        nf_gc = g.nodes['gc'].data['nf']
        nf_gs = g.nodes['gs'].data['nf']

        fgc0 = self.aux_fgc0(nf_gc)
        
        fgc1, fgs1 = self.em(g, nf_gc, nf_gs)
        fgc2, fgs2 = self.gsdm1(g, fgc1, fgc0, fgs1)
        fgc3, fgs3 = self.gsdm2(g, fgc2, fgc1, fgs2)
        fgc4, fgs4 = self.gsdm3(g, fgc3, fgc2, fgs3)
        
        fgc5, fgs5 = self.gcum1(g, fgc4, fgs4, fgs3)
        fgc6, fgs6 = self.gcum2(g, fgc5, fgs5, fgs2)
        fgc7, fgs7 = self.gcum3(g, fgc6, fgs6, fgs1)
        
        congestion_out = self.mlp_congestion(fgc7)
        congestion_out = self.lin_congestion(congestion_out)
        
        return congestion_out