import dgl
import torch
from torch import nn
import dgl.function as fn

# from model.myDataLoader import train_data, test_data

class MLP(nn.Module):
    def __init__(self, *sizes, dropout:float = 0, batchnorm = False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(nn.LeakyReLU(negative_slope=0.2))

                if dropout > 0.0: fcs.append(nn.Dropout(p=dropout))
                if batchnorm: fcs.append(nn.BatchNorm1d(sizes[i]))

        self.layers = nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class Residual_Block(nn.Module):
    def __init__(self, n_hid):
        super().__init__()
        self.lin1 = nn.Linear(n_hid, n_hid)
        self.lin2 = nn.Linear(n_hid, n_hid)

    def forward(self, x):
        return self.lin2(self.lin1(x)) + x

class Feature_Gen(nn.Module):
    def __init__(self, n_gcell, n_segment, n_hid):
        super().__init__()
        self.n_segment = n_segment
        self.n_gcell = n_gcell
        self.n_hid = n_hid

        self.lin_gc1 = nn.Linear(n_gcell, n_hid)
        self.lin_gn1 = nn.Linear(n_segment, n_hid)

        self.res_gc = Residual_Block(n_hid)
        self.res_gn = Residual_Block(n_hid)

        self.mlp_msg_Gnc = MLP(n_hid * 2, n_hid * 2, n_hid * 2 + 1)
        self.mlp_reduce_Gnc = MLP(n_hid * 3, n_hid)

        self.lin_gc2 = nn.Linear(n_hid * 2, n_hid)
        self.lin_gn2 = nn.Linear(n_hid, n_hid)

    def msg_n2c(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_Gnc(x)
        k, f1, f2 = torch.split(x, [1, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efo1': f1 * k, 'efo2': f2 * k}

    def reduce_n2c(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfo1'], nodes.data['nfo2']], dim=1)
        x = self.mlp_reduce_Gnc(x)
        return {'new_x': x}

    def forward(self, g, nf_gc, nf_gn):
        with (g.local_scope()):
            g.nodes['gc'].data['x'] =  nf_gc
            g.nodes['gn'].data['x'] =  nf_gn

            g.nodes['gc'].data['x'] = self.lin_gc1(g.nodes['gc'].data['x'])
            g.nodes['gn'].data['x'] = self.lin_gn1(g.nodes['gn'].data['x'])

            g.nodes['gc'].data['x'] = self.res_gc(g.nodes['gc'].data['x'])
            g.nodes['gn'].data['x'] = self.res_gn(g.nodes['gn'].data['x'])

            g.apply_edges(self.msg_n2c, etype=('gn', 'e_n2c', 'gc'))
            g.update_all(fn.copy_e('efo1', 'efo1'), fn.sum('efo1', 'nfo1'), etype=('gn', 'e_n2c', 'gc'))
            g.update_all(fn.copy_e('efo2', 'efo2'), fn.max('efo2', 'nfo2'), etype=('gn', 'e_n2c', 'gc'))
            g.apply_nodes(self.reduce_n2c, ntype='gc')

            # g.apply_nodes(self.reduce_ground, ntype='steiner')
            return self.lin_gc2(torch.cat((g.nodes['gc'].data['x'], g.nodes['gc'].data['new_x']), dim=1)), self.lin_gn2(g.nodes['gn'].data['x'])

class HyperMP_Block(nn.Module):
    def __init__(self, n_hid):
        super().__init__()
        self.n_hid = n_hid

        self.lin_gn_in1 = nn.Linear(n_hid, n_hid)
        self.lin_Gcn = nn.Linear(n_hid, n_hid)
        self.lin_postCatGcn = nn.Linear(n_hid * 2, n_hid)

        self.lin_gc_in1 = nn.Linear(n_hid, n_hid)
        self.lin_Gnc = nn.Linear(n_hid, n_hid)
        self.lin_postCatGnc = nn.Linear(n_hid * 2, n_hid)

        self.res_gn_1 = Residual_Block(n_hid)
        self.res_gc_1 = Residual_Block(n_hid)
        self.res_gn_2 = Residual_Block(n_hid)
        self.res_gc_2 = Residual_Block(n_hid)

        self.mlp_msg_c2n = MLP(n_hid * 2, n_hid * 2, n_hid * 2 + 1)
        self.mlp_reduce_c2n = MLP(n_hid * 3, n_hid)

        self.mlp_msg_n2c = MLP(n_hid * 2, n_hid * 2, n_hid * 2 + 1)
        self.mlp_reduce_n2c = MLP(n_hid * 3, n_hid)

    def msg_c2n(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_c2n(x)
        k, f1, f2 = torch.split(x, [1, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efcno1': f1 * k, 'efcno2': f2 * k}

    def reduce_c2n(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfno1'], nodes.data['nfno2']], dim=1)
        x = self.mlp_reduce_c2n(x)
        return {'new_x': x}

    def msg_n2c(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg_n2c(x)
        k, f1, f2 = torch.split(x, [1, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efnco1': f1 * k, 'efnco2': f2 * k}

    def reduce_n2c(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfco1'], nodes.data['nfco2']], dim=1)
        x = self.mlp_reduce_n2c(x)
        return {'new_x': x}

    def forward(self, g, nf_gc, nf_gn, nf_gc_in1, nf_gn_in1):
        with (g.local_scope()):
            g.nodes['gc'].data['x'] = nf_gc
            g.nodes['gn'].data['x'] = nf_gn
            g.nodes['gc'].data['x_in1'] = nf_gc_in1
            g.nodes['gn'].data['x_in1'] = nf_gn_in1

            g.nodes['gc'].data['x_in1'] = self.lin_gc_in1(g.nodes['gc'].data['x_in1'])
            g.nodes['gn'].data['x_in1'] = self.lin_gn_in1(g.nodes['gn'].data['x_in1'])

            # block c2n
            g.nodes['gc'].data['x'] = self.res_gc_1(g.nodes['gc'].data['x'])
            g.nodes['gn'].data['x'] = self.res_gn_1(g.nodes['gn'].data['x'])

            ## c2n
            g.apply_edges(self.msg_c2n, etype=('gc', 'e_c2n', 'gn'))
            g.update_all(fn.copy_e('efcno1', 'efcno1'), fn.sum('efcno1', 'nfno1'), etype=('gc', 'e_c2n', 'gn'))
            g.update_all(fn.copy_e('efcno2', 'efcno2'), fn.max('efcno2', 'nfno2'), etype=('gc', 'e_c2n', 'gn'))
            g.apply_nodes(self.reduce_c2n, ntype='gn')

            g.nodes['gn'].data['new_x'] = self.lin_Gcn(g.nodes['gn'].data['new_x'])
            cat_value_c2n = torch.cat((g.nodes['gn'].data['new_x'], g.nodes['gn'].data['x_in1']), dim=1)
            g.nodes['gn'].data['x'] = g.nodes['gn'].data['x'] + self.lin_postCatGcn(cat_value_c2n)

            # block n2c
            g.nodes['gn'].data['x'] = self.res_gn_2(g.nodes['gn'].data['x'])
            g.nodes['gc'].data['x'] = self.res_gc_2(g.nodes['gc'].data['x'])

            ## n2c
            g.apply_edges(self.msg_n2c, etype=('gn', 'e_n2c', 'gc'))
            g.update_all(fn.copy_e('efnco1', 'efnco1'), fn.sum('efnco1', 'nfco1'), etype=('gn', 'e_n2c', 'gc'))
            g.update_all(fn.copy_e('efnco2', 'efnco2'), fn.max('efnco2', 'nfco2'), etype=('gn', 'e_n2c', 'gc'))
            g.apply_nodes(self.reduce_n2c, ntype='gc')

            g.nodes['gc'].data['new_x'] = self.lin_Gnc(g.nodes['gc'].data['new_x'])
            cat_value_n2c = torch.cat((g.nodes['gc'].data['new_x'], g.nodes['gc'].data['x_in1']), dim=1)
            g.nodes['gc'].data['x'] = g.nodes['gc'].data['x'] + self.lin_postCatGnc(cat_value_n2c)

            return g.nodes['gc'].data['x'], g.nodes['gn'].data['x']

# TODO: random sample
class LatticeMP_Block(nn.Module):
    def __init__(self, n_hid):
        super().__init__()
        self.n_hid = n_hid

        self.res = Residual_Block(n_hid)
        self.lin1 = nn.Linear(n_hid, n_hid)
        self.lin2 = nn.Linear(n_hid, n_hid)

        self.mlp_msg = MLP(n_hid * 2, n_hid * 2, n_hid * 2 + 1)
        self.mlp_reduce = MLP(n_hid * 3, n_hid)

    def msg(self, edges):
        x = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        x = self.mlp_msg(x)
        k, f1, f2 = torch.split(x, [1, self.n_hid, self.n_hid], dim=1)
        k = torch.sigmoid(k)
        return {'efcco1': f1 * k, 'efcco2': f2 * k}

    def reduce(self, nodes):
        x = torch.cat([nodes.data['x'], nodes.data['nfoc1'], nodes.data['nfoc2']], dim=1)
        x = self.mlp_reduce(x)
        return {'new_x': x}

    def forward(self, g, nf_gc):
        with (g.local_scope()):
            g.nodes['gc'].data['x'] = nf_gc

            g.nodes['gc'].data['x'] = self.res(g.nodes['gc'].data['x'])

            g.nodes['gc'].data['x1'] = self.lin1(g.nodes['gc'].data['x'])

            g.apply_edges(self.msg, etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco1', 'efcco1'), fn.sum('efcco1', 'nfoc1'), etype=('gc', 'e_cc', 'gc'))
            g.update_all(fn.copy_e('efcco2', 'efcco2'), fn.max('efcco2', 'nfoc2'), etype=('gc', 'e_cc', 'gc'))
            g.apply_nodes(self.reduce, ntype='gc')

            g.nodes['gc'].data['x2'] = self.lin2(g.nodes['gc'].data['new_x'])

            return g.nodes['gc'].data['x1'] + g.nodes['gc'].data['x2']









