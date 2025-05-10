from model.model_blocks import *

class LHNN_R(nn.Module):
    def __init__(self, n_gc, n_gn, n_hid, n_out = 1):
        super(LHNN_R, self).__init__()
        self.feature_gen = Feature_Gen(n_gc, n_gn, n_hid)
        self.hyper_mp1 = HyperMP_Block(n_hid)
        self.hyper_mp2 = HyperMP_Block(n_hid)
        self.lattice_mp1 = LatticeMP_Block(n_hid)
        self.lattice_mp2 = LatticeMP_Block(n_hid)
        self.lattice_mp3 = LatticeMP_Block(n_hid)
        self.regression_head = nn.Linear(n_hid, n_out)

    def forward(self, g):
        gc_x1, gn_x1 = self.feature_gen(g, g.nodes['gc'].data['nf'], g.nodes['gn'].data['nf'])
        gc_x2, gn_x2 = self.hyper_mp1(g, gc_x1, gn_x1, gc_x1, gn_x1)
        gc_x3, gn_x3 = self.hyper_mp2(g, gc_x2, gn_x2, gc_x1, gn_x1)
        gc_x4 = self.lattice_mp1(g, gc_x3)
        gc_x5 = self.lattice_mp2(g, gc_x4)
        gc_x6 = self.lattice_mp3(g, gc_x5)
        out = self.regression_head(gc_x6)

        return out
