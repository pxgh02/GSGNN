import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=4):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # 输入层: 多头注意力
        self.layers.append(GATConv(
            in_feats=input_dim,
            out_feats=hidden_dim,
            num_heads=num_heads,
            feat_drop=0.5,
            attn_drop=0.5,
            residual=True
        ))
        
        # 中间层: 多头注意力
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(
                in_feats=hidden_dim * num_heads,  # 因为前一层是多头的
                out_feats=hidden_dim,
                num_heads=num_heads,
                feat_drop=0.5,
                attn_drop=0.5,
                residual=True
            ))
        
        # 输出层: 单头注意力，得到最终输出
        self.layers.append(GATConv(
            in_feats=hidden_dim * num_heads,
            out_feats=output_dim,
            num_heads=1,  # 最后一层使用单头
            feat_drop=0.5,
            attn_drop=0.5,
            residual=True
        ))
        
        self.activation = F.elu  # GAT论文中使用ELU激活函数

    def forward(self, graph):
        # 取特定边类型的子图
        subgraph = graph.edge_type_subgraph([('gc', 'e_cc', 'gc')])
        
        # 获取输入特征
        h = graph.nodes['gc'].data['nf']
        
        # 通过GAT层
        for i, layer in enumerate(self.layers):
            h = layer(subgraph, h)
            
            if i != len(self.layers) - 1:  # 非最后一层
                h = h.reshape(h.shape[0], -1)  # 将多头注意力的结果展平
                h = self.activation(h)
            else:  # 最后一层
                h = h.mean(1)  # 如果是多头的话取平均
        
        return h 