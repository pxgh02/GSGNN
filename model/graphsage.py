import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, aggregator_type='mean'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # 输入层
        self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type))
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type))
        # 输出层
        self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator_type))

        self.activation = F.relu
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph):
        # 取特定边类型的子图
        subgraph = graph.edge_type_subgraph([('gc', 'e_cc', 'gc')])

        # 确保输入特征只包含 gcell 类型节点
        h = graph.nodes['gc'].data['nf']
        for i, layer in enumerate(self.layers):
            h = layer(subgraph, h)
            if i != len(self.layers) - 1:  # 最后一层不需要激活和dropout
                h = self.activation(h)
                h = self.dropout(h)
        return h

