import copy

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import accuracy as accuracy_1d
from torch.nn import Dropout, SELU
from torch_geometric.nn import MessagePassing, SAGEConv, GCNConv, GATConv, TransformerConv, GATv2Conv, GraphConv
from torch_sparse import matmul, SparseTensor
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import to_dense_adj


class KProp(MessagePassing):
    def __init__(self, steps, aggregator, add_self_loops, normalize, cached, transform=lambda x: x):
        super().__init__(aggr=aggregator)
        self.transform = transform
        self.K = steps
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self._cached_x = None

    def forward(self, x, adj_t):
        if self._cached_x is None or not self.cached:
            self._cached_x = self.neighborhood_aggregation(x, adj_t)
        # print(self._cached_x.shape, 'cached x')
        return self._cached_x

    def neighborhood_aggregation(self, x, adj_t):
        if self.K <= 0:
            return x

        if self.normalize:
            adj_t = gcn_norm(adj_t, add_self_loops=False)

        if self.add_self_loops:
            adj_t = adj_t.set_diag()

        for k in range(self.K):
            x = self.propagate(adj_t, x=x)

        x = self.transform(x)
        return x

    def message_and_aggregate(self, adj_t, x):  # noqa
        return matmul(adj_t, x, reduce=self.aggr)


class GNN(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv1 = None
        self.conv2 = None
        self.dropout = Dropout(p=dropout)
        self.activation = SELU(inplace=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        return x


class GCN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)


class GAT(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        heads = 4
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(heads * hidden_dim, output_dim, heads=1, concat=False)


class GraphSAGE(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim, normalize=False, root_weight=True)
        self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=output_dim, normalize=False, root_weight=True)


class GraphTransformer(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, root_weight=True)
        self.conv2 = TransformerConv(in_channels=hidden_dim, out_channels=output_dim, root_weight=True)
        
        
class GAT2(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = GATv2Conv(in_channels=input_dim, out_channels=hidden_dim)
        self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=output_dim)


class GraphConvNN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = GraphConv(in_channels=input_dim, out_channels=hidden_dim)
        self.conv2 = GraphConv(in_channels=hidden_dim, out_channels=output_dim)


class DirGNNConv(torch.nn.Module):
    r"""A generic wrapper for computing graph convolution on directed
    graphs as described in the `"Edge Directionality Improves Learning on
    Heterophilic Graphs" <https://arxiv.org/abs/2305.10498>`_ paper.
    :class:`DirGNNConv` will pass messages both from source nodes to target
    nodes and from target nodes to source nodes.

    Args:
        conv (MessagePassing): The underlying
            :class:`~torch_geometric.nn.conv.MessagePassing` layer to use.
        alpha (float, optional): The alpha coefficient used to weight the
            aggregations of in- and out-edges as part of a convex combination.
            (default: :obj:`0.5`)
        root_weight (bool, optional): If set to :obj:`True`, the layer will add
            transformed root node features to the output.
            (default: :obj:`True`)
    """

    def __init__(
            self,
            conv: MessagePassing,
            alpha: float = 0.5,
            root_weight: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.root_weight = root_weight

        self.conv_in = copy.deepcopy(conv)
        self.conv_out = copy.deepcopy(conv)

        if hasattr(conv, 'add_self_loops'):
            self.conv_in.add_self_loops = False
            self.conv_out.add_self_loops = False
        if hasattr(conv, 'root_weight'):
            self.conv_in.root_weight = False
            self.conv_out.root_weight = False

        if root_weight:
            self.lin = torch.nn.Linear(conv.in_channels, conv.out_channels)
        else:
            self.lin = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv_in.reset_parameters()
        self.conv_out.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """"""  # noqa: D419
        x_in = self.conv_in(x, edge_index)
        
        
        # TODO: complains about sparse stuff. Note that I've bee
        # Original was simply:
        # x_out = self.conv_out(x, edge_index.flip([0]))
        # from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/dir_gnn_conv.html#DirGNNConv

        ### Trying to fix the complaints of the original x_out ###
        if isinstance(edge_index, SparseTensor):
            # Make dense, flip, make sparse again.
            dense_index = edge_index.to_dense()
            dense_index = torch.stack(list(dense_index.nonzero(as_tuple=True)), dim=1).permute(1, 0).long()
            flipped_edge = dense_index.flip([0])
            flipped_edge = to_dense_adj(flipped_edge).squeeze(0)
            flipped_edge = flipped_edge.to_sparse()
            flipped_edge = ToSparseTensor()(flipped_edge)                     
            x_out = self.conv_out(x, flipped_edge)
        else:
            x_out = self.conv_out(x, edge_index)
        ### Trying to fix the complaints of the original x_out ###

        out = self.alpha * x_out + (1 - self.alpha) * x_in
        

        if self.root_weight:
            out = out + self.lin(x)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.conv_in}, alpha={self.alpha})'


class DirGNN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = DirGNNConv(conv=GCNConv(input_dim, hidden_dim))
        self.conv2 = DirGNNConv(conv=GCNConv(input_dim, hidden_dim))


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 model: dict(help='backbone GNN model', choices=['gcn', 'sage', 'gat', 'dir', 'gt', 'gat2', 'graphconv']) = 'sage',
                 hidden_dim: dict(help='dimension of the hidden layers') = 16,
                 dropout: dict(help='dropout rate (between zero and one)') = 0.0,
                 x_steps: dict(help='KProp step parameter for features', option='-kx') = 0,
                 y_steps: dict(help='KProp step parameter for labels', option='-ky') = 0,
                 forward_correction: dict(help='applies forward loss correction', option='--forward') = True,
                 ):
        super().__init__()

        self.x_prop = KProp(steps=x_steps, aggregator='add', add_self_loops=False, normalize=True, cached=True)
        self.y_prop = KProp(steps=y_steps, aggregator='add', add_self_loops=False, normalize=True, cached=False,
                            transform=torch.nn.Softmax(dim=1))

        self.gnn = {'gcn': GCN, 'sage': GraphSAGE, 'gat': GAT, 'dir': DirGNN, 'gt': GraphTransformer, 'gat2': GAT2, 'graphconv': GraphConvNN}[model](
            input_dim=input_dim,
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.cached_yt = None
        self.forward_correction = forward_correction

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.x_prop(x, adj_t)
        x = self.gnn(x, adj_t)

        p_y_x = F.softmax(x, dim=1)  # P(y|x')
        p_yp_x = torch.matmul(p_y_x, data.T) if self.forward_correction else p_y_x  # P(y'|x')
        p_yt_x = self.y_prop(p_yp_x, data.adj_t)  # P(y~|x')

        return p_y_x, p_yp_x, p_yt_x

    def perturbed_forward(self, features, adj_t, kprop=False):
        if kprop:
            features = self.x_prop(features, adj_t)
        x = self.gnn(features, adj_t)

        return F.softmax(x, dim=1)

    def training_step(self, data):
        p_y_x, p_yp_x, p_yt_x = self(data)  # Passing test labels too?

        if self.cached_yt is None:
            # print('into if')
            yp = data.y.float()
            yp[data.test_mask] = 0  # to avoid using test labels
            self.cached_yt = self.y_prop(yp, data.adj_t)  # y~
            # print('training ',F.one_hot(self.cached_yt.argmax(dim=1)).shape, self.cached_yt.shape)

        loss = self.cross_entropy_loss(p_y=p_yt_x[data.train_mask], y=self.cached_yt[data.train_mask], weighted=False)

        metrics = {
            'train/loss': loss.item(),
            'train/acc': self.accuracy(pred=p_y_x[data.train_mask], target=data.y[data.train_mask]) * 100,
            'train/maxacc': data.T[0, 0].item() * 100,
        }

        return loss, metrics

    def validation_step(self, data):
        p_y_x, p_yp_x, p_yt_x = self(data)

        metrics = {
            'val/loss': self.cross_entropy_loss(p_yp_x[data.val_mask], data.y[data.val_mask]).item(),
            'val/acc': self.accuracy(pred=p_y_x[data.val_mask], target=data.y[data.val_mask]) * 100,
            'test/acc': self.accuracy(pred=p_y_x[data.test_mask], target=data.y[data.test_mask]) * 100,
        }

        return metrics

    @staticmethod
    def accuracy(pred, target):
        pred = pred.argmax(dim=1) if len(pred.size()) > 1 else pred
        target = target.argmax(dim=1) if len(target.size()) > 1 else target
        return accuracy_1d(pred=pred, target=target)

    @staticmethod
    def cross_entropy_loss(p_y, y, weighted=False):
        y_onehot = F.one_hot(y.argmax(dim=1), num_classes=y.shape[1])
        loss = -torch.log(p_y + 1e-20) * y_onehot
        loss *= y if weighted else 1
        loss = loss.sum(dim=1).mean()
        return loss
