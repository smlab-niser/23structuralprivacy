from datetime import date
import numpy as np
import torch
import math
from torch import optim
from torch._C import dtype
import torch.nn.functional as F
from torch_geometric.utils import subgraph, to_dense_adj, remove_self_loops
from torch_sparse import to_torch_sparse, SparseTensor
from torch_geometric.transforms import ToSparseTensor
from mechanisms import supported_feature_mechanisms, RandomizedResopnse
from models import KProp
import time


class FeatureTransform:
    supported_features = ['raw', 'rnd', 'one', 'ohd']

    def __init__(self, feature: dict(help='feature transformation method',
                                     choices=supported_features, option='-f') = 'raw'):

        self.feature = feature

    def __call__(self, data):

        if self.feature == 'rnd':
            data.x = torch.rand_like(data.x)
        elif self.feature == 'ohd':
            data = OneHotDegree(max_degree=data.num_features - 1)(data)
        elif self.feature == 'one':
            data.x = torch.ones_like(data.x)

        return data


class FeaturePerturbation:
    def __init__(self,
                 mechanism:     dict(help='feature perturbation mechanism', choices=list(supported_feature_mechanisms),
                                     option='-m') = 'mbm',
                 x_eps:         dict(help='privacy budget for feature perturbation', type=float,
                                     option='-ex') = np.inf,
                 data_range=None):

        self.mechanism = mechanism
        self.input_range = data_range
        self.x_eps = x_eps

    def __call__(self, data):
        if np.isinf(self.x_eps):
            return data

        if self.input_range is None:
            self.input_range = data.x.min().item(), data.x.max().item()

        data.x = supported_feature_mechanisms[self.mechanism](
            eps=self.x_eps,
            input_range=self.input_range
        )(data.x)

        return data


class LabelPerturbation:
    def __init__(self,
                 y_eps: dict(help='privacy budget for label perturbation',
                             type=float, option='-ey') = np.inf):
        self.y_eps = y_eps

    def __call__(self, data):
        data.y = F.one_hot(data.y, num_classes=data.num_classes)
        p_ii = 1  # probability of preserving the clean label i
        p_ij = 0  # probability of perturbing label i into another label j

        if not np.isinf(self.y_eps):
            mechanism = RandomizedResopnse(eps=self.y_eps, d=data.num_classes)
            perturb_mask = data.train_mask | data.val_mask
            y_perturbed = mechanism(data.y[perturb_mask])
            data.y[perturb_mask] = y_perturbed
            p_ii, p_ij = mechanism.p, mechanism.q

        # set label transistion matrix
        data.T = torch.ones(data.num_classes, data.num_classes, device=data.y.device) * p_ij
        data.T.fill_diagonal_(p_ii)


        return data


class OneHotDegree:
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, data):
        degree = data.adj_t.sum(dim=0).long()
        degree.clamp_(max=self.max_degree)
        data.x = F.one_hot(degree, num_classes=self.max_degree + 1).float()  # add 1 for zero degree
        return data


class Normalize:
    def __init__(self, low, high):
        self.min = low
        self.max = high

    def __call__(self, data):
        alpha = data.x.min(dim=0)[0]
        beta = data.x.max(dim=0)[0]
        delta = beta - alpha
        data.x = (data.x - alpha) * (self.max - self.min) / delta + self.min
        data.x = data.x[:, torch.nonzero(delta, as_tuple=False).squeeze()]  # remove features with delta = 0
        return data


class FilterTopClass:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        y = torch.nn.functional.one_hot(data.y)
        c = y.sum(dim=0).sort(descending=True)
        y = y[:, c.indices[:self.num_classes]]
        idx = y.sum(dim=1).bool()

        data.x = data.x[idx]
        data.y = y[idx].argmax(dim=1)
        data.num_nodes = data.y.size(0)

        if 'adj_t' in data:
            data.adj_t = data.adj_t[idx, idx]
        elif 'edge_index' in data:
            data.edge_index, data.edge_attr = subgraph(idx, data.edge_index, data.edge_attr, relabel_nodes=True)

        if 'train_mask' in data:
            data.train_mask = data.train_mask[idx]
            data.val_mask = data.val_mask[idx]
            data.test_mask = data.test_mask[idx]
        return data


class PrivatizeStructure:
    def __init__(self,  e_eps: dict(help='privacy budget for structure perturbation', type=float,
                                     option='-ee') = np.inf,
                        alpha: dict(help='node feature similarity coefficient', type=float,
                                     option='-alpha') = 0.5,
                        delta: dict(help='similarity threshold', type=float,
                                     option='-delta') = 0,
                        similarity: dict(help='similarity measure', type=str,
                                     option='-similarity') = 'cosine',
                        pick_neighbor: dict(help='method to select noisy neighbors', type=str,
                                     option='-neigh') = 'rr'
                        ):

        self.eps = e_eps
        self.alpha = alpha
        self.delta = delta
        self.pick_neighbor = pick_neighbor

        if self.pick_neighbor=='k_rr':
            self.value_k = delta

        k:int = 1

        if similarity   == 'cosine':
            self.similarity = self.cosine
        elif similarity == 'l1':
            self.similarity = self.l1_norm

        self.k_prop = KProp(steps=k, aggregator='add', add_self_loops=False, normalize=True, cached=True)

    def cosine(self, a, b):
        cos = torch.nn.CosineSimilarity(dim=-1)
        return cos(a, b)

    def l1_norm(self, a, b):
        pdist = torch.nn.PairwiseDistance(p=1, dim=-1)
        return pdist(a, b)

    def calculateSimilarity(self, data):
        x = data.x
        xa = self.k_prop(data.x, data.adj_t)

        dense_adj = SparseTensor.fill_diag(data.adj_t, 1).to_dense()
        sim = torch.zeros_like(dense_adj)

        for node in range(dense_adj.shape[0]): # x
            neighbors = torch.nonzero(dense_adj[node]) # A, B, C
            x_node = x[node]
            xa_node = xa[node]
            for u in neighbors:
                x_u, xa_u = data.x[u], xa[u]
                x_sim = self.similarity(x_node, x_u)
                xa_sim = self.similarity(xa_node, xa_u)
                s = self.alpha*x_sim + (1-self.alpha)*xa_sim

                if s > self.delta:
                    sim[node, u] = s.item()

        return sim


    def select_top(self, M):
        '''
        returns the top ranking element of the actual neighbor itself using RR
        '''
        d = 2
        rr = RandomizedResopnse(eps=self.eps, d=d)
        one_hot_M = F.one_hot(torch.tensor(1), num_classes=d)
        idx = torch.nonzero(rr(one_hot_M)).item()
        if idx == 0:
            return M[0]
        elif idx == 1:
            return M[-1]


    def select_with_rr(self, M):
        '''
        returns actual neighbor or any of the possible candidates using RR
        '''
        d = M.shape[0]
        rr = RandomizedResopnse(eps=self.eps, d=d)
        one_hot_M = F.one_hot(torch.tensor(d-1), num_classes=d)
        return M[torch.nonzero(rr(one_hot_M)).item()]

    def querySimilar(self, data):
        if self.pick_neighbor=='k_rr':
            self.delta = 0.0
        sim = self.calculateSimilarity(data)

        dense_adj = data.adj_t.to_dense()

        nodepairs = dense_adj.nonzero()

        for node in range(sim.shape[0]): # x
            neighbors = nodepairs[nodepairs[:, 0]==node, 1] # A, B, C
            for neighbor in neighbors:
                neighs_of_neighbor = nodepairs[nodepairs[:, 0]==neighbor, 1]
                # print(neighbor)
                if neighs_of_neighbor.size(dim=0) > 1:
                    nn = neighs_of_neighbor[neighs_of_neighbor==node]
                    if len(nn) > 0:
                        neighs_of_neighbor = torch.cat((neighs_of_neighbor[:neighs_of_neighbor[neighs_of_neighbor==node]], neighs_of_neighbor[neighs_of_neighbor[neighs_of_neighbor==node]+1:]))

                    if self.pick_neighbor == 'top':
                        if neighs_of_neighbor.shape[0] - torch.count_nonzero(sim[neighbor, neighs_of_neighbor]) == neighs_of_neighbor.shape[0]:
                            replacement = neighbor
                        else:
                            n = neighs_of_neighbor[torch.argmax(sim[neighbor, neighs_of_neighbor])]
                            replacement = self.select_top(torch.stack((n, neighbor)))
                    elif self.pick_neighbor == 'rr':
                        n_list = []
                        for n in torch.nonzero(sim[neighbor, neighs_of_neighbor]):
                            n_list.append(neighs_of_neighbor[n].item())

                        neighs_of_neighbor = torch.tensor(n_list).to(neighbor.device)
                        neighs_of_neighbor = torch.cat((neighs_of_neighbor, neighbor.unsqueeze(dim=0)))
                        replacement = self.select_with_rr(neighs_of_neighbor)
                    elif self.pick_neighbor =='k_rr':
                        if self.value_k:
                            _, idx_k = torch.topk(sim[neighbor, neighs_of_neighbor], math.ceil(self.value_k * neighs_of_neighbor.shape[0]))
                            n = neighs_of_neighbor[idx_k]
                        else:
                            _, idx_k = torch.topk(sim[neighbor, neighs_of_neighbor], 1)
                            n = neighs_of_neighbor[idx_k]

                        n = torch.cat((n, neighbor.unsqueeze(dim=0)))
                        replacement = self.select_with_rr(n)
                    dense_adj[node, neighbor] = 0
                    dense_adj[node, int(replacement)] = 1
        return dense_adj


    def randomNeighbors(self, data):
        dense_adj = data.adj_t.to_dense()
        nodepairs = dense_adj.nonzero()
        num_nodes = dense_adj.shape[0]
        for node in range(num_nodes):
            neighbors = nodepairs[nodepairs[:, 0]==node, 1]
            random_neighbors = torch.randint(0, num_nodes, neighbors.shape)
            dense_adj[node, neighbors] = 0
            dense_adj[node, random_neighbors] = 1
        return dense_adj

    def __call__(self, data):

        if self.eps != np.inf:
            pert_adj = self.querySimilar(data)

            data.edge_index = torch.stack(list(pert_adj.nonzero(as_tuple=True)), dim=1).permute(1, 0).long()
            data = ToSparseTensor()(data)

        return data


class TwoHopRRBaseline:

    def __init__(self,  e_eps: dict(help='privacy budget for structure perturbation',
                                    type=float, option='-ee') = np.inf):
        self.eps = e_eps
    def localRandomizedResponse(self, data):
        dense_adj = data.adj_t.to_dense()
        nodepairs = dense_adj.nonzero()

        rr = RandomizedResopnse(eps=self.eps, d=2)
        number_added_edges = 0
        number_removed_edges = 0

        print("LOCAL RAND")
        for node in range(dense_adj.shape[0]):  # x
            # break
            neighbors = nodepairs[nodepairs[:, 0] == node, 1]  # A, B, C
            non_neighbors = []
            for neighbor in neighbors:
                neighs_of_neighbor = nodepairs[nodepairs[:, 0] == neighbor, 1]
                if neighs_of_neighbor.size(dim=0) != 1:
                    nn = neighs_of_neighbor[neighs_of_neighbor==node]
                    if len(nn) > 0:
                        neighs_of_neighbor = torch.cat((neighs_of_neighbor[:neighs_of_neighbor[neighs_of_neighbor==node]], neighs_of_neighbor[neighs_of_neighbor[neighs_of_neighbor==node]+1:]))
                # Remove neighbors from neigh of neighbors, to get non_neighbors. Move to cpu (if necessary) and back.
                non_neighbors.append(torch.from_numpy(np.setdiff1d(neighs_of_neighbor.cpu().numpy(),
                                                                   neighbors.cpu().numpy())).to(dense_adj.device))

            if len(non_neighbors) > 0:
                # break
                # Getting all neighbors and non-neighbors and labelling them.
                non_neighbors = torch.cat(non_neighbors, dim=0).unique()
                neigh_flag = torch.ones(neighbors.size()[0], dtype=torch.int)
                non_neigh_flag = torch.zeros(non_neighbors.size()[0], dtype=torch.int)
                neighbor_list = torch.cat((neigh_flag, non_neigh_flag), dim=0)

                # Performing edge flips and updating adjacency matrix.
                flipped_neighbors = rr.perform_binary_flip(neighbor_list)
                # neighdiff = len(neighbor_list) - (flipped_neighbors == neighbor_list).sum()
                candidates = torch.cat((neighbors, non_neighbors), dim=0)
                for i in range(len(candidates)):
                    current = dense_adj[node, candidates[i]]
                    next = flipped_neighbors[i]
                    # dense_adj[node, candidates[i]] = flipped_neighbors[i]
                    # dense_adj[node, candidates[i]] = next.item()
                    if current != next:
                        if current == 0:
                            number_added_edges = number_added_edges + 1
                        else:
                            number_removed_edges = number_removed_edges + 1
                    dense_adj[node, candidates[i]] = next

            elif len(neighbors) > 0:
                # Performing RR on neighbors only.
                neighbor_list = torch.ones(neighbors.size()[0], dtype=torch.int)
                flipped_neighbors = rr.perform_binary_flip(neighbor_list)
                for i in range(len(neighbors)):
                    dense_adj[node, neighbors[i]] = flipped_neighbors[i]
            else:
                pass  # Isolated node, so do nothing.

        print(f"removed: {number_removed_edges}")
        print(f"added: {number_added_edges}")
        return dense_adj

    def __call__(self, data):

        # old_dense = data.adj_t.to_dense()

        if self.eps != np.inf:
            pert_adj = self.localRandomizedResponse(data)
            data.edge_index = torch.stack(list(pert_adj.nonzero(as_tuple=True)), dim=1).permute(1, 0).long()
            data = ToSparseTensor()(data)

            # non_changed_edges = (old_dense == pert_adj).sum()
            # total_edges = old_dense.size()[0]*old_dense.size()[1]

            # diff = total_edges-non_changed_edges
            # perc_diff = diff/total_edges

        return data


if __name__ == "__main__":
    print('hello')