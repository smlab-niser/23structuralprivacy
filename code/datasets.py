import os
from functools import partial
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToSparseTensor, RandomNodeSplit
from torch_geometric.utils import to_undirected

from transforms import Normalize, FilterTopClass


class KarateClub(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'twitch',
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['edges.csv', 'features.csv', 'target.csv']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for part in ['edges', 'features', 'target']:
            download_url(f'{self.url}/{self.name}/{part}.csv', self.raw_dir)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)['target']
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(x.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'KarateClub-{self.name}()'


supported_datasets = {
    'cora': partial(Planetoid, name='cora'),
    'citeseer': partial(Planetoid, name='citeseer'),
    'pubmed': partial(Planetoid, name='pubmed'),
    'facebook': partial(KarateClub, name='facebook'),
    'lastfm': partial(KarateClub, name='lastfm', transform=FilterTopClass(10)),
}


def load_dataset(
        dataset: dict(help='name of the dataset', option='-d', choices=supported_datasets) = 'cora',
        data_dir: dict(help='directory to store the dataset') = './datasets',
        data_range: dict(help='min and max feature value', nargs=2, type=float) = (0, 1),
        val_ratio: dict(help='fraction of nodes used for validation') = .25,
        test_ratio: dict(help='fraction of nodes used for test') = .25,
):
    data = supported_datasets[dataset](root=os.path.join(data_dir, dataset))
    data = RandomNodeSplit(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data[0])
    data = ToSparseTensor()(data)
    data.name = dataset
    data.num_classes = int(data.y.max().item()) + 1

    if data_range is not None:
        low, high = data_range
        data = Normalize(low, high)(data)

    return data


def get_edge_sets(data, random_order=False):
    # Takes a pyg data.Data dataset, computes and return
    # existing_edges = [(idx, idx),...], non_existing_edges = [(idx, idx),...].

    dense_adj = data.adj_t.to_dense()
    existing_edges = dense_adj.nonzero()
    non_existing_edges = (dense_adj == 0).nonzero()

    if random_order:
        existing_edges = existing_edges[torch.randperm(existing_edges.size()[0])]
        non_existing_edges = non_existing_edges[torch.randperm(existing_edges.size()[0])]

    return existing_edges, non_existing_edges


def generate_random_edge_sets(data, perc_ones=0.1):
    dense_adj = data.adj_t.to_dense()
    dense_adj = (torch.rand(size=dense_adj.size()) < perc_ones).int()
    existing_edges = dense_adj.nonzero()
    non_existing_edges = (dense_adj == 0).nonzero()

    return existing_edges, non_existing_edges


def compare_adjacency_matrices(data, non_sp_data):
    dense = data.adj_t.to_dense()
    non_sp_dense = non_sp_data.adj_t.to_dense()
    # print(dense)
    diff = int(torch.sum(torch.abs(dense - non_sp_dense)))
    print(f"Comparing datasets: the two adjacency matrices have {diff}/{torch.numel(dense)} different entries.")

    print("Number of edges:")
    print(f"Perturbed: {int(torch.sum(dense))} edges, {int(torch.numel(dense) - torch.sum(dense))} non-edges")
    print(
        f"Original: {int(torch.sum(non_sp_dense))} edges, {int(torch.numel(non_sp_dense) - torch.sum(non_sp_dense))} non-edges")

    # getting edge lists from data
    existing_edges, non_existing_edges = get_edge_sets(data)
    non_sp_existing_edges, non_sp_non_existing_edges = get_edge_sets(non_sp_data)

    # computing list intersections
    l1 = existing_edges.tolist()
    l2 = non_sp_existing_edges.tolist()
    common_edges = len([list(x) for x in set(tuple(x) for x in l1).intersection(set(tuple(x) for x in l2))])
    l1 = non_existing_edges.tolist()
    l2 = non_sp_non_existing_edges.tolist()
    common_non_edges = len([list(x) for x in set(tuple(x) for x in l1).intersection(set(tuple(x) for x in l2))])

    print(f"Common edges: {common_edges}")
    print(f"Common non-edges: {common_non_edges}")
