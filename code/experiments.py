from cmath import inf
import os.path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product
from collections.abc import Iterable
from pyexpat import features

from random import sample
import numpy as np
import pandas as pd
from utils import print_args, JobManager


class HyperParams:
    def __init__(self, path_dir):
        try:
            self.df_lwd = pd.read_csv(
                os.path.join(path_dir, 'lwd.csv'), index_col=['dataset', 'feature', 'x_eps', 'y_eps']
            )
        except FileNotFoundError:
            self.df_lwd = None

        try:
            self.df_steps = pd.read_csv(os.path.join(path_dir, 'steps.csv'), index_col=['dataset', 'x_eps', 'y_eps'])
        except FileNotFoundError:
            self.df_steps = None

    def get(self, dataset, feature, x_eps, y_eps):
        hparams = self.get_lwd(dataset=dataset, feature=feature, x_eps=x_eps, y_eps=y_eps)
        hparams.update(self.get_steps(dataset=dataset, x_eps=x_eps, y_eps=y_eps))
        return hparams

    def get_lwd(self, dataset, feature, x_eps, y_eps):
        params = {}
        if self.df_lwd is not None:
            if feature == '1rnd': feature = 'rnd'
            x_eps = np.inf if np.isinf(x_eps) else 1
            y_eps = np.inf if np.isinf(y_eps) else 1
            params = self.df_lwd.loc[dataset, feature, x_eps, y_eps].to_dict()

        return params

    def get_steps(self, dataset, x_eps, y_eps):
        params = {}
        if self.df_steps is not None:
            params = self.df_steps.loc[dataset, x_eps, y_eps].to_dict()

        return params


class CommandBuilder:
    BEST_VALUE = None

    def __init__(self, args, hparams_dir=None, random=None):
        self.random = random
        self.default_options = f" -s {args.seed} -r {args.repeats} -o {args.output_dir} --device {args.device}"
        if args.project:
            self.default_options += f" --log --log-mode collective --project-name {args.project} "
        self.hparams = HyperParams(path_dir=hparams_dir) if hparams_dir else None

    def build(self, dataset, feature, mechanism, model, x_eps, y_eps, e_eps, alpha, delta, similarity, pick_neighbor, forward_correction, x_steps, y_steps, learning_rate, weight_decay, dropout,
              attack=False):

        cmd_list = []
        configs = self.product_dict(
            dataset=self.get_list(dataset),
            feature=self.get_list(feature),
            mechanism=self.get_list(mechanism),
            model=self.get_list(model),
            x_eps=self.get_list(x_eps),
            y_eps=self.get_list(y_eps),
            e_eps=self.get_list(e_eps),
            alpha=self.get_list(alpha),
            delta=self.get_list(delta),
            similarity=self.get_list(similarity),
            pick_neighbor=self.get_list(pick_neighbor),
            forward_correction=self.get_list(forward_correction),
            x_steps=self.get_list(x_steps),
            y_steps=self.get_list(y_steps),
            learning_rate=self.get_list(learning_rate),
            weight_decay=self.get_list(weight_decay),
            dropout=self.get_list(dropout),
            attack=self.get_list(attack),
        )

        if self.random:
            configs = sample(list(configs), self.random)

        for config in configs:
            config = self.fill_best_params(config)
            options = ' '.join([f' --{param} {value} ' for param, value in config.items()])
            command = f'python main.py {options} {self.default_options}'
            cmd_list.append(command)

        return cmd_list

    def fill_best_params(self, config):
        if self.hparams:
            best_params = self.hparams.get(
                dataset=config['dataset'],
                feature=config['feature'],
                x_eps=config['x_eps'],
                y_eps=config['y_eps'],
            )

            for param, value in config.items():
                if value == self.BEST_VALUE:
                    config[param] = best_params[param]
        return config

    @staticmethod
    def get_list(param):
        if not isinstance(param, Iterable) or isinstance(param, str):
            param = [param]
        return param

    @staticmethod
    def product_dict(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))


def experiments(args):
    run_cmds = []
    cmdbuilder = CommandBuilder(args=args, hparams_dir='./hparams')
    # datasets = ['cora', 'pubmed', 'lastfm', 'facebook']
    datasets = ['facebook']
    
    # best steps from LPGNN
    steps ={'cora':     [16, 2],
            'pubmed':   [16, 0],
            'lastfm':   [16, 0],
            'facebook': [4, 2]}

    for dataset in datasets:
        run_cmds += cmdbuilder.build(
            dataset=dataset,
            feature='raw',
            mechanism='mbm',
            model=['gcn', 'gat', 'sage'],
            x_eps=[3],
            x_steps=steps[dataset][0],
            y_eps=[3],
            y_steps=steps[dataset][1],
            e_eps=[0.1, 1, 2, 8, np.inf],
            alpha=[0, 0.25, 0.5, 0.75, 1],
            delta=[0, 0.1, 0.25, 0.5, 1.0], 
            similarity=['cosine'], 
            pick_neighbor=['rr', 'top'],
            forward_correction=True,
            learning_rate=CommandBuilder.BEST_VALUE,
            weight_decay=CommandBuilder.BEST_VALUE,
            dropout=CommandBuilder.BEST_VALUE
        )

    run_cmds = list(set(run_cmds))  # remove duplicate runs
    return run_cmds

def top_k_experiments(args):
    run_cmds = []
    cmdbuilder = CommandBuilder(args=args, hparams_dir='./hparams')
    # datasets = ['cora', 'pubmed', 'lastfm', 'facebook']
    datasets = ['facebook']

    # best steps from LPGNN
    steps ={'cora':     [16, 2],
            'pubmed':   [16, 0],
            'lastfm':   [16, 0],
            'facebook': [4, 2]}
    
    for dataset in datasets:
        run_cmds += cmdbuilder.build(
            dataset=dataset,
            feature='raw',
            mechanism='mbm',
            model=['gcn', 'gat', 'sage'],
            x_eps=[3],
            x_steps=steps[dataset][0],
            y_eps=[3],
            y_steps=steps[dataset][1],
            e_eps=[0.1, 1, 2, 8, np.inf],
            alpha=[0, 0.25, 0.5, 0.75, 1],
            delta=[0, 0.25, 0.5, 0.75, 1.0], # delta = percentage of degree to consider as candidates for replacement
            similarity=['cosine'], 
            pick_neighbor=['k_rr'],
            forward_correction=True,
            learning_rate=CommandBuilder.BEST_VALUE,
            weight_decay=CommandBuilder.BEST_VALUE,
            dropout=CommandBuilder.BEST_VALUE
        )

    run_cmds = list(set(run_cmds))  # remove duplicate runs
    return run_cmds


def attack_experiments(args):
    run_cmds = []
    cmdbuilder = CommandBuilder(args=args, hparams_dir='./hparams')
    # datasets = ['cora', 'pubmed', 'lastfm', 'facebook']
    datasets = ['cora', 'pubmed', 'lastfm']

    # best steps from LPGNN
    steps ={'cora':     [16, 2],
            'pubmed':   [16, 0],
            'lastfm':   [16, 0]}
            # 'facebook': [4, 2]}

    for dataset in datasets:
        run_cmds += cmdbuilder.build(
            attack=True,
            dataset=dataset,
            feature='raw',
            mechanism='mbm',
            # model=['gcn', 'gat', 'sage'],
            # model=['gt', 'gcn2', 'gine'],
            model=['gt'],
            x_eps=[3],
            x_steps=steps[dataset][0],
            y_eps=[3],
            y_steps=steps[dataset][1],
            e_eps=[0.1, 1.0, 2.0, 8.0, np.inf],
            alpha=[0, 0.25, 0.5, 0.75, 1],
            delta=[0, 0.25, 0.5, 0.75, 1],
            similarity=['cosine'],
            pick_neighbor=['top'],
            forward_correction=True,
            learning_rate=CommandBuilder.BEST_VALUE,
            weight_decay=CommandBuilder.BEST_VALUE,
            dropout=CommandBuilder.BEST_VALUE
        )

    run_cmds = list(set(run_cmds))  # remove duplicate runs
    return run_cmds


def experiment_generator(args):
    run_cmds = []

    if args.threshold_based:
        run_cmds += experiments(args)

    if args.top_k_based:
        run_cmds += top_k_experiments(args)

    if args.attack:
        run_cmds += attack_experiments(args)

    return run_cmds


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser, parser_create = JobManager.register_arguments(parser)
    parser.add_argument('-o', '--output-dir', type=str, default='./results', help="directory to store the results")
    parser_create.add_argument('--device', help='desired device for training', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], default='cuda:0')
    parser_create.add_argument('--project', type=str, help='project name for wandb logging (omit to disable)')
    parser_create.add_argument('-s', '--seed', type=int, default=12345, help='initial random seed')
    parser_create.add_argument('-r', '--repeats', type=int, default=10, help="number of experiment iterations")
    parser_create.add_argument('--top_k_based', action='store_true')
    parser_create.add_argument('--threshold_based', action='store_true')
    parser_create.add_argument('--attack', action='store_true')
    args = parser.parse_args()
    print_args(args)

    JobManager(args, cmd_generator=experiment_generator).run()


if __name__ == '__main__':
    main()
