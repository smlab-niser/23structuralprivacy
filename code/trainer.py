import sys
import torch
from torch.optim import SGD, Adam
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, auc
from datasets import get_edge_sets, compare_adjacency_matrices, generate_random_edge_sets

class Trainer:
    def __init__(
            self,
            optimizer: dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
            max_epochs: dict(help='maximum number of training epochs') = 100,
            learning_rate: dict(help='learning rate') = 0.01,
            weight_decay: dict(help='weight decay (L2 penalty)') = 0.0,
            patience: dict(help='early-stopping patience window size') = 0,
            device='cuda',
            logger=None,
    ):
        self.optimizer_name = optimizer
        self.max_epochs = max_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.logger = logger
        self.model = None

    def configure_optimizers(self):
        if self.optimizer_name == 'sgd':
            return SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(self, model, data):
        self.model = model.to(self.device)
        data = data.to(self.device)
        optimizer = self.configure_optimizers()

        num_epochs_without_improvement = 0
        best_metrics = None

        epoch_progbar = tqdm(range(1, self.max_epochs + 1), desc='Epoch: ', leave=False, position=1, file=sys.stdout)
        for epoch in epoch_progbar:
            metrics = {'epoch': epoch}
            train_metrics = self._train(data, optimizer)
            metrics.update(train_metrics)

            val_metrics = self._validation(data)
            metrics.update(val_metrics)

            if self.logger:
                self.logger.log(metrics)

            if best_metrics is None or (
                    metrics['val/loss'] < best_metrics['val/loss'] and
                    best_metrics['val/acc'] < metrics['val/acc'] <= metrics['train/maxacc'] and
                    best_metrics['train/acc'] < metrics['train/acc'] <= 1.05 * metrics['train/maxacc']
            ):
                best_metrics = metrics
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                if num_epochs_without_improvement >= self.patience > 0:
                    break

            # display metrics on progress bar
            epoch_progbar.set_postfix(metrics)

        if self.logger:
            self.logger.log_summary(best_metrics)

        return best_metrics

    def get_gradient(self, u, v, data):
        # compare prediction of model with perturbed/non-perturbed u
        # TODO: implement influence parameter. This is one of the values used in the paper,
        # TODO: but likely this could be tweaked.
        influence = 0.001
        features, adj = data.x, data.adj_t
        perturbation = torch.zeros_like(features)
        perturbation[v] = features[v] * influence
        # print(perturbation[v])
        grad = (self.model.perturbed_forward(features + perturbation, adj).detach() - \
               self.model.perturbed_forward(features, adj).detach()) / influence
        # print(grad)
        return grad[u]

    def attack(self, data, non_sp_data):
        # perform some comparisons on the two dataset

        # compare_adjacency_matrices(data, non_sp_data)

        norm_existing = []
        norm_non_existing = []

        existing_edges, non_existing_edges = get_edge_sets(data, random_order=True)
        non_sp_existing_edges, non_sp_non_existing_edges = get_edge_sets(non_sp_data, random_order=True)

        # delimiter = 500
        # # comparing elements
        # list1 = existing_edges[:delimiter]
        # list2 = non_sp_existing_edges[:delimiter]
        # list1 = [tuple(x) for x in list1.tolist()]
        # list2 = [tuple(x) for x in list2.tolist()]
        # elem_intersection = set(list1).intersection(set(list2))
        # print(f"Elements in original: {len(list1)}\nElements in perturbed: {len(list2)}\nElements in common: {len(elem_intersection)}")
        #
        # print("--------------------------------------")
        #
        # delimiter = -1
        # # comparing elements
        # list1 = existing_edges[:delimiter]
        # list2 = non_sp_existing_edges[:delimiter]
        # list1 = [tuple(x) for x in list1.tolist()]
        # list2 = [tuple(x) for x in list2.tolist()]
        # elem_intersection = set(list1).intersection(set(list2))
        # print(f"Elements in original: {len(list1)}\nElements in perturbed: {len(list2)}\nElements in common: {len(elem_intersection)}")
        #
        # print("--------------------------------------")
        # print("--------------------------------------")
        #
        # random_existing_edges, random_non_existing_edges = generate_random_edge_sets(non_sp_data, perc_ones=0.0014)
        #
        # delimiter = -1
        # # comparing elements
        # list1 = existing_edges[:delimiter]
        # list2 = random_existing_edges[:delimiter]
        # list1 = [tuple(x) for x in list1.tolist()]
        # list2 = [tuple(x) for x in list2.tolist()]
        # elem_intersection = set(list1).intersection(set(list2))
        # print("Random edges.")
        # print(f"Elements in original: {len(list1)}\nElements in perturbed: {len(list2)}\nElements in common: {len(elem_intersection)}")
        # print("--------------------------------------")
        # print("--------------------------------------")

        # prediction on perturbed data
        with torch.no_grad():
            for u, v in tqdm(existing_edges[:500]):
                grad = self.get_gradient(u, v, data)
                norm_existing.append(grad.norm().item())
            for u, v in tqdm(non_existing_edges[:500]):
                grad = self.get_gradient(u, v, data)
                norm_non_existing.append(grad.norm().item())

        y = [1] * len(norm_existing) + [0] * len(norm_non_existing)
        pred = norm_existing + norm_non_existing

        fpr, tpr, thresholds = roc_curve(y, pred)
        print()
        perturbed_auc = auc(fpr, tpr)
        print('Perturbed auc =', perturbed_auc)
        print()

        # prediction wrt original data
        norm_existing = []
        norm_non_existing = []
        with torch.no_grad():
            for u, v in tqdm(non_sp_existing_edges[:500]):
                grad = self.get_gradient(u, v, data)
                norm_existing.append(grad.norm().item())
            for u, v in tqdm(non_sp_non_existing_edges[:500]):
                grad = self.get_gradient(u, v, data)
                norm_non_existing.append(grad.norm().item())

        y = [1] * len(norm_existing) + [0] * len(norm_non_existing)
        pred = norm_existing + norm_non_existing

        fpr, tpr, thresholds = roc_curve(y, pred)
        print()
        wrt_original_auc = auc(fpr, tpr)
        print('Wrt original auc =', wrt_original_auc)
        print()


        # prediction wrt random data
        # norm_existing = []
        # norm_non_existing = []
        # with torch.no_grad():
        #     for u, v in tqdm(random_existing_edges[:500]):
        #         grad = self.get_gradient(u, v, data)
        #         norm_existing.append(grad.norm().item())
        #     for u, v in tqdm(random_non_existing_edges[:500]):
        #         grad = self.get_gradient(u, v, data)
        #         norm_non_existing.append(grad.norm().item())
        #
        # y = [1] * len(norm_existing) + [0] * len(norm_non_existing)
        # pred = norm_existing + norm_non_existing
        #
        # fpr, tpr, thresholds = roc_curve(y, pred)
        # print()
        # wrt_random_auc = auc(fpr, tpr)
        # print('Wrt random auc =', wrt_random_auc)
        # print()

        return {"perturbed_auc": perturbed_auc, "original_auc": wrt_original_auc}


    def _train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        loss, metrics = self.model.training_step(data)
        loss.backward()
        optimizer.step()
        return metrics

    @torch.no_grad()
    def _validation(self, data):
        self.model.eval()
        return self.model.validation_step(data)
