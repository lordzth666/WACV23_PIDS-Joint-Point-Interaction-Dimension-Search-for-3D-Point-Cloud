from re import S
from typing import (
    List,
    Optional
)
import numpy as np
import torch
import copy

from tqdm import tqdm

from nasflow.dataset.nas_dataset import NASDataSet
from nasflow.io_utils.base_parser import parse_args_from_kwargs
from nasflow.optim.optimizer import get_optimizer_with_lr
from nasflow.optim.scheduler import get_lr_scheduler
from nasflow.losses.l2_loss import get_l2_loss
from nasflow.algo.surrogate.nn_predictor_utils import (
    NNModelBase,
    NNBackBone,
    NNBackBoneEmbedding,
    NNBackBoneDense,
    EMANNModelBase,
    init_weights,
)
from nasflow.losses.loss_factory import get_loss_fn_from_lib

from nasflow.algo.surrogate.dense_sparse_predictor_utils import (
    DenseSparseNN,
)

class BasePredictor:
    """
    An virtual predictor instance to wrap a surrogate predictive model for NAS.
    """
    def __init__(
            self,
            dataset: NASDataSet,
            **kwargs):
        """
        Args:
            dataset: The NAS dataset to provide data.
            in_dims: Input dimension for the predictor.
        """
        self.dataset = dataset
        self.core_ml_arch = None
        self.kwargs = kwargs

    def fit(self):
        raise BaseException("This is a base class!")

class BaseNNPredictor(BasePredictor):
    def __init__(self,
                 dataset: NASDataSet,
                 **kwargs):
        super(BaseNNPredictor, self).__init__(dataset, **kwargs)
        # Parse ther args.
        self.learning_rate = parse_args_from_kwargs(
            kwargs, "learning_rate", 0.01)
        self.optimizer = parse_args_from_kwargs(kwargs, "optimizer", "adam")
        self.weight_decay = parse_args_from_kwargs(
            kwargs, "weight_decay", 1e-4)
        self.num_epochs = parse_args_from_kwargs(kwargs, 'num_epochs', 90)
        self.batch_size = parse_args_from_kwargs(kwargs, 'batch_size', 64)
        self.drop_last_batch = parse_args_from_kwargs(
            kwargs, 'drop_last_batch', True)
        self.units = parse_args_from_kwargs(kwargs, 'units', [100, ])
        self.activation = parse_args_from_kwargs(kwargs, 'activation', "relu")
        self.lr_scheduler = parse_args_from_kwargs(
            kwargs, 'lr_scheduler', 'cosine-pred')
        self.ema_decay = parse_args_from_kwargs(
            kwargs, 'ema_decay', 0.0
        )
        # Whether to use gpu or not.
        self.use_gpu = parse_args_from_kwargs(kwargs, "use_gpu", False)
        # Define NN Regressor.
        self.core_ml_arch = NNModelBase()
        self.steps = 0
        self.epochs = 0

    def _train_one_epoch(
            self,
            model: torch.nn.Module,
            epoch: int,
            optimizer: torch.optim.Optimizer,
            verbose: bool = True,
            **kwargs
        ):
        grad_clip_norm = parse_args_from_kwargs(kwargs, "grad_clip_norm", 100.0)
        train_losses = []
        dataset_iterator = self.dataset.iter_map_and_batch(
            split='train',
            shuffle=True,
            batch_size=self.batch_size,
            drop_last_batch=self.drop_last_batch)
        model.train()

        for _, batch in tqdm(enumerate(dataset_iterator)):
            inputs, labels = [x[0] for x in batch], [x[1] for x in batch]
            inputs, labels = torch.tensor(inputs), torch.tensor(labels)
            if self.use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = self.loss(outputs, labels)
            if self.ema_decay != 0:
                l2_loss = get_l2_loss(model.model, self.weight_decay)
            else:
                l2_loss = get_l2_loss(model, self.weight_decay)
            total_loss = l2_loss + loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            if self.ema_decay != 0:
                model.update()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        if verbose:
            print("Epoch [{:d}] Training Loss: {}".format(epoch, avg_train_loss))
        return avg_train_loss

    def _test_one_epoch(
            self,
            model: torch.nn.Module,
            epoch: int,
            verbose: bool = True
        ):
        dataset_iterator = self.dataset.iter_map_and_batch(
            split='test',
            shuffle=False,
            batch_size=self.batch_size,
            drop_last_batch=False)
        model.eval()
        with torch.no_grad():
            all_outputs = []
            all_labels = []
            for _, batch in enumerate(dataset_iterator):
                inputs, labels = [x[0] for x in batch], [x[1] for x in batch]
                inputs, labels = torch.tensor(inputs), torch.tensor(labels)
                if self.use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                all_outputs.append(outputs.flatten())
                all_labels.append(labels)

            all_outputs = torch.cat(all_outputs, 0)
            all_labels = torch.cat(all_labels, 0)
            avg_test_loss = self.loss(all_outputs, all_labels).item()
        if verbose: print("Epoch [{:d}] Testing Loss: {}".format(epoch, avg_test_loss))
        return avg_test_loss

    def fit(self, verbose: bool = True, **kwargs):
        print("Fitting with the following hyperparameters ...")
        print(vars(self))
        optimizer = get_optimizer_with_lr(self.core_ml_arch, self.learning_rate)
        scheduler = get_lr_scheduler(optimizer, self.lr_scheduler, self.num_epochs)
        early_stopping = parse_args_from_kwargs(kwargs, 'early_stopping', False)
        patience = parse_args_from_kwargs(kwargs, 'patience', 10)
        init_epochs_no_stopping = parse_args_from_kwargs(kwargs, 'init_epochs_no_stopping', 20)
        last_test_loss = 9999999
        cur_patience = 0
        # best_model = None
        for epoch in range(self.num_epochs):
            self._train_one_epoch(self.core_ml_arch, epoch, optimizer, verbose=verbose, **kwargs)
            test_loss = self._test_one_epoch(self.core_ml_arch, epoch, verbose=verbose)
            cur_patience = cur_patience + 1 if test_loss > last_test_loss else 0
            if cur_patience >= patience and epoch > init_epochs_no_stopping and early_stopping:
                print("Early stopping at epoch {}: Loss did not improve within {} epochs.".format(epoch, cur_patience))
                break
            #else:
            #    if test_loss < last_test_loss:
            #        last_test_loss = test_loss
            #        best_model = copy.deepcopy(self.core_ml_arch)
            scheduler.step()
            self.epochs += 1

        print("Done!")
        #self.core_ml_arch = copy.deepcopy(best_model)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor):
        if self.use_gpu:
            inputs = inputs.cuda()
        outputs = self.core_ml_arch(inputs)
        return outputs

    def load_weights(
            self,
            ckpt_path: Optional[str] = None,
            exclude_ckpt_keys: Optional[List[str]] = None):
        self.core_ml_arch.load_weights(ckpt_path, exclude_ckpt_keys)

    def save_weights(
            self,
            ckpt_path: Optional[str] = None
    ):
        self.core_ml_arch.save_weights(ckpt_path)

    def loss(self, outputs, labels):
        raise BaseException("This is a base class used for NN utilities.")

    def train(self):
        self.core_ml_arch.train()

    def eval(self):
        self.core_ml_arch.eval()

class NNPredictor(BaseNNPredictor):
    def __init__(self,
                 dataset: NASDataSet,
                 **kwargs):
        super(NNPredictor, self).__init__(dataset, **kwargs)
        self.in_dims = parse_args_from_kwargs(
            kwargs, 'in_dims', 100)
        self.loss_fn_name = parse_args_from_kwargs(
            kwargs, 'loss_fn_name', "mse-loss")
        self.ranking_loss_fn_name = parse_args_from_kwargs(
            kwargs, 'ranking_loss_fn_name', "margin-ranking-loss")
        self.ranking_loss_warmup_epochs = parse_args_from_kwargs(
            kwargs, 'ranking_loss_warmup_epochs', 75)
        self.margin = parse_args_from_kwargs(
            kwargs, 'margin', 0.0)
        self.ranking_loss_coef = parse_args_from_kwargs(
            kwargs, 'ranking_loss_coef', 1.0)
        self.ema_decay = parse_args_from_kwargs(
            kwargs, 'ema_decay', 0.9)
        self.core_ml_arch = NNBackBone(
            self.in_dims, self.units, self.activation)
        self.core_ml_arch.apply(init_weights)
        if self.ema_decay != 0.0:
            print("Wrapping model with EMA decay={:.5f}".format(self.ema_decay))
            self.core_ml_arch = EMANNModelBase(self.core_ml_arch, self.ema_decay)
        if self.use_gpu:
            self.core_ml_arch = self.core_ml_arch.cuda()
        self.loss_fn = get_loss_fn_from_lib(self.loss_fn_name)()
        self.ranking_loss_fn = get_loss_fn_from_lib(self.ranking_loss_fn_name)(self.margin)

    def loss(self, outputs, labels):
        normal_loss = self.loss_fn(outputs.flatten(), labels)
        if not self.core_ml_arch.training or self.epochs < self.ranking_loss_warmup_epochs:
            return normal_loss
        self.steps += 1
        bsize = outputs.size(0)
        outputs_flat = outputs.flatten()
        indx1 = np.random.choice(bsize, bsize, replace=False)
        indx2 = np.random.choice(bsize, bsize, replace=False)
        group_1, group_2 = outputs_flat[indx1], outputs_flat[indx2]
        rank_labels = ((labels[indx1] > labels[indx2]).float() - 0.5) * 2
        rank_loss = self.ranking_loss_fn(group_1, group_2, rank_labels)
        #print("Normal Loss: {}".format(normal_loss.item()))
        #print("Rank Loss: {}".format(rank_loss.item()))
        ranking_loss_warmup_coef = 0.0 if self.epochs < self.ranking_loss_warmup_epochs else 1
        return normal_loss + rank_loss * ranking_loss_warmup_coef * self.ranking_loss_coef

class NNPredictorWithEmbedding(BaseNNPredictor):
    def __init__(self,
                 dataset: NASDataSet,
                 margin: float = 0.0,
                 **kwargs):
        super(NNPredictorWithEmbedding, self).__init__(dataset, **kwargs)
        self.margin = margin
        self.in_dims = parse_args_from_kwargs(
            kwargs, 'in_dims', 100)
        self.embedding_num_inputs = parse_args_from_kwargs(
            kwargs, 'embedding_num_inputs', 10)
        self.embedding_table_size = parse_args_from_kwargs(
            kwargs, 'embedding_table_size', 10)
        self.embedding_dim = parse_args_from_kwargs(
            kwargs, 'embedding_dim', 8)
        self.loss_fn_name = parse_args_from_kwargs(
            kwargs, 'loss_fn_name', "mse-loss")
        self.ranking_loss_fn_name = parse_args_from_kwargs(
            kwargs, 'ranking_loss_fn_name', "margin-ranking-loss")
        self.ranking_loss_warmup_epochs = parse_args_from_kwargs(
            kwargs, 'ranking_loss_warmup_epochs', 25)
        self.margin = parse_args_from_kwargs(
            kwargs, 'margin', 0.0)
        self.ranking_loss_coef = parse_args_from_kwargs(
            kwargs, 'ranking_loss_coef', 1.0)
        self.ema_decay = parse_args_from_kwargs(
            kwargs, 'ema_decay', 0.9)
        self.core_ml_arch = NNBackBoneEmbedding(
            self.in_dims,
            self.units,
            self.activation,
            self.embedding_num_inputs,
            self.embedding_table_size,
            self.embedding_dim)
        self.core_ml_arch.apply(init_weights)
        if self.ema_decay != 0.0:
            print("Wrapping model with EMA decay={:.5f}".format(self.ema_decay))
            self.core_ml_arch = EMANNModelBase(self.core_ml_arch, self.ema_decay)
        if self.use_gpu:
            self.core_ml_arch = self.core_ml_arch.cuda()
        self.loss_fn = get_loss_fn_from_lib(self.loss_fn_name)()
        self.ranking_loss_fn = get_loss_fn_from_lib(self.ranking_loss_fn_name)(self.margin)

    def loss(self, outputs, labels):
        normal_loss = self.loss_fn(outputs.flatten(), labels)
        if not self.core_ml_arch.training or self.epochs < self.ranking_loss_warmup_epochs:
            use_rank_loss = 0
        else:
            use_rank_loss = 1
        if use_rank_loss == 0:
            return normal_loss
        else:
            bsize = outputs.size(0)
            outputs_flat = outputs.flatten()
            rank_sample_size = max(200, 4 * bsize)
            indx1 = np.random.choice(bsize, rank_sample_size, replace=True)
            indx2 = np.random.choice(bsize, rank_sample_size, replace=True)
            group_1, group_2 = outputs_flat[indx1], outputs_flat[indx2]
            rank_labels = ((labels[indx1] > labels[indx2]).float() - 0.5) * 2
            rank_loss = self.ranking_loss_fn(group_1, group_2, rank_labels)
            #print("Normal Loss: {}".format(normal_loss.item()))
            #print("Rank Loss: {}".format(rank_loss.item()))
            ranking_loss_warmup_coef = 0.0 if self.epochs < self.ranking_loss_warmup_epochs else 1.0
            return rank_loss * ranking_loss_warmup_coef * self.ranking_loss_coef + normal_loss

class NNPredictorDense(BaseNNPredictor):
    def __init__(self,
                 dataset: NASDataSet,
                 **kwargs):
        super(NNPredictorDense, self).__init__(dataset, **kwargs)
        self.in_dims = parse_args_from_kwargs(
            kwargs, 'in_dims', 100)
        self.loss_fn_name = parse_args_from_kwargs(
            kwargs, 'loss_fn_name', "mse-loss")
        self.ranking_loss_fn_name = parse_args_from_kwargs(
            kwargs, 'ranking_loss_fn_name', "margin-ranking-loss")
        self.ranking_loss_warmup_epochs = parse_args_from_kwargs(
            kwargs, 'ranking_loss_warmup_epochs', 25)
        self.margin = parse_args_from_kwargs(
            kwargs, 'margin', 0.0)
        self.ranking_loss_coef = parse_args_from_kwargs(
            kwargs, 'ranking_loss_coef', 1.0)
        self.ema_decay = parse_args_from_kwargs(
            kwargs, 'ema_decay', 0.9)
        self.use_sigmoid = parse_args_from_kwargs(
            kwargs, 'use_sigmoid', False)
        self.dropout = parse_args_from_kwargs(
            kwargs, 'dropout', 0.0)
        self.core_ml_arch = NNBackBoneDense(
            self.in_dims,
            self.units,
            self.activation,
            dropout=self.dropout,
            use_sigmoid=self.use_sigmoid)
        self.core_ml_arch.apply(init_weights)
        if self.ema_decay != 0.0:
            print("Wrapping model with EMA decay={:.5f}".format(self.ema_decay))
            self.core_ml_arch = EMANNModelBase(self.core_ml_arch, self.ema_decay)
        if self.use_gpu:
            self.core_ml_arch = self.core_ml_arch.cuda()
        self.loss_fn = get_loss_fn_from_lib(self.loss_fn_name)()
        self.ranking_loss_fn = get_loss_fn_from_lib(self.ranking_loss_fn_name)(self.margin)

    def loss(self, outputs, labels):
        normal_loss = self.loss_fn(outputs.flatten(), labels)
        if not self.core_ml_arch.training or self.epochs < self.ranking_loss_warmup_epochs:
            use_rank_loss = 0
        else:
            use_rank_loss = 1
        if use_rank_loss == 0:
            return normal_loss
        else:
            bsize = outputs.size(0)
            outputs_flat = outputs.flatten()
            rank_sample_size = max(200, 4 * bsize)
            indx1 = np.random.choice(bsize, rank_sample_size, replace=True)
            indx2 = np.random.choice(bsize, rank_sample_size, replace=True)
            group_1, group_2 = outputs_flat[indx1], outputs_flat[indx2]
            rank_labels = ((labels[indx1] > labels[indx2]).float() - 0.5) * 2
            rank_loss = self.ranking_loss_fn(group_1, group_2, rank_labels)
            #print("Normal Loss: {}".format(normal_loss.item()))
            #print("Rank Loss: {}".format(rank_loss.item()))
            ranking_loss_warmup_coef = 0.0 if self.epochs < self.ranking_loss_warmup_epochs else 1.0
            return rank_loss * ranking_loss_warmup_coef * self.ranking_loss_coef + normal_loss

class DenseSparseNNPredictor(BaseNNPredictor):
    def __init__(self,
                 dataset: NASDataSet,
                 **kwargs):
        super(DenseSparseNNPredictor, self).__init__(dataset, **kwargs)
        self.dense_nn_pos_units = parse_args_from_kwargs(
            kwargs, 'dense_nn_pos_units', [128, 64])
        self.dense_nn_arch_units = parse_args_from_kwargs(
            kwargs, 'dense_nn_arch_units', [128, 64])
        self.over_nn_units = parse_args_from_kwargs(
            kwargs, 'over_nn_units', [64, 16])
        self.activation = parse_args_from_kwargs(kwargs, 'activation', 'relu')
        self.num_dense_pos_features = int(parse_args_from_kwargs(
            kwargs, 'num_dense_pos_features', 14))
        self.num_dense_arch_features = int(parse_args_from_kwargs(
            kwargs, 'num_dense_arch_features', 14))
        self.num_sparse_features = int(parse_args_from_kwargs(
            kwargs, 'num_sparse_features', 7))
        self.num_pos_sparse_interact_outputs = parse_args_from_kwargs(
            kwargs, 'num_pos_sparse_interact_outputs', 50)
        self.num_arch_sparse_interact_outputs = parse_args_from_kwargs(
            kwargs, 'num_arch_sparse_interact_outputs', 50)
        self.embedding_table_size = int(parse_args_from_kwargs(
            kwargs, 'embedding_table_size', 3))
        self.embedding_dim = parse_args_from_kwargs(
            kwargs, 'embedding_dim', 16)
        self.dropout = parse_args_from_kwargs(
            kwargs, 'dropout', 0.0)
        self.margin = parse_args_from_kwargs(
            kwargs, 'margin', 0.0)
        self.loss_fn_name = parse_args_from_kwargs(
            kwargs, 'loss_fn_name', "mse-loss")
        self.ranking_loss_fn_name = parse_args_from_kwargs(
            kwargs, 'ranking_loss_fn_name', "margin-ranking-loss")
        self.ranking_loss_warmup_epochs = parse_args_from_kwargs(
            kwargs, 'ranking_loss_warmup_epochs', 25)
        self.margin = parse_args_from_kwargs(
            kwargs, 'margin', 0.0)
        self.ranking_loss_coef = parse_args_from_kwargs(
            kwargs, 'ranking_loss_coef', 1.0)
        self.ema_decay = parse_args_from_kwargs(
            kwargs, 'ema_decay', 0.9)
        self.core_ml_arch = DenseSparseNN(
            self.dense_nn_pos_units,
            self.dense_nn_arch_units,
            self.over_nn_units,
            self.activation,
            self.num_dense_pos_features,
            self.num_dense_arch_features,
            self.num_sparse_features,
            self.num_pos_sparse_interact_outputs,
            self.num_arch_sparse_interact_outputs,
            self.embedding_table_size,
            self.embedding_dim,
            self.dropout,
        )
        self.core_ml_arch.apply(init_weights)
        if self.ema_decay != 0.0:
            print("Wrapping model with EMA decay={:.5f}".format(self.ema_decay))
            self.core_ml_arch = EMANNModelBase(self.core_ml_arch, self.ema_decay)
        if self.use_gpu:
            self.core_ml_arch = self.core_ml_arch.cuda()
        self.loss_fn = get_loss_fn_from_lib(self.loss_fn_name)()
        self.ranking_loss_fn = get_loss_fn_from_lib(self.ranking_loss_fn_name)(self.margin)

    def loss(self, outputs, labels):
        normal_loss = self.loss_fn(outputs.flatten(), labels)
        if not self.core_ml_arch.training or self.epochs < self.ranking_loss_warmup_epochs \
            or self.ranking_loss_coef == 0.0:
            use_rank_loss = 0
        else:
            self.steps += 1
            use_rank_loss = 1 if self.steps % 2 == 0 else 0
        if use_rank_loss == 0:
            return normal_loss
        else:
            bsize = outputs.size(0)
            outputs_flat = outputs.flatten()
            rank_sample_size = bsize
            indx1 = np.random.choice(bsize, rank_sample_size, replace=False)
            indx2 = np.random.choice(bsize, rank_sample_size, replace=False)
            group_1, group_2 = outputs_flat[indx1], outputs_flat[indx2]
            rank_labels = ((labels[indx1] > labels[indx2]).float() - 0.5) * 2
            rank_loss = self.ranking_loss_fn(group_1, group_2, rank_labels)
            #print("Normal Loss: {}".format(normal_loss.item()))
            #print("Rank Loss: {}".format(rank_loss.item()))
            ranking_loss_warmup_coef = 0.0 if self.epochs < self.ranking_loss_warmup_epochs else 1.0
            return rank_loss * ranking_loss_warmup_coef * self.ranking_loss_coef + normal_loss
