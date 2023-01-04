from typing import Optional

from nasflow.dataset.nas_dataset import NASDataSet
from nasflow.algo.surrogate.predictor import BasePredictor

from nasflow.algo.surrogate.predictor import (
    NNPredictor,
    NNPredictorWithEmbedding,
    NNPredictorDense,
    DenseSparseNNPredictor
)

def get_predictor(
        predictor_name: str,
        nas_dataset: NASDataSet,
        in_dims: int,
        num_epochs: int,
        num_inputs: int,
        loss_fn_name: Optional[str] = None,
        ranking_loss_fn_name: Optional[str] = None,
        batch_size: int = 64,
        **kwargs):
    if predictor_name == "embedding-nn":
        predictor = NNPredictorWithEmbedding(
            nas_dataset,
            optimizer='sgd',
            in_dims=in_dims,
            units=[256, 256],
            num_epochs=num_epochs,
            use_gpu=True,
            embedding_num_inputs=7 * num_inputs,
            embedding_table_size=3,
            embedding_dim=64,
            batch_size=batch_size,
            loss_fn_name=loss_fn_name,
            ranking_loss_fn_name=ranking_loss_fn_name,
            dropout=0.0,
            ema_decay=0.99,
            **kwargs)
    elif predictor_name == "dense-nn":
        predictor = NNPredictorDense(
            nas_dataset,
            optimizer='sgd',
            in_dims=in_dims,
            units=[256, 256],
            num_epochs=num_epochs,
            use_gpu=True,
            batch_size=batch_size,
            loss_fn_name=loss_fn_name,
            ranking_loss_fn_name=ranking_loss_fn_name,
            dropout=0.0,
            ema_decay=0.99,
            **kwargs)
    elif predictor_name == 'dense-sparse-nn':
        num_sparse_features = 2 * num_inputs
        dense_nn_pos_units=[64, 128, 256]
        dense_nn_arch_units=[64, 128, 256]
        predictor = DenseSparseNNPredictor(
            nas_dataset,
            optimizer='sgd',
            dense_nn_pos_units=dense_nn_pos_units,
            dense_nn_arch_units=dense_nn_arch_units,
            over_nn_units=[256, 256],
            activation="relu",
            num_dense_pos_features=2 * num_inputs,
            num_dense_arch_features=1 * num_inputs,
            num_sparse_features=num_sparse_features,
            num_pos_sparse_interact_outputs=dense_nn_pos_units[-1] + \
                (num_sparse_features+2) * (num_sparse_features+1) // 2,
            num_arch_sparse_interact_outputs=dense_nn_arch_units[-1] + \
                (num_sparse_features+2) * (num_sparse_features+1) // 2,
            embedding_table_size=3,
            embedding_dim=256,
            batch_size=batch_size,
            num_epochs=num_epochs,
            use_gpu=True,
            loss_fn_name=loss_fn_name,
            ranking_loss_fn_name=ranking_loss_fn_name,
            dropout=0.5,
            ema_decay=0.99,
            **kwargs
        )
        print(predictor.core_ml_arch)
    else:
        raise NotImplementedError("Predictor {} not supported!".format(predictor_name))
    return predictor
