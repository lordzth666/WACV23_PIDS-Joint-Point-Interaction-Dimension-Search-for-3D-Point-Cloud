from typing import Any
from torch.utils.data import DataLoader

from pids_core.datasets.SemanticKitti import (
    SemanticKittiDataset,
    SemanticKittiSampler,
    SemanticKittiCollate
    )
from pids_core.datasets.ModelNet40 import (
    ModelNet40Dataset,
    ModelNet40Sampler,
    ModelNet40Collate
    )
from pids_core.datasets.S3DIS import (
    S3DISDataset,
    S3DISSampler,
    S3DISCollate
    )

def get_SemanticKITTI_dataset(split: str = "eval",
                              config: Any = None,
                              balance_class_test: bool = False,
                              untouched_ratio: float = 0.9):
    """
    Get the dataset split of the SemanticKITTI dataset.
    Args:
        split (str): Dataset split. Can be ['search', 'eval' and 'eval-test'].
        config (Any): Configuration.
    Returns:
        A tuple containing: (train dataloader, test dataloader, training_dataset,
        testing_dataset, training_sampler, testing_sampler).
    """
        # Initialize datasets
    skitti_set_lib = {
        'search': ("mini-train", "mini-val"),
        'eval': ("training", "validation"),
        'eval-test': ("training", "test"),
        'trainval-val': ("trainval", "validation"),
        'trainval-test': ("trainval", "test")
    }
    assert split in skitti_set_lib.keys(), KeyError("Invalid split key {}! \
        Expected one of {}.".format(split, skitti_set_lib.keys()))
    train_set, test_set = skitti_set_lib[split]
    training_dataset = SemanticKittiDataset(config, set=train_set,
                                            balance_classes=True)
    test_dataset = SemanticKittiDataset(config, set=test_set,
                                        balance_classes=balance_class_test)

    # Initialize samplers
    training_sampler = SemanticKittiSampler(training_dataset)
    test_sampler = SemanticKittiSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=SemanticKittiCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SemanticKittiCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate max_in_point value
    training_sampler.calib_max_in(config, training_loader, verbose=True, untouched_ratio=untouched_ratio)
    test_sampler.calib_max_in(config, test_loader, verbose=True, untouched_ratio=untouched_ratio)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True, untouched_ratio=untouched_ratio)
    test_sampler.calibration(test_loader, verbose=True,
                             untouched_ratio=untouched_ratio)
    return training_loader, test_loader, training_dataset, test_dataset, \
        training_sampler, test_sampler

def get_ModelNet40_dataset(split: str = "eval",
                           config: Any = None):
    """
    Get the dataset split of the ModelNet40 dataset.
    Args:
        split (str): Dataset split. Can be ['search', 'eval' and 'eval-test'].
        config (Any): Configuration.
    Returns:
        A tuple containing: (train dataloader, test dataloader, training_dataset,
  testing_dataset, training_sampler, testing_sampler).
    """
    # Initialize datasets
    modelnet40_set_lib = {
        'search': ("mini-train", "mini-val"),
        'eval': ("train", "test"),
    }
    assert split in modelnet40_set_lib.keys(), KeyError("Invalid split key {}! \
        Expected one of {}.".format(split, modelnet40_set_lib.keys()))
    train_set, test_set = modelnet40_set_lib[split]
    # Initialize datasets
    training_dataset = ModelNet40Dataset(config, split=train_set)
    test_dataset = ModelNet40Dataset(config, split=test_set)

    # Initialize samplers
    training_sampler = ModelNet40Sampler(training_dataset, balance_labels=True)
    test_sampler = ModelNet40Sampler(test_dataset, balance_labels=True)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=ModelNet40Collate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=ModelNet40Collate,
                             num_workers=config.input_threads,
                             pin_memory=True)
    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)
    return training_loader, test_loader, training_dataset, test_dataset, \
        training_sampler, test_sampler

def get_S3DIS_dataset(split: str = "eval",
                      config: Any = None):
    """
    Get the dataset split of the S3DIS dataset.
    Args:
        split (str): Dataset split. Can be ['search', 'eval' and 'eval-test'].
        config (Any): Configuration.
    Returns:
        A tuple containing: (train dataloader, test dataloader, training_dataset,
  testing_dataset, training_sampler, testing_sampler).
    """
    # Initialize datasets
    s3dis_set_lib = {
        'search': ("training", "validation"),
        'eval': ("training", "validation"),
    }
    assert split in s3dis_set_lib.keys(), KeyError("Invalid split key {}! \
        Expected one of {}.".format(split, s3dis_set_lib.keys()))
    train_set, test_set = s3dis_set_lib[split]
        # Initialize datasets
    training_dataset = S3DISDataset(
        config, set=train_set, use_potentials=True, validation_split=config.s3dis_validation_split)
    test_dataset = S3DISDataset(
        config, set=test_set, use_potentials=True, validation_split=config.s3dis_validation_split)

    # Initialize samplers
    training_sampler = S3DISSampler(training_dataset)
    test_sampler = S3DISSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=S3DISCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=S3DISCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)
    return training_loader, test_loader, training_dataset, test_dataset, \
        training_sampler, test_sampler
