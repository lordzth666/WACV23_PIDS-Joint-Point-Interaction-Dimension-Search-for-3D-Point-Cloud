import os
from glob import glob
from typing import (
    Optional,
    Any,
)
import numpy as np
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nasflow.io_utils.base_io import maybe_load_pickle_file
class NASDataSet:
    """
    Create a dataset iterator to load NAS datasets containing of records of
    different networks.
    """
    def __init__(
            self,
            root_dir: str,
            pattern: str,
            record_name: str,
            max_shards: Optional[int] = None,
            cache: bool = False,
            map_fn: Any = None,
            random_state: Optional[int] = 233,
            max_records: int = None):
        self.root_dir = root_dir
        self.pattern = pattern
        self.record_name = record_name
        self.max_shards = max_shards
        # Now, identify the root directories
        self.all_directories = glob(os.path.join(root_dir, pattern))
        self.map_fn = map_fn
        print("Seeking records in the following directories: {}".format(self.all_directories))
        # Cat the records so that we do not need to process too much.
        if self.max_shards is not None and self.max_shards < len(self.all_directories):
            self.all_directories = self.all_directories[:self.max_shards]
        # Initialize train records and test records.
        self.train_records = None
        self.test_records = None
        self.max_records = max_records
        self.load_and_split_data(random_state=random_state)
        self.cache = cache
        if self.cache:
            print("Warning: cached mapped records will override original records.")
            self.train_records = process_map(self.map_fn, self.train_records, max_workers=16, chunksize=800)
            # self.train_records = [self.map_fn(x) for x in tqdm(self.train_records)]
            self.test_records = process_map(self.map_fn, self.test_records, max_workers=16, chunksize=800)
            print("Done!")

    def load_and_split_data(
            self,
            test_size: float = .15,
            random_state: Optional[int] = 233,
            verbose: bool = True):
        all_records = []
        for directory in self.all_directories:
            full_record_name = os.path.join(directory, self.record_name)
            record = maybe_load_pickle_file(full_record_name)
            all_records += record
        if self.max_records is not None and len(all_records) > self.max_records:
            if random_state is not None:
                np.random.seed(random_state)
            all_records = np.random.choice(all_records, self.max_records, replace=False)
            if random_state is not None:
                np.random.seed(None)
        all_records = np.asarray(all_records)
        # Filter out non-unique indices.
        print("Loading all unique records...")
        if 'hash' in all_records:
            print("Removing duplicate records...")
            all_hashes = [x['hash'] for x in all_records]
            _, unq_indices = np.unique(all_hashes, return_index=True)
            all_records = all_records[unq_indices]
        self.train_records, self.test_records = train_test_split(
            all_records, test_size=test_size, random_state=random_state)
        if verbose:
            print("Done! Training set contains {} records, testing set contains {} records.".format(
                len(self.train_records), len(self.test_records)))

    def iter_map_and_batch(
            self,
            shuffle: bool = True,
            batch_size: int = 64,
            drop_last_batch: bool = True,
            split: str = 'train'):
        if split == "train":
            return self.iter_map_and_batch_train_data(
                shuffle,
                batch_size,
                drop_last_batch
                )
        return self.iter_map_and_batch_test_data(
            shuffle,
            batch_size,
            drop_last_batch,
            )

    def iter_map_and_batch_train_data(
            self,
            shuffle: bool = True,
            batch_size: int = 64,
            drop_last_batch: bool = True,):
        assert self.train_records is not None, "Please load training records before proceed."
        if shuffle:
            np.random.shuffle(self.train_records)
        idx = 0
        while idx < len(self.train_records):
            start_idx, end_idx = idx, min(idx + batch_size, len(self.train_records))
            if drop_last_batch and idx + batch_size > len(self.train_records):
                pass
            else:
                result = self.train_records[start_idx:end_idx] if self.cache \
                    else [self.map_fn(x) for x in self.train_records[start_idx:end_idx]]
                yield result
            idx += batch_size

    def iter_map_and_batch_test_data(
            self,
            shuffle: bool = True,
            batch_size: int = 64,
            drop_last_batch: bool = True,
        ):
        assert self.test_records is not None, "Please load training records before proceed."
        if shuffle:
            np.random.shuffle(self.test_records)
        idx = 0
        while idx < len(self.test_records):
            start_idx, end_idx = idx, min(idx + batch_size, len(self.test_records))
            idx += batch_size
            if drop_last_batch and idx + batch_size - 1 > len(self.test_records):
                pass
            else:
                result = self.test_records[start_idx:end_idx] if self.cache \
                    else [self.map_fn(x) for x in self.test_records[start_idx:end_idx]]
                yield result
