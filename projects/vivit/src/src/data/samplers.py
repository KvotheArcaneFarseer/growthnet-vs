"""
Batch Samplers for PyTorch DataLoaders

This module provides custom batch samplers for PyTorch that optimize data loading
by grouping sequences into length-based buckets. This approach minimizes padding
overhead and improves training efficiency for variable-length sequence data.

Classes:
    LengthShuffledBucketBatchSampler: Groups sequences by length into buckets,
        shuffles within buckets, and returns randomized batches.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, BatchSampler, RandomSampler
from typing import Any, Iterator


class LengthShuffledBucketBatchSampler(BatchSampler):
    """
    A PyTorch batch sampler that groups sequences into buckets based on their lengths.

    This sampler creates buckets based on the unique sequence lengths found in the dataset,
    shuffles samples within each bucket, creates batches from each bucket, and then shuffles
    the resulting batches. This approach ensures that sequences of similar lengths are batched
    together (minimizing padding) while maintaining randomness across epochs.

    Attributes:
        batch_size (int): Number of samples per batch.
        drop_last (bool): Whether to drop the last incomplete batch.
        seed (int): Random seed for reproducibility.
        indices (list[int]): List of dataset indices to sample from.
        unique_lengths (list[int]): Sorted list of unique sequence lengths in the dataset.
        sequence_lengths (list[int]): Sequence length for each sample in the dataset.
        num_samples (int): Total number of samples.
        buckets (list[np.ndarray]): List of arrays, each containing indices for one bucket.
        num_buckets (int): Total number of buckets created.
    """
    
    def __init__(
            self,
            dataset: Dataset,
            batch_size: int,
            drop_last: bool = False,
            seed: int = 42
    ) -> None:
        """
        Initializes the LengthShuffledBucketBatchSampler.

        Creates buckets based on unique sequence lengths in the dataset, assigns samples
        to buckets, and prepares the sampler for iteration.

        Args:
            dataset : Dataset : The PyTorch dataset to sample from. Can be a Subset or regular Dataset.
                Must have a 'data' attribute containing list of dictionaries with 'dates' keys.
            batch_size : int : Number of samples per batch.
            drop_last : bool : If True, drops the last batch if it's incomplete. Defaults to False.
            seed : int : Random seed for shuffling operations. Defaults to 42.

        Returns:
            None
        """
        # Build dummy sampler
        generator = torch.Generator()
        generator.manual_seed(seed)
        dummy_sampler = RandomSampler(dataset, generator=generator)

        # Initialize
        super().__init__(dummy_sampler, batch_size, drop_last)

        # Attributes for accelerate
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0
        self._g = generator

        # Compute the samples based on the dataset
        is_subset = isinstance(dataset, torch.utils.data.Subset)
        self.indices = dataset.indices if is_subset else list(range(len(dataset)))
        main_dataset = dataset.dataset if is_subset else dataset
        
        # Get the sequence lengths
        sequence_lengths, unique_lengths = self._get_data_lengths(main_dataset.data)
        self.unique_lengths = unique_lengths
        self.sequence_lengths = sequence_lengths
        
        self.num_samples = len(self.indices)
        self._create_buckets()
        self._len = self._compute_len()

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for proper random state synchronization in distributed training.

        This method updates the random generator seed based on the epoch number to ensure
        different shuffling patterns across epochs while maintaining reproducibility.

        Args:
            epoch : int : The current epoch number.

        Returns:
            None
        """
        self._g.manual_seed(epoch + self.seed)
    
    def _shuffle(
            self,
            ids: list[int]
    ) -> list[int]:
        """
        Shuffles a list of indices using the internal random generator.

        Returns the input unchanged if the list has 0 or 1 elements.

        Args:
            ids : list[int] : List of indices to shuffle.

        Returns:
            shuffled_ids : list[int] : Shuffled list of indices.
        """
        if len(ids) <= 1:
            return ids
        
        # Shuffle
        order = torch.randperm(len(ids), generator=self._g).tolist()

        return [ids[i] for i in order]

    def _get_data_lengths(
            self,
            data: list[dict[str, Any]]
    ) -> tuple[list[int], list[int]]:
        """
        Computes sequence lengths for each sample in the dataset.

        Extracts the length of the 'dates' field from each data dictionary to determine
        sequence lengths.

        Args:
            data : list[dict[str, Any]] : List of data dictionaries, each containing a 'dates' key
                with a sequence (list or array) value.

        Returns:
            data_sequence_lengths : list[int] : Sequence length for each sample, shape (n_samples,).
            unique_lengths : list[int] : Sorted array of unique sequence lengths found in the dataset,
                shape (n_unique_lengths,).
        """
        data_sequence_lengths = []

        for d in data:
            data_sequence_lengths.append(len(d["dates"]))
        
        return data_sequence_lengths, np.unique(data_sequence_lengths).tolist()
    
    def _create_buckets(self) -> None:
        """
        Creates buckets and assigns samples to them based on sequence length.

        Each bucket contains indices of samples with the same sequence length. Buckets are
        stored as numpy arrays for efficient indexing. Empty buckets are filtered out.

        Args:
            None

        Returns:
            None
        """
        bucket_boundaries = self.unique_lengths

        # Assign each sample to a bucket
        self.buckets = [ [] for _ in range(len(self.unique_lengths)) ]
        for i, seq_len in enumerate(self.sequence_lengths):
            bucket_id = np.searchsorted(bucket_boundaries, seq_len)
            self.buckets[bucket_id].append(i)
        
        # Convert to numpy arrays for efficiency
        self.buckets = [ np.array(bucket) for bucket in self.buckets if len(bucket) > 0 ]
        self.num_buckets = len(self.buckets)
    
    def _compute_len(self) -> int:
        """
        Computes the total number of batches that will be yielded.

        Calculates the number of batches by dividing each bucket's size by batch_size
        and summing across all buckets.

        Args:
            None

        Returns:
            total_batches : int : The total number of batches that will be produced.
        """
        total_batches = 0
        for bucket in self.buckets:
            total_batches += int(np.ceil(len(bucket) / self.batch_size))
        return total_batches
    
    def __iter__(self) -> Iterator[list[int]]:
        """
        Returns an iterator over batches of sample indices.

        Shuffles samples within each bucket, creates batches from each bucket, shuffles
        the batches, and yields them one at a time. This ensures both intra-bucket and
        inter-batch randomization.

        Args:
            None

        Returns:
            batch_iterator : Iterator[list[int]] : An iterator that yields batches of sample indices,
                where each batch is a list of integers of length up to batch_size.
        """
        bucket_indices = []
        # Shuffle the bucket contents
        for bucket in self.buckets:
            indices = self._shuffle(bucket)
            bucket_indices.append(torch.tensor(indices))
        
        # Create batches from each bucket
        all_batches = []
        for indices in bucket_indices:
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                all_batches.append(batch.tolist())
        
        # Shuffle the batches to ensure randomness across epochs
        batch_order = torch.randperm(len(all_batches), generator=self._g).tolist()

        # Yield the batches
        for j in batch_order:
            yield list(all_batches[j])

    def __len__(self) -> int:
        """
        Returns the total number of batches.

        Args:
            None

        Returns:
            length : int : The total number of batches.
        """
        return self._len