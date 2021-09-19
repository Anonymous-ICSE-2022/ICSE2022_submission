import logging
import math
from operator import itemgetter
from typing import Any, Iterator, Optional, Sized

from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler, SubsetRandomSampler

logger = logging.getLogger(__name__)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler: Sized = sampler
        self.sampler_list: Optional[Any] = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.
    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self, data, sort_key=lambda x: x):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.
    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.
    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular libraries like
        ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together examples with a similar
        size length to reduce the padding required for each batch while maintaining some noise
        through bucketing.
        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py
        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225
    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    Example:
        >>> from torchnlp.random import set_seed
        >>> set_seed(123)
        >>>
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        sampler,
        batch_size,
        drop_last=False,
        sort_key=lambda x: x,
        bucket_size_multiplier=100,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        _bucket_size = batch_size * bucket_size_multiplier
        if hasattr(sampler, "__len__"):
            _bucket_size = min(_bucket_size, len(sampler))
        self.sampler = sampler
        self.bucket_sampler = BatchSampler(sampler, _bucket_size, False)
        self.current_epoch = 0

    def __iter__(self):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self.current_epoch)
            self.current_epoch += 1
            logging.info(f"Incrementing epoch (to {self.current_epoch})")

        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in SubsetRandomSampler(
                list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))
            ):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

    @property
    def dataset(self):
        return self.sampler

