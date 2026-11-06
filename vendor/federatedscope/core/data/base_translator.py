import json
import logging
from collections import Counter
from collections.abc import Mapping

import numpy as np

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.data import ClientData, StandaloneDataDict

logger = logging.getLogger(__name__)


def _sample_label(sample):
    """Return one integer label from tuple- or mapping-style datasets."""
    if isinstance(sample, Mapping):
        label_keys = [key for key in ("label", "labels", "y")
                      if key in sample]
        if len(label_keys) != 1:
            raise ValueError(
                "Expected exactly one label key in mapping sample, found "
                f"{label_keys}")
        label = sample[label_keys[0]]
    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        label = sample[1]
    else:
        raise TypeError(
            "Cannot extract a splitter label from sample type "
            f"{type(sample)!r}")

    if hasattr(label, "detach"):
        label = label.detach().cpu()
    if hasattr(label, "item"):
        try:
            label = label.item()
        except (RuntimeError, ValueError) as error:
            raise ValueError("Splitter labels must be scalar") from error
    if isinstance(label, np.ndarray):
        if label.size != 1:
            raise ValueError("Splitter labels must be scalar")
        label = label.reshape(-1)[0].item()
    return int(label)


def _client_label_lists(client_splits):
    """Extract client-keyed labels without assuming tuple-style samples."""
    return [[_sample_label(sample) for sample in client_split]
            for client_split in client_splits]


def _client_label_counts(client_labels):
    return {
        str(client_id): {
            str(label): int(count)
            for label, count in sorted(Counter(labels).items())
        }
        for client_id, labels in enumerate(client_labels, start=1)
    }


def _label_total_variation(train_labels, val_labels):
    """Measure per-client label-distribution mismatch in [0, 1]."""
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    if train_total <= 0 or val_total <= 0:
        raise ValueError("Cannot compare empty train/validation labels")
    labels = set(train_counts) | set(val_counts)
    return 0.5 * sum(
        abs(train_counts[label] / train_total -
            val_counts[label] / val_total)
        for label in labels)


class BaseDataTranslator:
    """
    Translator is a tool to convert a centralized dataset to \
    ``StandaloneDataDict``, which is the input data of runner.

    Notes:
        The ``Translator`` is consist of several stages:

        Dataset -> ML split (``split_train_val_test()``) -> \
        FL split (``split_to_client()``) -> ``StandaloneDataDict``

    """
    def __init__(self, global_cfg, client_cfgs=None):
        """
        Convert data to `StandaloneDataDict`.

        Args:
            global_cfg: global CfgNode
            client_cfgs: client cfg `Dict`
        """
        self.global_cfg = global_cfg
        self.client_cfgs = client_cfgs
        self.splitter = get_splitter(global_cfg)

    def __call__(self, dataset):
        """
        Args:
            dataset: `torch.utils.data.Dataset`, `List` of (feature, label)
                or split dataset tuple of (train, val, test) or Tuple of
                split dataset with [train, val, test]

        Returns:
            datadict: instance of `StandaloneDataDict`, which is a subclass of
            `dict`.
        """
        datadict = self.split(dataset)
        datadict = StandaloneDataDict(datadict, self.global_cfg)

        return datadict

    def split(self, dataset):
        """
        Perform ML split and FL split.

        Returns:
            dict of ``ClientData`` with client_idx as key to build \
            ``StandaloneDataDict``
        """
        train, val, test = self.split_train_val_test(dataset)
        datadict = self.split_to_client(train, val, test)
        return datadict

    def split_train_val_test(self, dataset, cfg=None):
        """
        Split dataset to train, val, test if not provided.

        Returns:
             List: List of split dataset, like ``[train, val, test]``
        """
        from torch.utils.data import Dataset, Subset

        if cfg is not None:
            splits = cfg.data.splits
        else:
            splits = self.global_cfg.data.splits
        if isinstance(dataset, tuple):
            # No need to split train/val/test for tuple dataset.
            error_msg = 'If dataset is tuple, it must contains ' \
                        'train, valid and test split.'
            assert len(dataset) == len(['train', 'val', 'test']), error_msg
            return [dataset[0], dataset[1], dataset[2]]

        index = np.random.permutation(np.arange(len(dataset)))
        train_size = int(splits[0] * len(dataset))
        val_size = int(splits[1] * len(dataset))

        if isinstance(dataset, Dataset):
            train_dataset = Subset(dataset, index[:train_size])
            val_dataset = Subset(dataset,
                                 index[train_size:train_size + val_size])
            test_dataset = Subset(dataset, index[train_size + val_size:])
        else:
            train_dataset = [dataset[x] for x in index[:train_size]]
            val_dataset = [
                dataset[x] for x in index[train_size:train_size + val_size]
            ]
            test_dataset = [dataset[x] for x in index[train_size + val_size:]]
        return train_dataset, val_dataset, test_dataset

    def split_to_client(self, train, val, test):
        """
        Split dataset to clients and build ``ClientData``.

        Returns:
            dict: dict of ``ClientData`` with ``client_idx`` as key.
        """

        # Initialization
        client_num = self.global_cfg.federate.client_num
        split_train, split_val, split_test = [[None] * client_num] * 3
        train_label_distribution = None

        # Split train/val/test to client
        if len(train) > 0:
            split_train = self.splitter(train)
            if self.global_cfg.data.consistent_label_distribution:
                try:
                    train_label_distribution = _client_label_lists(
                        split_train)
                except (KeyError, TypeError, ValueError) as error:
                    if str(self.global_cfg.federate.method).lower() == \
                            'cenmag_new':
                        raise RuntimeError(
                            "CenMag requires client-consistent train/val "
                            "label partitions, but train labels could not be "
                            "extracted") from error
                    logger.warning(
                        'Cannot access train label distribution for '
                        'splitter, split dataset without considering train '
                        'label.')
        if len(val) > 0:
            split_val = self.splitter(val, prior=train_label_distribution)
        if len(test) > 0:
            split_test = self.splitter(test, prior=train_label_distribution)

        if str(self.global_cfg.federate.method).lower() == 'cenmag_new' and \
                train_label_distribution is not None and len(val) > 0:
            val_label_distribution = _client_label_lists(split_val)
            train_counts = _client_label_counts(train_label_distribution)
            val_counts = _client_label_counts(val_label_distribution)
            label_total_variation = {
                str(client_id): _label_total_variation(
                    train_labels, val_labels)
                for client_id, (train_labels, val_labels) in enumerate(
                    zip(train_label_distribution, val_label_distribution),
                    start=1)
            }
            logger.info(
                "CENMAG_CLIENT_LABEL_COUNTS=%s",
                json.dumps({
                    "train": train_counts,
                    "val": val_counts,
                    "total_variation": label_total_variation,
                },
                           sort_keys=True))
            unseen = {}
            for client_id, (train_labels, val_labels) in enumerate(
                    zip(train_label_distribution, val_label_distribution),
                    start=1):
                train_support = set(train_labels)
                unseen_labels = sorted(set(val_labels) - train_support)
                if unseen_labels:
                    unseen[str(client_id)] = unseen_labels
            if unseen:
                raise RuntimeError(
                    "CenMag validation contains labels absent from the same "
                    f"client's train partition: {unseen}")
            excessive_mismatch = {
                client_id: distance
                for client_id, distance in label_total_variation.items()
                if distance > 0.05
            }
            if excessive_mismatch:
                raise RuntimeError(
                    "CenMag train/validation client-label partitions exceed "
                    "the 0.05 total-variation alignment gate: "
                    f"{excessive_mismatch}")

        # Build data dict with `ClientData`, key `0` for server.
        data_dict = {
            0: ClientData(self.global_cfg, train=train, val=val, test=test)
        }
        for client_id in range(1, client_num + 1):
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg
            data_dict[client_id] = ClientData(client_cfg,
                                              train=split_train[client_id - 1],
                                              val=split_val[client_id - 1],
                                              test=split_test[client_id - 1])
        return data_dict
