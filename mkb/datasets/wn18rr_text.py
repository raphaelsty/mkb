import os
import pathlib

import pandas as pd

from .dataset import Dataset


__all__ = ["Wn18rrText"]


class Wn18rrText(Dataset):
    """Wn18rr dataset with textual information about entities.

    Wn18rr aim to iterate over the associated dataset. It provide positive samples, corresponding
    weights and the mode (head batch / tail batch).

    Parameters:
        batch_size (int): Size of the batch.
        classification (bool): Must be set to True when using ConvE model to optimize BCELoss.
        shuffle (bool): Whether to shuffle the dataset or not.
        pre_compute (bool): Pre-compute parameters such as weights when using translationnal model
            (TransE, DistMult, RotatE, pRotatE, ComplEx). When using ConvE, pre-compute target
            matrix. When pre_compute is set to True, the model training is faster but it needs more
            memory.
        num_workers (int): Number of workers dedicated to iterate on the dataset.
        seed (int): Random state.

    Attributes:
        train (list): Training set.
        valid (list): Validation set.
        test (list): Testing set.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.

    Example:

        >>> from mkb import datasets

        >>> dataset = datasets.Wn18rrText(batch_size=1, shuffle=False, pre_compute=False, seed=42)

        >>> dataset
        Wn18rrText dataset
            Batch size          1
            Entities            41105
            Relations           11
            Shuffle             False
            Train triples       86835
            Validation triples  3034
            Test triples        3134

        >>> list(dataset.entities.keys())[:3]
        ['land_reform.n.01', 'cover.v.01', 'botany.n.02']

    References:
        1. [Datasets for Knowledge Graph Completion with Textual Information about Entities](https://github.com/villmow/datasets_knowledge_embedding)
        2. [Dettmers, Tim, et al. "Convolutional 2d knowledge graph embeddings." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.](https://arxiv.org/pdf/1707.01476.pdf)

    """

    def __init__(
        self,
        batch_size,
        classification=False,
        shuffle=True,
        pre_compute=True,
        num_workers=1,
        seed=None,
    ):

        self.filename = "wn18rr_text"

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=self._read_csv(file_path=f"{path}/train.csv"),
            valid=self._read_csv(file_path=f"{path}/valid.csv"),
            test=self._read_csv(file_path=f"{path}/test.csv"),
            batch_size=batch_size,
            shuffle=shuffle,
            classification=classification,
            pre_compute=pre_compute,
            num_workers=num_workers,
            seed=seed,
        )

    @classmethod
    def _read_csv(cls, file_path):
        return list(
            pd.read_csv(
                file_path,
                sep="|",
                header=None,
            ).itertuples(index=False, name=None)
        )
