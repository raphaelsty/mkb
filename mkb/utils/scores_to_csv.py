__all__ = ["ScoresToCsv"]

import pandas as pd
import os
import pickle

from pandas.core.algorithms import mode
from ..evaluation import Evaluation
from ..evaluation import find_threshold
from ..evaluation import accuracy


class ScoresToCsv:
    """Export scores using pandas.

    Parameters:

        path (str): Directory path to save the scores.


    Example:

        >>> from mkb import models
        >>> from mkb import datasets
        >>> from mkb import evaluation
        >>> from mkb import utils
        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset_1 = datasets.Umls(1, shuffle = False)

        >>> dataset_2 = datasets.Umls(1, shuffle = False)

        >>> model_1 = models.TransE(1, entities = dataset_1.entities, relations = dataset_1.relations,
        ...     gamma = 0)

        >>> model_2 = models.TransE(1, entities = dataset_2.entities, relations = dataset_2.relations,
        ...     gamma = 0)

        >>> datasets = {0: dataset_1, 1: dataset_2}
        >>> models = {0: model_1, 1: model_2}

        >>> scores_to_csv = utils.ScoresToCsv(
        ...    models = models,
        ...    datasets = datasets,
        ...    path = 'scores.csv',
        ...    detail_path = 'detail_scores.csv',
        ...    accuracy_path = 'accuracy_scores.csv',
        ...    save_dir = '',
        ...    prefix = 'X_distill_X',
        ...    device = 'cpu',
        ... )

        >>> scores = scores_to_csv.process(models = models, datasets = datasets, step = 10)

        >>> scores[["valid_MR", "test_MR", "model", "hidden_dim", "id", "dataset"]]
            valid_MR  test_MR   model  hidden_dim  id dataset
        0   58.7186  59.1717  TransE           1   0    Umls
        0   58.6051  58.6437  TransE           1   1    Umls

        >>> scores = scores_to_csv.detail_eval(datasets=datasets)

        >>> scores["scores_accuracy"]
            dataset   model id step  treshold accuracy_valid accuracy_test
        0    Umls  TransE  1   10 -0.856527       0.506135      0.484115
        1    Umls  TransE  0   10 -0.961645       0.516871       0.53177



    """

    def __init__(
        self,
        models,
        datasets,
        path=None,
        detail_path=None,
        accuracy_path=None,
        save_dir=None,
        prefix=None,
        device="cuda",
    ):
        self.path = path
        self.detail_path = detail_path
        self.accuracy_path = accuracy_path
        self.save_dir = save_dir
        self.prefix = prefix
        self.device = device

        self.evaluation = {}

        for id in models.keys():
            self.evaluation[id] = Evaluation(
                entities=datasets[id].entities,
                relations=datasets[id].relations,
                batch_size=8,
                true_triples=datasets[id].true_triples,
                device=self.device,
            )

    def to_dataframe(self, x):
        return pd.DataFrame.from_dict(x, orient="index").T

    def eval(self, model, dataset, evaluation, prefix=""):
        scores = evaluation.eval(dataset=dataset, model=model)
        scores = {f"{prefix}_{metric}": value for metric, value in scores.items()}
        return scores

    def save(self, models, datasets, step):
        """Save model."""
        for id in models.keys():
            filename = f"{models[id].name}_{id}_{datasets[id].name}_{step}.pickle"

            if self.prefix is not None:
                filename = f"{self.prefix}_{filename}"

            models[id].cpu().save(
                os.path.join(
                    self.save_dir,
                    filename,
                )
            )

            models[id] = models[id].to(self.device)

    def add_metadata(self, model, score, dataset, step, id, kwargs):
        score["step"] = step
        score["gamma"] = model.gamma.item()
        score["model"] = model.name
        score["hidden_dim"] = model.hidden_dim
        score["id"] = id
        score["dataset"] = dataset.name

        for key, value in kwargs.items():
            score[key] = value
        return score

    def process(self, models, datasets, step, **kwargs):

        scores = []

        for id in models.keys():

            valid_scores = self.eval(
                model=models[id],
                dataset=datasets[id].valid,
                evaluation=self.evaluation[id],
                prefix="valid",
            )

            test_scores = self.eval(
                model=models[id],
                dataset=datasets[id].test,
                evaluation=self.evaluation[id],
                prefix="test",
            )

            score = pd.concat(
                [self.to_dataframe(valid_scores), self.to_dataframe(test_scores)],
                axis="columns",
            )

            score = self.add_metadata(
                model=models[id],
                score=score,
                step=step,
                kwargs=kwargs,
                id=id,
                dataset=datasets[id],
            )

            scores.append(score)

        scores = pd.concat(scores, axis="rows")

        if self.path is not None:
            scores.to_csv(self.path, index=False)

        self.save(models=models, datasets=datasets, step=step)

        return scores

    def detail_eval(self, datasets, **kwargs):
        """Detailled evaluation with accuracy."""
        scores = []
        scores_accuracy = []

        df = pd.read_csv(
            self.path,
        )

        merge = ["dataset", "id", "model"]

        df = (
            df.groupby(merge + ["step"])
            .first()
            .reset_index()
            .sort_values(
                ["valid_MR"],
            )
            .groupby(merge)
            .first()
            .reset_index()
            .sort_values(
                ["valid_MR"],
            )
        )

        for _, (dataset, model, step, id,) in df[
            [
                "dataset",
                "model",
                "step",
                "id",
            ]
        ].iterrows():

            filename = f"{model}_{id}_{dataset}_{step}.pickle"

            if self.prefix is not None:
                filename = f"{self.prefix}_{filename}"

            with open(os.path.join(self.save_dir, filename), "rb") as model_file:

                model = pickle.load(model_file)

            score = self.evaluation[id].detail_eval(
                model=model, dataset=datasets[id].test
            )

            score["type"] = "test"

            score = self.add_metadata(
                model=model,
                score=score,
                step=step,
                id=id,
                dataset=datasets[id],
                kwargs=kwargs,
            )

            scores.append(score)

            treshold = find_threshold(
                model=model,
                X=datasets[id].classification_valid["X"],
                y=datasets[id].classification_valid["y"],
                batch_size=10,
                device=self.device,
            )

            accuracy_valid = accuracy(
                model=model,
                X=datasets[id].classification_valid["X"],
                y=datasets[id].classification_valid["y"],
                threshold=treshold,
                batch_size=10,
                device=self.device,
            )

            accuracy_test = accuracy(
                model=model,
                X=datasets[id].classification_test["X"],
                y=datasets[id].classification_test["y"],
                threshold=treshold,
                batch_size=10,
                device=self.device,
            )

            scores_accuracy.append(
                pd.DataFrame.from_dict(
                    {
                        "dataset": dataset,
                        "model": model.name,
                        "id": id,
                        "step": step,
                        "treshold": treshold,
                        "accuracy_valid": accuracy_valid,
                        "accuracy_test": accuracy_test,
                    },
                    orient="index",
                ).T
            )

        scores = pd.concat(scores, axis="rows")

        scores.to_csv(self.detail_path)

        scores_accuracy = pd.concat(scores_accuracy, axis="rows").reset_index(drop=True)

        scores_accuracy.to_csv(self.accuracy_path, index=False)

        return {"scores_accuracy": scores_accuracy, "scores": scores}
