import csv

import pandas as pd

__all__ = ["read_csv", "read_csv_classification"]


def read_csv(file_path):
    """Read csv file composed of triplets and convert it as list of tuples.

    [
        [e_1, r_1, e2],
        ...
        [e_n, r_n, e_p],
    ]

    """
    with open(f"{file_path}", "r") as csv_file:
        return [
            (int(head), int(relation), int(tail)) for head, relation, tail in csv.reader(csv_file)
        ]


def read_csv_classification(path):
    """Read triplets dedicated to classification. Released by NTN (Socher et al. 2013)."""
    df = pd.read_csv(path, header=None)
    df.columns = ["head", "relation", "tail", "label"]
    return {
        "X": df[["head", "relation", "tail"]].values.tolist(),
        "y": df["label"].values.tolist(),
    }
