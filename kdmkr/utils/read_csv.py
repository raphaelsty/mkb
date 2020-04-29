import csv


__all__ = ['read_csv']

def read_csv(file_path):
    """Read csv file composed of triplets and convert it as list of list.

    [
        [e_1, r_1, e2],
        ...
        [e_n, r_n, e_p],
    ]

    """
    with open(f'{file_path}', 'r') as csv_file:
        return [(int(head), int(relation), int(tail))
            for head, relation, tail in csv.reader(csv_file)]