import tqdm

__all__ = ['Bar']


class Bar:
    """Wrapper for tqdm bar.

    Parameters:
        step (int): Number of iterations.
        update_every (int): Frequency of updates of tqdm bar.
        position (int): Position of the progress bar.

    >>> from mkb import datasets
    >>> from mkb import utils

    >>> for data in utils.Bar(dataset = datasets.CountriesS1(batch_size=10), update_every = 2):
    ...     pass

    """

    def __init__(self, dataset, update_every, position=0):
        self.bar = tqdm.tqdm(
            dataset,
            position=position,
        )
        self.update_every = update_every
        self.n = 0

    def __iter__(self, loss=None):
        yield from self.bar

    def set_description(self, text):
        if self.n % self.update_every == 0:
            self.bar.set_description(text)
        self.n += 1


class BarRange:
    """Wrapper for tqdm bar.

    Parameters:
        step (int): Number of iterations.
        update_every (int): Frequency of updates of tqdm bar.
        position (int): Position of the progress bar.

    >>> from mkb import datasets
    >>> from mkb import utils

    >>> for data in utils.Bar(dataset = datasets.CountriesS1(batch_size=10), update_every = 2):
    ...     pass

    """

    def __init__(self, step, update_every, position=0):
        self.bar = tqdm.tqdm(
            range(step),
            position=position,
        )
        self.update_every = update_every
        self.n = 0

    def __iter__(self, loss=None):
        yield from self.bar

    def set_description(self, text):
        if self.n % self.update_every == 0:
            self.bar.set_description(text)
        self.n += 1
