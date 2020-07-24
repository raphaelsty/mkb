import tqdm

__all__ = ['Bar']


class Bar:
    """Wrapper for tqdm bar.

    Parameters:
        step (int): Number of iterations.
        update_every (int): Frequency of updates of tqdm bar.
        position (int): Position of the progress bar.

    """

    def __init__(self, step, update_every, position=0):
        self.bar = tqdm.tqdm(range(step), position=position)
        self.update_every = update_every
        self.n = 0

    def __call__(self, loss=None):
        return self.bar

    def set_description(self, text):
        if self.n % self.update_every == 0:
            self.bar.set_description(text)
        self.n += 1
