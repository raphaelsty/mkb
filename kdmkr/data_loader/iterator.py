# Reference: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

__all__ = ['Iterator']


class Iterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.iterator(dataloader_head)
        self.iterator_tail = self.iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
