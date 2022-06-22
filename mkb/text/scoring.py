__all__ = ["ComplEx", "DistMult", "pRotatE", "RotatE", "Scoring", "TransE"]

from math import pi

import torch


class Scoring:
    def __init__(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def _repr_title(self):
        return f"{self.name} scoring"

    def __repr__(self):
        return f"{self._repr_title}"


class TransE(Scoring):
    """TransE scoring function.

    Examples
    --------
    >>> from mkb import text
    >>> text.TransE()
    TransE scoring

    """

    def __init__(self):
        super().__init__()

    def __call__(self, head, relation, tail, gamma, mode, **kwargs):
        """Compute the score of given facts (heads, relations, tails).

        Parameters
        ----------
        head
            Embeddings of heads.
        relation
            Embeddings of relations.
        tail
            Embeddings of tails.
        gamma
            Constant integer to stretch the embeddings.
        mode
            head-batch or tail-batch.

        """
        if mode == "head-batch":

            score = head + (relation - tail)

        else:

            score = (head + relation) - tail

        return gamma.item() - torch.norm(score, p=1, dim=2)


class RotatE(Scoring):
    """RotatE scoring function."""

    def __init__(self):
        super().__init__()
        self.pi = pi

    def __call__(self, head, relation, tail, gamma, embedding_range, mode, **kwargs):
        """Compute the score of given facts (heads, relations, tails).

        Parameters
        ----------
        head
            Embeddings of heads.
        relation
            Embeddings of relations.
        tail
            Embeddings of tails.
        gamma
            Constant integer to stretch the embeddings.
        embedding_range
            Range of the embeddings.
        mode
            head-batch or tail-batch.

        """
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (embedding_range.item() / self.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = gamma.item() - score.sum(dim=2)

        return score


class pRotatE(Scoring):
    """pRotatE scoring function."""

    def __init__(self):
        super().__init__()
        self.pi = pi

    def __call__(self, head, relation, tail, gamma, embedding_range, modulus, mode, **kwargs):
        """Compute the score of given facts (heads, relations, tails).

        Parameters
        ----------
        head
            Embeddings of heads.
        relation
            Embeddings of relations.
        tail
            Embeddings of tails.
        gamma
            Constant integer to stretch the embeddings.
        embedding_range
            Range of the embeddings.
        modulus
            Constant to multiply the score
        mode
            head-batch or tail-batch.

        """

        phase_head = head / (embedding_range.item() / self.pi)
        phase_relation = relation / (embedding_range.item() / self.pi)
        phase_tail = tail / (embedding_range.item() / self.pi)

        if mode == "head-batch":
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        return gamma.item() - score.sum(dim=2) * modulus


class DistMult(Scoring):
    """DistMult scoring function."""

    def __init__(self):
        super().__init__()

    def __call__(self, head, relation, tail, mode, **kwargs):
        """Compute the score of given facts (heads, relations, tails).

        Parameters
        ----------
        head
            Embeddings of heads.
        relation
            Embeddings of relations.
        tail
            Embeddings of tails.
        mode
            head-batch or tail-batch.

        """
        if mode == "head-batch":

            score = head * (relation * tail)

        else:

            score = (head * relation) * tail

        return score.sum(dim=2)


class ComplEx(Scoring):
    """ComplEx scoring function."""

    def __init__(self):
        super().__init__()

    def __call__(self, head, relation, tail, mode, **kwargs):
        """Compute the score of given facts (heads, relations, tails).

        Parameters
        ----------
        head
            Embeddings of heads.
        relation
            Embeddings of relations.
        tail
            Embeddings of tails.
        gamma
            Constant integer to stretch the embeddings.
        embedding_range
            Range of the embeddings.
        modulus
            Constant to multiply the score
        mode
            head-batch or tail-batch.

        """
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        return score.sum(dim=2)
