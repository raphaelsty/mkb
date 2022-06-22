__all__ = ["DPR"]

import torch

from ..text.scoring import TransE
from .base import TextBaseModel, mean_pooling


class DPR(TextBaseModel):
    """DPR with Sentence Transformers models.

    Parameters
    ----------
    gamma
        A higher gamma parameter increases the upper and lower bounds of the latent space and
        vice-versa.
    entities
        Mapping between entities id and entities label.
    relations
        Mapping between relations id and entities label.

    Examples
    --------

    >>> from transformers import AutoTokenizer, AutoModel
    >>> from mkb import datasets, models, text

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> dataset = datasets.Semanlink(1, pre_compute=False)

    >>> model = models.DPR(
    ...    head_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2'),
    ...    tail_model =  AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2'),
    ...    head_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2'),
    ...    tail_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2'),
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    gamma = 9,
    ...    device = 'cpu',
    ... )

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> model(sample)
    tensor([[3.5574],
            [3.4563]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
    >>> model(sample)
    tensor([[-73.1184],
            [-75.2535]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
    >>> model(sample)
    tensor([[-73.0617],
            [-75.3501]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> negative_sample = torch.tensor([[0], [2]])

    >>> model(sample, negative_sample, mode='head-batch')
    tensor([[3.5574],
            [3.4563]], grad_fn=<ViewBackward0>)

    >>> model(sample, negative_sample, mode='tail-batch')
    tensor([[3.5574],
            [3.4563]], grad_fn=<ViewBackward0>)

    References
    ----------
    1. [Sentence Similarity models](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads)

    """

    def __init__(
        self,
        head_model,
        tail_model,
        head_tokenizer,
        tail_tokenizer,
        entities,
        relations,
        scoring=TransE(),
        hidden_dim=None,
        gamma=9,
        device="cuda",
    ):

        if hidden_dim is None:
            hidden_dim = head_model.config.hidden_size

        super(DPR, self).__init__(
            hidden_dim=hidden_dim,
            entities=entities,
            relations=relations,
            scoring=scoring,
            gamma=gamma,
        )

        self.head_tokenizer = head_tokenizer
        self.tail_tokenizer = tail_tokenizer

        self.head_model = head_model
        self.tail_model = tail_model

        self.head_max_length = list(self.head_tokenizer.max_model_input_sizes.values())[0]
        self.tail_max_length = list(self.tail_tokenizer.max_model_input_sizes.values())[0]

        self.device = device

        if hidden_dim != self.head_model.config.hidden_size:

            self.linear_head = torch.nn.Linear(
                self.head_model.config.hidden_size, hidden_dim, bias=False
            )
            self.linear_tail = torch.nn.Linear(
                self.tail_model.config.hidden_size, hidden_dim, bias=False
            )

        else:

            self.linear_head = None
            self.linear_tail = None

    @property
    def twin(self):
        return True

    def encoder(self, e, mode=None):
        """Encode input entities descriptions.

        Parameters:
            e (list): List of description of entities.

        Returns:
            Torch tensor of encoded entities.
        """

        if mode is None:
            mode = "head"

        tokenizer, max_length = (
            self.head_tokenizer if mode == "head" else self.tail_tokenizer,
            self.head_max_length if mode == "head" else self.tail_max_length,
        )

        inputs = tokenizer.batch_encode_plus(
            e,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )

        input_ids, attention_mask = inputs["input_ids"].to(self.device), inputs[
            "attention_mask"
        ].to(self.device)

        if mode == "head":
            output = self.head_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            output = self.tail_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        sentence_embeddings = mean_pooling(
            hidden_state=output.last_hidden_state, attention_mask=attention_mask
        )

        if mode == "head" and self.linear_head is not None:
            return self.linear_head(sentence_embeddings)
        elif self.linear_tail is not None:
            return self.linear_tail(sentence_embeddings)

        return sentence_embeddings
