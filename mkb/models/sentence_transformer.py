__all__ = ["Similarity"]

import torch

from ..text.scoring import TransE
from .base import TextBaseModel, mean_pooling


class SentenceTransformer(TextBaseModel):
    """SentenceTransformers models wrapper.

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

    >>> model = models.SentenceTransformer(
    ...    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
    ...    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    gamma = 9,
    ...    device = "cpu",
    ... )

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> model(sample)
    tensor([[3.3444],
            [3.6713]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
    >>> model(sample)
    tensor([[-72.9755],
            [-75.9919]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
    >>> model(sample)
    tensor([[-73.2439],
            [-74.5327]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> negative_sample = torch.tensor([[0], [2]])

    >>> model(sample, negative_sample, mode='head-batch')
    tensor([[3.3444],
            [3.6713]], grad_fn=<ViewBackward0>)

    >>> model(sample, negative_sample, mode='tail-batch')
    tensor([[3.3444],
            [3.6713]], grad_fn=<ViewBackward0>)

    References
    ----------
    1. [Sentence Similarity models](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads)

    """

    def __init__(
        self,
        model,
        tokenizer,
        entities,
        relations,
        scoring=TransE(),
        hidden_dim=None,
        gamma=9,
        device="cuda",
    ):
        init_linear = True
        if hidden_dim is None:
            hidden_dim = model.config.hidden_size
            init_linear = False

        super(SentenceTransformer, self).__init__(
            hidden_dim=hidden_dim,
            entities=entities,
            relations=relations,
            scoring=scoring,
            gamma=gamma,
        )

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = list(self.tokenizer.max_model_input_sizes.values())[0]

        self.linear = None
        if init_linear:
            self.linear = torch.nn.Linear(model.config.hidden_size, hidden_dim, bias=False)

    def encoder(self, e, **_):
        """Encode input entities descriptions.

        Parameters
        ----------
        e
            List of description of entities.

        Returns:
            Torch tensor of encoded entities.
        """
        inputs = self.tokenizer.batch_encode_plus(
            e,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sentence_embeddings = mean_pooling(
            hidden_state=output.last_hidden_state, attention_mask=attention_mask
        )

        if self.linear is not None:
            sentence_embeddings = self.linear(sentence_embeddings)

        return sentence_embeddings
