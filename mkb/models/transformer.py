import torch

from ..text.scoring import TransE
from .base import TextBaseModel

__all__ = ["Transformer"]


class Transformer(TextBaseModel):
    """Transformer for contextual representation of entities.

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

    >>> model = models.Transformer(
    ...    model = AutoModel.from_pretrained("bert-base-uncased"),
    ...    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased"),
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    gamma = 9,
    ...    device = "cpu",
    ... )

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> model(sample)
    tensor([[3.5391],
            [3.4745]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
    >>> model(sample)
    tensor([[-240.7030],
            [-209.8567]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
    >>> model(sample)
    tensor([[-240.2732],
            [-210.2421]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[9, 0, 0], [9, 2, 2]])
    >>> negative_sample = torch.tensor([[0], [2]])
    >>> model(sample, negative_sample, mode='head-batch')
    tensor([[3.5391],
            [3.4745]], grad_fn=<ViewBackward0>)

    >>> sample = torch.tensor([[0, 0, 9], [2, 2, 9]])
    >>> model(sample, negative_sample, mode='tail-batch')
    tensor([[3.5391],
            [3.4745]], grad_fn=<ViewBackward0>)

    """

    def __init__(
        self,
        model,
        tokenizer,
        entities,
        relations,
        scoring=TransE(),
        hidden_dim=None,
        max_length=None,
        gamma=9,
        device="cuda",
    ):
        if hidden_dim is None:
            hidden_dim = model.config.hidden_size

        super(Transformer, self).__init__(
            hidden_dim=hidden_dim,
            entities=entities,
            relations=relations,
            scoring=scoring,
            gamma=gamma,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = self.tokenizer.model_max_length if max_length is None else max_length

        if hidden_dim != self.model.config.hidden_size:
            self.linear = torch.nn.Linear(self.model.config.hidden_size, hidden_dim, bias=False)
        else:
            self.linear = None

    def encoder(self, e, mode=None):
        """Encode input entities descriptions.
        Parameters:
            e (list): List of description of entities.
        Returns:
            Torch tensor of encoded entities.
        """
        inputs = self.tokenizer.batch_encode_plus(
            e,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_token_type_ids=True,
        )

        output = self.model(
            input_ids=torch.tensor(inputs["input_ids"]).to(self.device),
            attention_mask=torch.tensor(inputs["attention_mask"]).to(self.device),
        )

        hidden_state = output[0]
        return self.linear(hidden_state[:, 0]) if self.linear is not None else hidden_state[:, 0]
