__all__ = ["learn"]


import collections

import torch
from mkb import utils
from river import stats

from ..sampling import positive_triples


def learn(
    model,
    dataset,
    optimizer,
    loss,
    evaluation,
    negative_sampling_size,
    device,
    epochs,
    eval_every,
    early_stopping_rounds,
):
    """Pipeline dedicated to automate training model.

    Parameters
    ----------
    dataset
        Dataset to dedicated to train the model.
    model
        Transformer based model.
    sampling
        Negative sampling method.
    epochs
        Number of epochs to train the model.
    validation
        Validation module.
    eval_every
        Eval the model every selected steps with the validation module.
    early_stopping_rounds
        Early stopping between validation steps.
    device
        Either cpu or cuda device.

    Examples
    --------

    >>> from mkb import losses, evaluation, datasets, text, models
    >>> from transformers import AutoTokenizer, AutoModel

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> train = [
    ...    ("jaguar", "cousin", "cat"),
    ...    ("tiger", "cousin", "cat"),
    ...    ("dog", "cousin", "wolf"),
    ...    ("dog", "angry_against", "cat"),
    ...    ("wolf", "angry_against", "jaguar"),
    ... ]

    >>> valid = [
    ...     ("cat", "cousin", "jaguar"),
    ...     ("cat", "cousin", "tiger"),
    ...     ("dog", "angry_against", "tiger"),
    ... ]

    >>> test = [
    ...     ("wolf", "angry_against", "tiger"),
    ...     ("wolf", "angry_against", "cat"),
    ... ]

    >>> dataset = datasets.Dataset(
    ...     batch_size = 5,
    ...     train = train,
    ...     valid = valid,
    ...     test = test,
    ...     seed = 42,
    ... )

    >>> device = "cpu"

    >>> model = models.SentenceTransformer(
    ...    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
    ...    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    gamma = 9,
    ...    device = device,
    ... )

    >>> model = model.to(device)

    >>> optimizer = torch.optim.Adam(
    ...     filter(lambda p: p.requires_grad, model.parameters()),
    ...     lr = 0.00005,
    ... )

    >>> evaluation = evaluation.TransformerEvaluation(
    ...     entities = dataset.entities,
    ...     relations = dataset.relations,
    ...     true_triples = dataset.train + dataset.valid + dataset.test,
    ...     batch_size = 2,
    ...     device = device,
    ... )

    >>> model = text.learn(
    ...     model = model,
    ...     dataset = dataset,
    ...     evaluation = evaluation,
    ...     optimizer = optimizer,
    ...     loss = losses.Adversarial(alpha=0.5),
    ...     negative_sampling_size = 5,
    ...     epochs = 1,
    ...     eval_every = 5,
    ...     early_stopping_rounds = 3,
    ...     device = device,
    ... )
    Validation:
        MRR: 0.2639
        MR: 3.8333
        HITS@1: 0.0
        HITS@3: 0.1667
        HITS@10: 1.0
        MRR_relations: 0.6667
        MR_relations: 1.6667
        HITS@1_relations: 0.3333
        HITS@3_relations: 1.0
        HITS@10_relations: 1.0
    Test:
        MRR: 0.3542
        MR: 3.0
        HITS@1: 0.0
        HITS@3: 0.75
        HITS@10: 1.0
        MRR_relations: 1.0
        MR_relations: 1.0
        HITS@1_relations: 1.0
        HITS@3_relations: 1.0
        HITS@10_relations: 1.0

    """
    metric_loss = stats.RollingMean(1000)
    round_without_improvement_valid, round_without_improvement_test = 0, 0
    history_valid, history_test = collections.defaultdict(float), collections.defaultdict(float)
    valid_scores, test_scores = {}, {}
    evaluation_done = False
    step = 0

    true_head, true_tail = positive_triples(triples=dataset.train + dataset.valid + dataset.test)

    entities = {id_e: e for e, id_e in dataset.entities.items()}

    for epoch in range(epochs):

        bar = utils.Bar(dataset=dataset, update_every=10)

        for data in bar:

            sample = data["sample"].to(device)
            weight = data["weight"].to(device)
            mode = data["mode"]

            triples = []
            for h, r, t in sample:
                h, r, t = h.item(), r.item(), t.item()
                triples.append((h, r, t))

            negative = in_batch_negative_triples(
                triples,
                negative_sampling_size=negative_sampling_size,
                mode=mode,
                true_head=true_head,
                true_tail=true_tail,
            )

            if not negative[0]:
                continue

            mapping_heads = {}
            mapping_tails = {}

            if model.twin:

                h_encode = []
                t_encode = []

                for index, (h, r, t) in enumerate(triples):
                    h_encode.append(entities[h])
                    t_encode.append(entities[t])
                    mapping_heads[h] = index
                    mapping_tails[t] = index

                embeddings_h = model.encoder(h_encode, mode="head")
                embeddings_t = model.encoder(t_encode, mode="tail")
                heads = torch.stack([e for e in embeddings_h], dim=0).unsqueeze(1)
                tails = torch.stack([e for e in embeddings_t], dim=0).unsqueeze(1)

            else:

                e_encode = []

                for index, (h, r, t) in enumerate(triples):
                    e_encode.append(entities[h])
                    e_encode.append(entities[t])
                    mapping_heads[h] = index
                    mapping_tails[t] = index

                embeddings = model.encoder(e_encode)

                heads = torch.stack(
                    [e for index, e in enumerate(embeddings) if index % 2 == 0], dim=0
                ).unsqueeze(1)
                tails = torch.stack(
                    [e for index, e in enumerate(embeddings) if index % 2 != 0], dim=0
                ).unsqueeze(1)

            relations = torch.index_select(
                model.relation_embedding, dim=0, index=sample[:, 1]
            ).unsqueeze(1)

            score = model.scoring(
                head=heads.to(device),
                relation=relations.to(device),
                tail=tails.to(device),
                mode=mode,
                gamma=model.gamma,
            )

            negative_scores = []
            for index, negative_sample in enumerate(negative):

                tensor_h = []
                tensor_r = []
                tensor_t = []

                for h, r, t in negative_sample:
                    tensor_h.append(heads[mapping_heads[h]])
                    tensor_r.append(relations[index])
                    tensor_t.append(tails[mapping_tails[t]])

                tensor_h = torch.stack(tensor_h, dim=0)
                tensor_r = torch.stack(tensor_r, dim=0)
                tensor_t = torch.stack(tensor_t, dim=0)

                negative_scores.append(
                    model.scoring(
                        head=tensor_h.to(device),
                        relation=tensor_r.to(device),
                        tail=tensor_t.to(device),
                        mode=mode,
                        gamma=model.gamma,
                    ).T
                )

            negative_scores = torch.stack(negative_scores, dim=1).squeeze(0)

            error = loss(score, negative_scores, weight)

            error.backward()

            _ = optimizer.step()

            optimizer.zero_grad()

            metric_loss.update(error.item())

            bar.set_description(f"Epoch: {epoch}, loss: {metric_loss.get():4f}")

            # Avoid doing evaluation twice for the same parameters.
            evaluation_done = False
            step += 1

            if evaluation is not None and not evaluation_done:

                if (step + 1) % eval_every == 0:

                    update_embeddings = True
                    evaluation_done = True

                    print(f"\n Epoch: {epoch}, step {step}.")

                    if dataset.valid:

                        valid_scores = evaluation.eval(
                            model=model,
                            dataset=dataset.valid,
                            update_embeddings=update_embeddings,
                        )

                        update_embeddings = False

                        valid_scores.update(
                            evaluation.eval_relations(
                                model=model,
                                dataset=dataset.valid,
                                update_embeddings=update_embeddings,
                            )
                        )

                        print_metrics(description="Validation:", metrics=valid_scores)

                    if dataset.test:

                        test_scores = evaluation.eval(
                            model=model,
                            dataset=dataset.test,
                            update_embeddings=update_embeddings,
                        )

                        update_embeddings = False

                        test_scores.update(
                            evaluation.eval_relations(
                                model=model,
                                dataset=dataset.test,
                                update_embeddings=update_embeddings,
                            )
                        )

                        print_metrics(description="Test:", metrics=test_scores)

                        if (
                            history_test["HITS@3"] > test_scores["HITS@3"]
                            and history_test["HITS@1"] > test_scores["HITS@1"]
                        ):
                            round_without_improvement_test += 1
                        else:
                            round_without_improvement_test = 0
                            history_test = test_scores
                    else:
                        if (
                            history_valid["HITS@3"] > valid_scores["HITS@3"]
                            and history_valid["HITS@1"] > valid_scores["HITS@1"]
                        ):
                            round_without_improvement_valid += 1
                        else:
                            round_without_improvement_valid = 0
                            history_valid = valid_scores

                    if (
                        round_without_improvement_valid == early_stopping_rounds
                        or round_without_improvement_test == early_stopping_rounds
                    ):

                        print(f"\n Early stopping at epoch {epoch}, step {step}.")

                        return model

    update_embeddings = True

    if dataset.valid and not evaluation_done and evaluation is not None:

        valid_scores = evaluation.eval(
            model=model, dataset=dataset.valid, update_embeddings=update_embeddings
        )

        update_embeddings = False

        valid_scores.update(evaluation.eval_relations(model=model, dataset=dataset.valid))

        print_metrics(description="Validation:", metrics=valid_scores)

    if dataset.test and not evaluation_done:

        test_scores = evaluation.eval(
            model=model, dataset=dataset.test, update_embeddings=update_embeddings
        )

        update_embeddings = False

        test_scores.update(evaluation.eval_relations(model=model, dataset=dataset.test))

        print_metrics(description="Test:", metrics=test_scores)

    return model


def print_metrics(description, metrics):
    print(f"\t {description}")
    for metric, value in metrics.items():
        print(f"\t\t {metric}: {value}")


def in_batch_negative_triples(triples, negative_sampling_size, mode, true_tail={}, true_head={}):
    """Generate in batch negative triples. All input sample will have the same number of fake triples."""
    negative = []
    for index_head, (h, r, _) in enumerate(triples):
        fake = []
        for index_tail, (_, _, t) in enumerate(triples):

            if index_head == index_tail:
                continue

            if t not in true_tail[(h, r)]:
                fake.append((h, r, t))

        negative.append(fake)

    for index_tail, (_, r, t) in enumerate(triples):
        for index_head, (h, _, _) in enumerate(triples):

            if index_head == index_tail:
                continue

            if h not in true_head[(r, t)]:
                negative[index_tail].append((h, r, t))

    min_length = min(map(len, negative))
    return [x[: min(negative_sampling_size, min_length)] for x in negative]
