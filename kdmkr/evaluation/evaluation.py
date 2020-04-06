import torch

from creme import stats


__all__ = ['evaluation']


def evaluation(model, test_dataloader_head, test_dataloader_tail):
    """Evaluate selected model with the metrics: MRR, MR, HITS@1, HITS@3, HITS@10"""
    model.eval()

    metrics = {
        metric: stats.Mean() for metric in ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']
    }

    with torch.no_grad():

        for test_dataset in [test_dataloader_tail, test_dataloader_head]:

            for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                positive_sample = positive_sample.cuda()
                negative_sample = negative_sample.cuda()
                filter_bias = filter_bias.cuda()

                score = model(sample = (positive_sample, negative_sample), mode = mode)

                score += filter_bias

                argsort = torch.argsort(score, dim = 1, descending=True)

                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]

                batch_size = positive_sample.size(0)
                for i in range(batch_size):
                    #Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()

                    metrics['MRR'].update(1.0/ranking)
                    metrics['MR'].update(ranking)
                    metrics['HITS@1'].update(1.0 if ranking <= 1 else 0.0)
                    metrics['HITS@3'].update(1.0 if ranking <= 3 else 0.0)
                    metrics['HITS@10'].update(1.0 if ranking <= 10 else 0.0)

    return {i: xi.get() for i, xi in metrics.items()}
