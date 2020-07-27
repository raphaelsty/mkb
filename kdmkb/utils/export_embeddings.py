import json
import os

__all__ = ['export_embebbedings']


def export_embebbedings(folder, dataset, model):
    """Export embeddings as json file.

    Parameters:
        folder (str): Folder in which relations and entities embeddings will be exported.
        dataset (kdmkb.datasets): Dataset.
        model (kdmkb.models): Model.

    """
    entities = {value: key for key, value in dataset.entities.items()}
    relations = {value: key for key, value in dataset.relations.items()}

    embeddings_e = {
        entities[key]: value.tolist() for key, value in model.embeddings['entities'].items()}

    embeddings_r = {
        entities[key]: value.tolist() for key, value in model.embeddings['relations'].items()}

    with open(os.path.join(f'{folder}/entities.json'), 'w') as output:
        json.dump(embeddings_e, output, indent=4)

    with open(os.path.join(f'{folder}/relations.json'), 'w') as output:
        json.dump(embeddings_r, output, indent=4)
