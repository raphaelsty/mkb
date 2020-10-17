import json
import os

__all__ = ['export_embeddings']


def export_embeddings(folder, model):
    """Export embeddings as json file.

    Parameters:
        folder (str): Folder in which relations and entities embeddings will be exported.
        model (mkb.models): Model.

    """
    embeddings_e = {
        key: value.tolist() for key, value in model.embeddings['entities'].items()
    }

    embeddings_r = {
        key: value.tolist() for key, value in model.embeddings['relations'].items()
    }

    with open(os.path.join(f'{folder}/entities.json'), 'w') as output:
        json.dump(embeddings_e, output, indent=4)

    with open(os.path.join(f'{folder}/relations.json'), 'w') as output:
        json.dump(embeddings_r, output, indent=4)
