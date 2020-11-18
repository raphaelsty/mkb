import pandas as pd

from sklearn import decomposition


__all__ = ['dataframe_to_kg', 'map_embeddings', 'decompose']


def dataframe_to_kg(df, keys, prefix={}):
    """Convert pandas DataFrame to knowledge graph.

    Parameters:
        df (pd.DataFrame): dataset.
        keys (dict): Edges of the graphs.
        prefix (dict): Prefix to add to avoid collisions. Common columns should have common prefixes.

    Example:

        >>> from mkb import utils

        >>> df = pd.DataFrame({
        ...    'user': [1, 2, 3, 4, 5],
        ...    'banque': ['Societe Generale', 'Credit Lyonnais', 'Chinese National Bank', 'Chinese National Bank', 'QIWI'],
        ...    'country': ['France', 'France', 'China', 'China', 'Russia'],
        ... })

        >>> keys = {
        ...    'user': ['banque'],
        ...    'banque': ['country'],
        ... }

        >>> prefix = {
        ...    'user': 'user_',
        ...    'banque': 'banque_',
        ...    'country': 'country_',
        ... }

        >>> utils.DataFrameToKG(df, keys, prefix)
        [('user_1', 'user_banque', 'banque_Societe Generale'), ('user_2', 'user_banque', 'banque_Credit Lyonnais'), ('user_3', 'user_banque', 'banque_Chinese National Bank'), ('user_4', 'user_banque', 'banque_Chinese National Bank'), ('user_5', 'user_banque', 'banque_QIWI'), ('banque_Societe Generale', 'banque_country', 'country_France'), ('banque_Credit Lyonnais', 'banque_country', 'country_France'), ('banque_Chinese National Bank', 'banque_country', 'country_China'), ('banque_QIWI', 'banque_country', 'country_Russia')]

    """

    kg = []

    for head, tails in keys.items():

        if not isinstance(tails, list):
            tails = [tails]

        for tail in tails:

            subset = df[[head, tail]].drop_duplicates().copy(deep=True)

            # Add prefix to avoid collisions:
            if head in prefix:
                subset[head] = prefix[head] + subset[head].astype('str')

            if tail in prefix:
                subset[tail] = prefix[tail] + subset[tail].astype('str')

            subset.columns = ['head', 'tail']
            subset['relation'] = f'{head}_{tail}'

            kg = kg + \
                list(subset[['head', 'relation', 'tail']
                            ].to_records(index=False))

    return kg


def map_embeddings(df, prefix, embeddings, n_components, batch_size=None):
    """Map embeddings to input dataframe to train machine learning models. Apply PCA before mapping
    embeddigs. If batch size is defined, apply incremental PCA.
    """

    df_embeddings = df.copy()

    embeddings = decompose(embeddings=embeddings,
                           n_components=n_components, batch_size=batch_size)

    for column, prefix in prefix.items():

        df_embeddings[column] = prefix + df_embeddings[column].astype(str)

    for column in df.columns:

        df_embeddings[column] = df_embeddings[column].map(embeddings)

    columns_to_keep = []

    for column in df.columns:

        for i in range(n_components):

            df_embeddings[f'{column}_dim_{i}'] = df_embeddings[column].str[i]

            columns_to_keep.append(f'{column}_dim_{i}')

    return df_embeddings[columns_to_keep]


def decompose(embeddings, n_components, batch_size=None):
    """Apply CPA over input dataset. If batch size is not None, use incremental PCA."""

    embeddings = pd.DataFrame(embeddings).T.reset_index()

    embeddings = embeddings.set_index('index')

    if batch_size is None:
        PCA = decomposition.PCA(n_components=n_components)

    else:
        PCA = decomposition.IncrementalPCA(
            n_components=n_components, batch_size=batch_size)

    X = PCA.fit_transform(embeddings)

    output = {}

    for i, label in enumerate(embeddings.index):

        output[label] = X[i]

    return output
