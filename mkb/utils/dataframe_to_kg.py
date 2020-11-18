import pandas as pd


__all__ = ['DataFrameToKG']


def DataFrameToKG(df, keys):
    """

    Example:

        >>> from mkb import utils

        >>> df = pd.DataFrame({
        ...    'user': [1, 2, 3, 4, 5],
        ...    'banque': ['Societe Generale', 'Credit Lyonnais', 'Chinese National Bank', 'Chinese National Bank', 'QIWI'],
        ...    'country': ['France', 'France', 'China', 'China', 'Russia']
        ... })

        >>> keys = {
        ...    'user': ['banque'],
        ...    'banque': ['country'],
        ... }

        >>> utils.DataFrameToKG(df, keys)
        [(1, 'user_banque', 'Societe Generale'), (2, 'user_banque', 'Credit Lyonnais'), (3, 'user_banque', 'Chinese National Bank'), (4, 'user_banque', 'Chinese National Bank'), (5, 'user_banque', 'QIWI'), ('Societe Generale', 'banque_country', 'France'), ('Credit Lyonnais', 'banque_country', 'France'), ('Chinese National Bank', 'banque_country', 'China'), ('QIWI', 'banque_country', 'Russia')]

    """

    kg = []

    for head, tails in keys.items():

        if not isinstance(tails, list):
            tails = [tails]

        for tail in tails:

            subset = df[[head, tail]].drop_duplicates().copy(deep=True)
            subset.columns = ['head', 'tail']
            subset['relation'] = f'{head}_{tail}'

            kg = kg + \
                list(subset[['head', 'relation', 'tail']
                            ].to_records(index=False))

    return kg
