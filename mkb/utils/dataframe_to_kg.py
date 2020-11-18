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
        [('user_1', 'user_banque', 'banque_Societe Generale'), ('user_2', 'user_banque', 'banque_Credit Lyonnais'), ('user_3', 'user_banque', 'banque_Chinese National Bank'), ('user_4', 'user_banque', 'banque_Chinese National Bank'), ('user_5', 'user_banque', 'banque_QIWI'), ('banque_Societe Generale', 'banque_country', 'country_France'), ('banque_Credit Lyonnais', 'banque_country', 'country_France'), ('banque_Chinese National Bank', 'banque_country', 'country_China'), ('banque_QIWI', 'banque_country', 'country_Russia')]

    """

    kg = []

    for head, tails in keys.items():

        if not isinstance(tails, list):
            tails = [tails]

        for tail in tails:

            subset = df[[head, tail]].drop_duplicates().copy(deep=True)

            subset[head] = f'{head}_' + subset[head].astype('str')
            subset[tail] = f'{tail}_' + subset[tail].astype('str')

            subset.columns = ['head', 'tail']
            subset['relation'] = f'{head}_{tail}'

            kg = kg + \
                list(subset[['head', 'relation', 'tail']
                            ].to_records(index=False))

    return kg
