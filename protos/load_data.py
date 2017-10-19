import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import getLogger

TRAIN_DATA = '../input/train.csv'
TEST_DATA = '../input/test.csv'

logger = getLogger(__name__)


def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)
    """
    for col in tqdm(df.columns.values):
        if 'cat' in col:
            logger.info('categorical: {}'.format(col))
            tmp = pd.get_dummies(df[col], col)
            for col2 in tmp.columns.values:
                df[col2] = tmp[col2].values
            df.drop(col, axis=1, inplace=True)
    """
    logger.debug('exit')
    return df


def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    logger.debug('exit')
    return df


def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
