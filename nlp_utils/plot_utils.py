import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def plot_col_group(df_train, col, key='label', transform=np.log1p, **kwargs):
    fig = plt.figure(dpi=200, **kwargs)
    df_plot = np.log1p(df_train[col]) if transform is not None else df_train[col]
    sns.distplot(df_plot.groupby(key).get_group(0)[col], hist=False, kde=True, **kwargs)
    sns.distplot(df_plot.groupby(key).get_group(1)[col], hist=False, kde=True, **kwargs)
    plt.legend(labels=['0', '1'])


def plot_col(df_train, df_test, col, transform=np.log1p, **kwargs):
    fig = plt.figure(dpi=200, **kwargs)
    if transform is not None:
        df_train_plot, df_test_plot = transform(df_train[col]), transform(df_test[col])
    else:
        df_train_plot, df_test_plot = df_train[col], df_test[col]
    sns.distplot(df_train_plot, hist=False, kde=True)
    sns.distplot(df_test_plot, hist=False, kde=True)
    plt.legend(labels=['train', 'test'])
