import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def plot_col_group(df_train, col, key='label', label_names=None, transform=None, dpi=300, hist=False, kde=True, **kwargs):
    """
    Plot distribution of a column in train data grouped by label.
    """
    fig = plt.figure(dpi=dpi, **kwargs)
    df_plot = transform(df_train[col]) if transform is not None else df_train[col]
    if label_names is None:
        label_names = df_train[key].unique()
    for label in label_names:
        sns.distplot(df_plot[df_train[key] == label], hist=hist, kde=kde)
    plt.legend(labels=label_names)


def plot_col_train_test(df_train, df_test, col, transform=None, dpi=300, hist=False, kde=True, **kwargs):
    """
    Plot distribution of a column in train and test data.
    """
    plot_col_multi_df([df_train, df_test], col, ['train', 'test'], transform=transform, dpi=dpi, hist=hist, kde=kde, **kwargs)


def plot_col_multi_df(df_list, col, df_name_list=None, transform=None, dpi=300, hist=False, kde=True, **kwargs):
    """
    Plot distribution of a column in multiple dataframes.
    """
    fig = plt.figure(dpi=dpi, **kwargs)
    for df in df_list:
        df_plot = transform(df[col]) if transform is not None else df[col]
        sns.distplot(df_plot, hist=hist, kde=kde, **kwargs)
    if df_name_list is None:
        df_name_list = [f"df_{i}" for i in range(len(df_list))]
    plt.legend(labels=df_name_list)