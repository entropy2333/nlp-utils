import math
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


warnings.filterwarnings("ignore")


def plot_col_group(
    df_train, col, key="label", label_names=None, transform=None, dpi=300, hist=False, kde=True, **kwargs
):
    """
    Plot distribution of a column in train data grouped by label.
    """
    plt.figure(dpi=dpi, **kwargs)
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
    plot_col_multi_df(
        [df_train, df_test], col, ["train", "test"], transform=transform, dpi=dpi, hist=hist, kde=kde, **kwargs
    )


def plot_col_multi_df(df_list, col, df_name_list=None, transform=None, dpi=300, hist=False, kde=True, **kwargs):
    """
    Plot distribution of a column in multiple dataframes.
    """
    plt.figure(dpi=dpi, **kwargs)
    for df in df_list:
        df_plot = transform(df[col]) if transform is not None else df[col]
        sns.distplot(df_plot, hist=hist, kde=kde, **kwargs)
    if df_name_list is None:
        df_name_list = [f"df_{i}" for i in range(len(df_list))]
    plt.legend(labels=df_name_list)


def read_cv_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def visualize(image, titles="", nrows=None, ncols=None):
    """

    example:
    >>> visualize(image, titles='image')
    >>> visualize([image, aug_image], titles=['image', 'aug_image'])
    """

    def get_nearest_factor(n: int):
        assert n > 0
        x = math.ceil(math.sqrt(n))
        y = x
        while x * y != n:
            x += 1
            y = int(n / x)
        return x, y

    if isinstance(image, list):
        n_image = len(image)
        if nrows is None or ncols is None:
            nrows, ncols = get_nearest_factor(n_image)
            # make sure ncols >= nrows
            if nrows > ncols:
                nrows, ncols = ncols, nrows
        assert nrows * ncols == n_image
        logger.info(f"n_image: {n_image}, nrows: {nrows}, ncols: {ncols}")
        fig, (axes) = plt.subplots(nrows, ncols)
        fig.set_dpi(200)
        titles = titles or [""] * len(image)
        if nrows == 1 or ncols == 1:
            axes = np.array([axes])
        logger.info(f"axes: {axes}")
        for i, (img, title) in enumerate(zip(image, titles)):
            logger.info(f"i: {i}, i // nrows: {i // nrows}, i % ncols: {i % ncols} title: {title}")
            axes[i // ncols, i % ncols].imshow(img)
            axes[i // ncols, i % ncols].set_title(title)
            axes[i // ncols, i % ncols].axis("off")
    else:
        plt.imshow(image)
        plt.axis("off")
        plt.title(titles)
