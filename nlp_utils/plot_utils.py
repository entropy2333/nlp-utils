import math
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


warnings.filterwarnings("ignore")


def _setup_figure(**kwargs):
    dpi = kwargs.pop("dpi", 300)
    figsize = kwargs.pop("figsize", None)
    fig = plt.figure(figsize=figsize)
    fig.set_dpi(dpi)
    return fig


def plot_col_group(
    df_train, col, key="label", label_names=None, transform=None, dpi=300, hist=False, kde=True, **kwargs
):
    """
    Plot distribution of a column in train data grouped by label.
    """
    _setup_figure(**kwargs)
    df_plot = transform(df_train[col]) if transform is not None else df_train[col]
    if label_names is None:
        label_names = df_train[key].unique()
    for label in label_names:
        sns.distplot(df_plot[df_train[key] == label], hist=hist, kde=kde)
    plt.legend(labels=label_names)


def plot_col_train_test(df_train, df_test, col, transform=None, hist=False, kde=True, **kwargs):
    """
    Plot distribution of a column in train and test data.
    """
    plot_col_multi_df([df_train, df_test], col, ["train", "test"], transform=transform, hist=hist, kde=kde, **kwargs)


def plot_col_multi_df(df_list, col, df_name_list=None, transform=None, hist=False, kde=True, **kwargs):
    """
    Plot distribution of a column in multiple dataframes.
    """
    _setup_figure(**kwargs)
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


def visualize(image, titles="", nrows=None, ncols=None, **kwargs):
    """

    example:
    >>> visualize(image, titles='image')
    >>> visualize([image, aug_image], titles=['image', 'aug_image'])
    >>> visualize(list(map(read_cv_image, image_paths)), titles=image_paths)
    """

    def get_nearest_factor(n: int):
        assert n > 0, "n must be positive"
        x = math.ceil(math.sqrt(n))
        y = n / x
        while int(y) != y:
            x += 1
            y = n / x
        assert x * y == n
        return x, int(y)

    if isinstance(image, list) and len(image) > 0:
        n_image = len(image)
        if nrows is None or ncols is None:
            nrows, ncols = get_nearest_factor(n_image)
            # make sure ncols >= nrows
            if nrows > ncols:
                nrows, ncols = ncols, nrows
        assert nrows * ncols == n_image
        logger.info(f"n_image: {n_image}, nrows: {nrows}, ncols: {ncols}")
        fig, (axes) = plt.subplots(nrows, ncols)
        dpi = kwargs.get("dpi", 300)
        fig.set_dpi(dpi)
        titles = titles or [""] * len(image)
        axes = axes.flatten()
        logger.debug(f"axes: {axes}")
        for i, (img, title) in enumerate(zip(image, titles)):
            logger.info(f"i: {i}, img: {img.shape}, title: {title}")
            axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis("off")
        plt.subplots_adjust(**kwargs)
        plt.tight_layout()
    else:
        _setup_figure(**kwargs)
        logger.info(f"image: {image.shape}, title: {titles}")
        plt.imshow(image)
        plt.axis("off")
        plt.title(titles)
