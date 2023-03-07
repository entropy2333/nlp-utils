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


def plot_images(image, titles="", nrows=None, ncols=None, **kwargs):
    """

    example:
    >>> plot_images(image, titles='image')
    >>> plot_images([image, aug_image], titles=['image', 'aug_image'])
    >>> plot_images(list(map(read_cv_image, image_paths)), titles=image_paths)
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

    if isinstance(image, list):
        assert len(image) > 0, "image list must not be empty"
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


def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def make_mask_image(image, bboxes, labels=None, colors=None, fill=True):
    """
    Args:
        image: image to draw on
        bboxes: list of bounding boxes
        labels: list of labels
        colors: list of colors
    Returns:
        mask image with bounding boxes
    """
    if colors is None:
        colors = [(255, 255, 255) for _ in range(len(bboxes))]
    if labels is None:
        labels = [""] * len(bboxes)
    # create a mask image
    mask = np.zeros(image.shape, dtype=np.uint8)
    for bbox, label, color in zip(bboxes, labels, colors):
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            if fill:
                cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            if fill:
                pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                # fill the polygon with the color
                cv2.fillPoly(mask, [pts], color)
            else:
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
                cv2.line(image, (x2, y2), (x3, y3), color, 2)
                cv2.line(image, (x3, y3), (x4, y4), color, 2)
                cv2.line(image, (x4, y4), (x1, y1), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return mask


def plot_venn_2(set1: set, set2: set, dpi: int = 150, colors=("blue", "green"), labels=("A", "B")):
    """
    plot venn graph on 2 sets

    >>> plot_venn_2(
        {"apple", "banana", "orange", "pear"},
        {"banana", "orange", "pear", "kiwi"},
    )
    """
    from matplotlib_venn import venn2, venn2_circles

    # create figure
    plt.figure(dpi=dpi)

    # create venn graph
    v = venn2(
        subsets=(set1, set2),
        set_colors=colors,
        set_labels=labels,
    )

    # plot intersection
    v.get_label_by_id("10").set_text("\n".join(set1 - set2))
    v.get_label_by_id("01").set_text("\n".join(set2 - set1))
    v.get_label_by_id("11").set_text("\n".join(set1 & set2))

    # add circle
    venn2_circles(subsets=(set1, set2), lw=0.5, color="black")

    # add title
    plt.title("Set Intersection Visualization")

    # show the graph
    plt.show()


def plot_venn_3(set1, set2, set3, dpi: int = 150, colors=("blue", "green", "red"), labels=("A", "B", "C")):
    """
    plot venn graph on 3 sets

    >>> plot_venn_3(
        {"apple", "banana", "orange", "pear"},
        {"banana", "orange", "pear", "kiwi"},
        {"banana", "orange", "pear", "grape"},
    )
    """
    from matplotlib_venn import venn3, venn3_circles

    # create figure
    plt.figure(dpi=dpi)

    # create venn graph
    v = venn3(
        subsets=(set1, set2, set3),
        set_colors=colors,
        set_labels=labels,
    )

    # plot intersection
    v.get_label_by_id("100").set_text("\n".join(set1 - set2 - set3))
    v.get_label_by_id("010").set_text("\n".join(set2 - set1 - set3))
    v.get_label_by_id("110").set_text("\n".join(set1 & set2 - set3))
    v.get_label_by_id("001").set_text("\n".join(set3 - set1 - set2))
    v.get_label_by_id("101").set_text("\n".join(set1 & set3 - set2))
    v.get_label_by_id("011").set_text("\n".join(set2 & set3 - set1))
    v.get_label_by_id("111").set_text("\n".join(set1 & set2 & set3))

    # add circle
    venn3_circles(subsets=(set1, set2, set3), lw=0.5, color="black")

    # add title
    plt.title("Set Intersection Visualization")

    # show the graph
    plt.show()
