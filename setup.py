# Lint as: python3
""" Utils for deep learning.
"""

import os

from setuptools import find_packages, setup


install_requires = [
    "tqdm",
    "dill",
    "omegaconf",
    "numpy",
    "pandas",
    "seaborn",
    "matplotlib",
    "scikit-learn",
]

extras = {}
extras["dev"] = [
    "black",
    "flake8",
    "isort",
]

setup(
    name="nlp-utils",
    version="0.1.1dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Utils for NLP.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch BERT",
    install_requires=install_requires,
    extras_requires=extras,
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,  # Required for mypy to find the py.typed file
)
