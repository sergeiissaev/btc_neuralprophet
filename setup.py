# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="btc_prediction_repo",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    version="0.1.0",
    description="Predicting bitcoin with neural prophet",
    author="Sergei Issaev",
)
