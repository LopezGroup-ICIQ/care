#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="care",
    version="0.1",
    description="CARE (Catalytic Automated Reaction Evaluator), framework for automatically creating and manioulating chemical reaction networks in heterogeneous catalysis.",
    author="Santiago Morandi, Oliver Loveday",
    author_email="smorandi@iciq.es, oloveday@iciq.es",
    package_dir={"": "src"},
    packages=find_packages("src"),
    keywords=["heterogeneous catalysis", "chemical reaction networks", "graph neural networks", "machine learning"],
)
