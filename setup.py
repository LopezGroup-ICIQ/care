#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="GAMER-Net",
    version="0.1",
    description="GAMER-Net Setup",
    author="Santiago Morandi, Oliver Loveday",
    author_email="smorandi@iciq.es, oloveday@iciq.es",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
