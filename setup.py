# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

if os.path.exists("../version/version"):
    with open("../version/version") as f:
        version = f.read()
else:
    version = "0.2.0dev"


with open("README.md") as f:
    README = f.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="lightning_project_skeleton",
    version=version,
    description="Lightning Project Skeleton",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Niki Martinel",
    author_email="niki.martinel@gmail.com",
    url="https://github.com/iN1k1/lightning-project-skeleton",
    license="MIT",
    install_requires=requirements,
    setup_requires=requirements,
    pbr=True,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
