# -*- coding: utf-8 -*-
"""
Setup file

@author: Umair Akram
"""

import setuptools
from setuptools import setup,find_packages


with open("README.md", "r") as long_desc:
    long_description = long_desc.read()

setup(
    name = 'BrainStrokeClassifier',
    version = '0.2.2',
    description = 'Brain Stroke Classfier using Machine Learning!',
    author = 'Umair Akram',
    url = 'https://github.com/MUmairAB/BrainStrokeClassifier.git',
    #include_package_data=True,
    #package_data = find_packages(),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    
    packages=find_packages('src'),
    package_dir={'': 'src'},    
    keywords=['stroke','brain stroke','brain stroke prediction','stroke prediction','umair akram','the umair akram','trained model','xgboost'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    py_modules=['BrainStrokeClassifier'],
    install_requires = [
        'numpy>=1.21.5',
        'pandas>=1.5.3',
        'scikit-learn>=1.2.1',
        'xgboost==1.7.4',
        'pickle-mixin==1.0.2'
    ]
)