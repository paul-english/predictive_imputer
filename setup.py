#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'scikit-learn'
]

test_requirements = [
]

setup(
    name='predictive_imputer',
    version='0.2.0',
    description="Predictive imputation of missing values with sklearn interface. This is a simple implementation of the idea presented in the MissForest R package.",
    long_description=readme + '\n\n' + history,
    author="Paul English",
    author_email='paulnglsh@gmail.com',
    url='https://github.com/log0ymxm/predictive_imputer',
    packages=[
        'predictive_imputer',
    ],
    package_dir={'predictive_imputer':
                 'predictive_imputer'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='predictive_imputer',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
