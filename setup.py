#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'pandas==1.3.5',
    'numpy==1.21.4',
    'scikit-learn==1.0.1',
    'jpype1',   
    'pubchempy',
    'deepchem==2.5.0',
    'pytest',
    'lightgbm==3.3.2',
    'xgboost==1.5.2',
    'bayesian-optimization==1.2.0',    
    ]

test_requirements = [ ]

setup(
    author="Whitestone Yang",
    author_email='qyang_cope@163.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Molecular descriptor generation, data processing and model training.",
    entry_points={
        'console_scripts': [
            'spoc=spoc.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='spoc',
    name='spoc',
    packages=find_packages(include=['spoc', 'spoc.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/WhitestoneYang/spoc',
    version='0.1.0',
    zip_safe=False,
)
