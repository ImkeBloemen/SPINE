SPINE
=====

.. image:: https://img.shields.io/pypi/v/spine.svg
   :target: https://pypi.python.org/pypi/spine
   :alt: PyPI version

.. image:: https://img.shields.io/travis/ImkeBloemen/spine.svg
   :target: https://travis-ci.com/ImkeBloemen/spine
   :alt: Build Status

.. image:: https://readthedocs.org/projects/spine/badge/?version=latest
   :target: https://spine.readthedocs.io/en/latest/?version=latest
   :alt: Documentation Status

.. image:: https://pyup.io/repos/github/ImkeBloemen/spine/shield.svg
   :target: https://pyup.io/repos/github/ImkeBloemen/spine/
   :alt: Dependency Updates

**SPINE** (Local Enriched Decision Boundary Map Visualization for Machine Learning Classifiers)
----------------------------------------------------------------------------------------------

SPINE provides a visualization technique for enriched, local decision boundary mapping in machine learning. 
It assists researchers and practitioners in better understanding model behavior near decision boundaries, 
especially for classification tasks. This tool is developed to foster reproducibility and clarity when 
analyzing localized performance of ML models.

.. contents::
   :local:
   :depth: 2

Features
--------

- **Local Decision Boundary Visualization**: Generates an enriched map illustrating how a classifier’s decision boundary
  shifts across different regions of the input space.
- **Extensibility**: Designed as a Python package for easy integration into existing ML workflows.
- **Support for Multiple Classifiers**: Works with various scikit-learn (and potentially other) classifiers to
  highlight local variations in performance.
- **Documentation**: Comprehensive usage and API documentation to help you get started quickly.

Installation
------------

This project uses **Poetry** for dependency management and packaging. There are two main ways to install and use SPINE:

1. **Install SPINE as a dependency in your Poetry-based project**:

   .. code-block:: bash

      poetry add spine

   This will add SPINE to your project’s dependencies and install it automatically.

2. **Clone the repository and install locally** (useful for development or contributions):

   .. code-block:: bash

      git clone https://github.com/ImkeBloemen/SPINE.git
      cd SPINE
      poetry install

   This command sets up a virtual environment (if not already active) and installs all dependencies, including SPINE 
   itself, for local development.

Usage
-----

Once installed, you can import SPINE into your Python code:

.. code-block:: python

   import spine

   from sklearn.datasets import make_classification
   

License
-------

This project is licensed under the **MIT License**. See the ``LICENSE`` file for details.

Credits
-------

SPINE was created using 
`Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the 
`audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`_ project template.  

Developed and maintained by `ImkeBloemen <https://github.com/ImkeBloemen>`_.

**Source Code**: https://github.com/ImkeBloemen/SPINE
