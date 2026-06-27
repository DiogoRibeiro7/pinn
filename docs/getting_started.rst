Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install .

Development install:

.. code-block:: bash

   pip install -e .[dev,viz]

Quick check
-----------

.. doctest::
   :options: +ELLIPSIS

   >>> import pinn
   >>> hasattr(pinn, '__version__')
   True
