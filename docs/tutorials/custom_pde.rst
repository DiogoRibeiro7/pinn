Custom PDE Implementation Guide
===============================

How to implement a custom PDE residual for use with the PINN framework.

.. code-block:: python

   def residual(model, t, x):
       return model(torch.cat([t, x], dim=1))
