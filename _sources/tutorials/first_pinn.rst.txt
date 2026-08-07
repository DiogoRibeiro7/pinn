Your First PINN: Solving Burgers Equation
=========================================

This tutorial walks through constructing a simple PINN for the Burgers equation.

.. doctest::
   :options: +ELLIPSIS

   >>> from pinn.models.mlp import MLP
   >>> model = MLP(in_dim=2, hidden_layers=1, width=10, out_dim=1)
   >>> import torch
   >>> inp = torch.zeros(1,2)
   >>> model(inp).shape
   torch.Size([1, 1])
