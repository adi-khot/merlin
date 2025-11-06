.. _feedforward_block:
==========================
Feed-forward (FF)
==========================

This page shows **minimal, working** recipes for building feed-forward pipelines with
:class:`merlin.algorithms.feed_forward.FeedForwardBlock` and
:class:`merlin.algorithms.feed_forward.PoolingFeedForward`, along with a short sketch
for customizing branches.

-----------------------------
A. Minimal feed-forward block
-----------------------------

The snippet below constructs a small FF tree that maps classical features to a
distribution over output Fock keys. The parameter choices keep the internal
state shapes simple and make the example easy to reuse.


.. code-block:: python

   import torch
   from merlin.algorithms.feed_forward import FeedForwardBlock

   # Minimal, stable configuration
   ffb = FeedForwardBlock(
       input_size=4,     # <= small classical input per sample
       n=2,              # number of photons
       m=4,              # number of modes
       depth=2,          # two FF steps
       conditional_modes=[0],  # single conditional mode
       state_injection=False,
   )

   # Introspection helpers
   n_outputs = ffb.get_output_size()           # positive integer
   n_k1      = ffb.size_ff_layer(1)            # number of branches at k=1
   sizes_k1  = ffb.input_size_ff_layer(1)      # allocated classical inputs per branch

   # Forward pass: returns probabilities that sum to ~1 per row
   x = torch.rand(4, ffb.input_size)           # batch of 4
   y = ffb(x)
   assert y.shape == (4, n_outputs)
   assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=1e-3)

   # Output key list (ordering is stable across calls without structural changes)
   keys = ffb.output_keys
   print(f"{len(keys)} output keys:", keys[:6], "...")


---------------------------------------
B. Quick training loop (gradient check)
---------------------------------------

A tiny optimization loop you can drop into notebooks or tests.

.. code-block:: python

   import torch
   from merlin.algorithms.feed_forward import FeedForwardBlock

   ffb = FeedForwardBlock(
       input_size=4, n=2, m=4, depth=2, conditional_modes=[0], state_injection=False
   )
   opt = torch.optim.Adam(ffb.parameters(), lr=1e-3)

   x = torch.rand(1, 4, requires_grad=True)
   y = ffb(x)                      # probabilities
   loss = (y ** 2).sum()           # dummy objective
   loss.backward()
   opt.step()
   opt.zero_grad()

   # Gradients should flow to inputs and parameters
   assert x.grad is not None and not torch.isnan(x.grad).any()


-------------------------------
C. Pooling feed-forward (PFF)
-------------------------------

:class:`PoolingFeedForward` takes amplitudes over a larger mode space and
re-indexes them into a smaller mode space, keeping compatible output keys.

Below we show two ways to drive it:

1) **Stand-alone** pooling with synthetic amplitudes (no dependency on a quantum layer).
This is the simplest way to understand shapes and verify behavior.

.. code-block:: python

   import torch
   from merlin.algorithms.feed_forward import PoolingFeedForward

   # Pool 16 modes (2 photons) down to 8 modes
   pff = PoolingFeedForward(n_modes=16, n_photons=2, n_output_modes=8)

   # Synthetic amplitudes over the input key set (match_indices + exclude_indices)
   n_in = len(pff.match_indices) + len(pff.exclude_indices)
   batch_size = 4
   amplitudes = torch.rand(batch_size, n_in)

   pooled = pff(amplitudes)  # shape: (batch_size, len(pff.keys_out))
   print(pooled.shape, "->", len(pff.keys_out), "output keys")

2) **End-to-end** pooling between quantum layers. If you already have layers that
produce/consume amplitudes, you can place the pooling module in between them. In
tests we use helpers that instantiate such layers; adapt to your layer utilities.

.. code-block:: python

   import torch
   from merlin.algorithms.feed_forward import PoolingFeedForward, define_layer_no_input

   # A simple pre → pool → post chain
   pff = PoolingFeedForward(n_modes=16, n_photons=2, n_output_modes=8)
   pre  = define_layer_no_input(16, 2)  # produces amplitudes on (16, 2)
   post = define_layer_no_input(8,  2)  # consumes amplitudes on (8,  2)

   amps = pre()          # forward on the pre-layer
   amps = pff(amps)      # pool down to 8 modes
   post.set_input_state(amps)
   res = post()          # continue computation (e.g., another amplitude map)

   assert isinstance(res, torch.Tensor) and res.requires_grad


---------------------------------------------
D. Sketch: customizing a branch (conceptual)
---------------------------------------------

You can customize per-branch quantum processing by replacing auto-built layers
at a given step ``k``. The outline below shows how you would:

1) Inspect how many branches exist at ``k`` and what classical input each branch
   receives; then
2) Build your branch-specific layers; finally
3) Register them into the FF block.

Use your preferred builder to construct circuits/layers (e.g., a circuit builder
utility) and ensure each custom layer’s input shape matches the branch allocation.

.. code-block:: python

   import torch
   from merlin.algorithms.feed_forward import FeedForwardBlock

   # Start from the minimal recipe
   ffb = FeedForwardBlock(
       input_size=4, n=2, m=4, depth=2, conditional_modes=[0], state_injection=False
   )

   # 1) Discover branches at step k=1
   k = 1
   n_branches = ffb.size_ff_layer(k)
   per_branch_inputs = ffb.input_size_ff_layer(k)  # list[int] of length n_branches

   # 2) (Sketch) Build your layers for each branch.
   #    Replace this with your own layer factory/builder for amplitudes over (m - k) modes.
   #    Example signature: make_layer(n_modes, n_photons, input_size) -> QuantumLayer
   def make_layer(n_modes, n_photons, input_size):
       # ... build your circuit/layer here ...
       raise NotImplementedError

   custom_layers_k = [
       make_layer(n_modes=ffb.m - k, n_photons=ffb.n_photons, input_size=inp)
       for inp in per_branch_inputs
   ]

   # 3) Register: assign in the same order as branches are enumerated internally
   ffb.define_ff_layer(k=k, layers=custom_layers_k)

   # From here, forward() uses your custom layers on step k
   _ = ffb(torch.rand(1, ffb.input_size))


--------------------------
Shape & parameter checklist
--------------------------

- **Minimal FF** (Section A/B): ``input_size=4, n=2, m=4, depth=2, conditional_modes=[0], state_injection=False``.
  This setting keeps amplitude tensors 2-D in practice and avoids shape pitfalls.

- **Pooling** (Section C):
  - Stand-alone: feed a tensor with width ``len(match_indices)+len(exclude_indices)``.
  - End-to-end: place :class:`PoolingFeedForward` between a producer and a consumer
    of amplitudes defined on matching mode spaces.

- **Custom branches** (Section D): for a step ``k``, use ``n_modes = m - k`` and
  the per-branch input sizes returned by ``input_size_ff_layer(k)``. Keep the photon
  number consistent with the block configuration.

