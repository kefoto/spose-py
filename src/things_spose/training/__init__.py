"""Stage 2 — the model, the training loop, run logging, and checkpointing.

``model`` is the SPoSE module; ``train`` the objective and loop; ``run_log``
per-run logging; ``checkpoint`` saving/loading model+optimizer state and the
fitted embedding. These pull in PyTorch. Import submodules directly::

    from things_spose.training import train, checkpoint
"""
