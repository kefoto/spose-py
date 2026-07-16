"""Shared infrastructure: filesystem layout, compute backend, cache, stats.

Nothing here is specific to a pipeline stage; every other subpackage may depend
on it, and it depends on none of them. Import submodules directly::

    from things_spose.core import paths, backend
"""
