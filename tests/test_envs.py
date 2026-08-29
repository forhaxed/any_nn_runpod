"""
Which environment a recipe describes, and when one has to be built.

The expensive mistakes here are silent: a venv that quietly lacks torch, or a
2.5 GB download for an environment that was already correct.
"""

from __future__ import annotations

import sys

from any_nn_runpod import envs


def test_an_empty_recipe_is_satisfied_by_whatever_is_running():
    """The common case, and the reason `run.py local` starts instantly."""
    assert envs.satisfied_by_current({}) is True
    assert envs.resolve({}, say=lambda _t: None) == sys.executable


def test_a_matching_python_and_torch_need_no_build():
    import torch

    recipe = {
        "python": ".".join(sys.version.split()[0].split(".")[:2]),
        "torch": torch.__version__.split("+")[0],
    }
    assert envs.satisfied_by_current(recipe) is True


def test_a_different_python_or_torch_needs_a_build():
    assert not envs.satisfied_by_current({"python": "2.7"})
    assert not envs.satisfied_by_current({"torch": "0.4.0"})


def test_a_pinned_requirement_is_never_assumed_present():
    """Pins exist because the exact version matters.

    Guessing that an installed `diffusers` is the pinned one is worse than a
    rebuild: the whole reason it was pinned is that the wrong version breaks.
    """
    assert not envs.satisfied_by_current({"requirements": ["pytest==1.0.0"]})
    assert not envs.satisfied_by_current({"requirements": ["diffusers>=0.37"]})
    assert not envs.satisfied_by_current({"requirements": ["git+https://x/y"]})


def test_an_unpinned_requirement_counts_if_it_imports():
    assert envs.satisfied_by_current({"requirements": ["pytest"]})
    assert not envs.satisfied_by_current({"requirements": ["nonexistent_module_xyz"]})


def test_a_recipe_that_does_not_replace_torch_inherits_the_base_environment():
    """Otherwise the venv has no torch and the image's correct one is ignored.

    A recipe naming packages but no torch version means "what is already here,
    plus these". Building it in isolation produces an environment where torch
    is missing, or is whatever version some dependency happened to pull -- and
    the CUDA build sitting in the pod's image goes unused.
    """
    # The flag is chosen from the recipe alone, so it can be checked without
    # spending two minutes and a gigabyte proving it.
    assert envs._inherits_base({"requirements": ["diffusers==0.37.1"]})
    assert envs._inherits_base({})

    # Replacing torch or the interpreter means isolation is the point.
    assert not envs._inherits_base({"torch": "2.9.1"})
    assert not envs._inherits_base({"python": "3.12"})


def test_the_cache_key_ignores_everything_but_the_environment():
    """Changing which GPU you rent must not rebuild the environment."""
    a = {"python": "3.11", "torch": "2.9.1", "requirements": ["x"]}
    b = {"requirements": ["x"], "torch": "2.9.1", "python": "3.11"}
    assert envs.spec_key(a) == envs.spec_key(b)
    assert envs.spec_key(a) != envs.spec_key({**a, "torch": "2.10.0"})


def test_series_compares_minor_versions_not_builds():
    assert envs.series("2.10.0+cu128") == "2.10"
    assert envs.series("3.11.9") == "3.11"
