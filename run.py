#!/usr/bin/env python
"""
The launcher.  Everything starts here.

    python run.py local              build the environment here, train here
    python run.py local --no-link    the same, with no local side at all

Add ``--root <dir>`` to drive a project that is not this directory -- the
examples, for instance:

    python run.py --root any_nn_runpod/examples/cifar local

It is called run.py and not setup.py on purpose: pip treats a file of that name
in the current directory as a build script, and would try to execute this one.
"""

import os
import sys

# Work from a checkout, whether or not any_nn_runpod is pip-installed.
_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "any_nn_runpod")
if os.path.isdir(os.path.join(_LIBRARY, "any_nn_runpod")):
    sys.path.insert(0, _LIBRARY)

from any_nn_runpod.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
