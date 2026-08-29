"""
Process entry for a training session:

    python -m any_nn_runpod.run_session --entry train.py --port 7778

Deliberately a separate module from ``runtime``.  ``python -m X`` imports the
package first and then executes X as ``__main__``, so pointing it at a module
the package's ``__init__`` already imported would load that module twice, under
two names, with two copies of every object in it -- including the ``session``
singleton the training script is supposed to find.  Python warns about exactly
this.  A module nothing else imports has no second copy to disagree with.
"""

from any_nn_runpod.runtime import main

if __name__ == "__main__":
    raise SystemExit(main())
