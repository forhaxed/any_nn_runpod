"""
any_nn_runpod -- a deliberate pipeline for training on a rented GPU.

Three things are yours to write:

    remote/train.py   the training script.  Written like any_nn, but its
                      dataloader may be somewhere else and its logs go home.
    remote/anr.toml   the recipe: what environment that script needs, and what
                      pod can host it.  Travels with the code.
    local/local.py    what this machine offers the run: datasets, and handlers
                      for anything it should finish here.

And one command runs them:

    python run.py local     build the environment here and train here
    python run.py start     rent a pod, ship remote/, train there

The same ``remote/train.py`` serves all of it.  It never learns which.
"""

from any_nn_runpod.dataset import DatasetWrapper
from any_nn_runpod.ema import AnyEMA
from any_nn_runpod.link import Link, NotConnected, NullLink, RemoteError
from any_nn_runpod.local import Local
from any_nn_runpod.reporting import LoggerWrapper, Reporter
from any_nn_runpod.runtime import Session, session
from any_nn_runpod.trainer import RunpodTrainer

__all__ = [
    "AnyEMA",
    "DatasetWrapper",
    "Link",
    "Local",
    "LoggerWrapper",
    "NotConnected",
    "NullLink",
    "RemoteError",
    "Reporter",
    "RunpodTrainer",
    "Session",
    "session",
]
__version__ = "0.1.0"
