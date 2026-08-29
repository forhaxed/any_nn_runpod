"""
The training side of a run.

``remote/train.py`` is an ordinary script.  What makes it a *remote* script is
one object:

    from any_nn_runpod import DatasetWrapper, RunpodTrainer, session

    trainer = MyTrainer(output_dir=session.output_dir)
    trainer.train_dataloader = DatasetWrapper("train", precache=32)
    session.bind(trainer)          # link, logger, datasets -- one call
    trainer.init()
    trainer.train()

``session`` is always there.  Launched by ``run.py`` it holds a live link to
your machine; run as a plain ``python remote/train.py`` it holds a ``NullLink``
and the same script trains standalone, writing its logs next to itself.  The
script does not branch on which.

This module is also the process entry point: ``python -m any_nn_runpod.run_session
--entry train.py --port 7778`` is what the launcher and the pod's supervisor
both start.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import traceback

from any_nn_runpod.link import Link, NullLink
from any_nn_runpod.wire.transport import TcpListener


class Session:
    """What the training script can reach: the link, the logger, the paths."""

    def __init__(self, link=None, output_dir: str = "./out", workdir: str | None = None):
        self.link = link or NullLink()
        self.output_dir = output_dir
        self.workdir = workdir or os.getcwd()
        self.logger = None

    @property
    def connected(self) -> bool:
        return bool(self.link.connected)

    def path(self, *parts) -> str:
        """A path inside the deployed ``remote/`` directory.

        Use it for anything you shipped alongside the script -- weights, a
        tokenizer, a config -- so the same line works here and on the pod.
        """
        return os.path.join(self.workdir, *parts)

    def bind(self, trainer):
        """Give a trainer the link, a logger, and connected datasets."""
        from any_nn_runpod.dataset import DatasetWrapper
        from any_nn_runpod.reporting import LoggerWrapper

        trainer.link = self.link
        if trainer.logger is None:
            trainer.logger = LoggerWrapper(self.link, trainer.output_dir)
        self.logger = trainer.logger

        for value in vars(trainer).values():
            if isinstance(value, DatasetWrapper):
                value.bind(self.link)
        return trainer

    # -- direct access, for anything this library does not model -----
    def call(self, name: str, payload=None, timeout=None):
        return self.link.call(name, payload, timeout=timeout)

    def notify(self, name: str, payload=None):
        self.link.notify(name, payload)

    def on(self, name: str):
        return self.link.on(name)


#: The session the training script sees.  Replaced by ``serve`` when the script
#: is launched with a link; a standalone run gets this one as it is.
session = Session()


# ======================================================================
#  Process entry
# ======================================================================
def serve(entry: str, host: str, port: int, output_dir: str, wait: float | None):
    """Accept one local connection (or don't), then run ``entry``."""
    global session

    link = NullLink()
    listener = None
    if wait is not None:
        listener = TcpListener(host, port)
        print(
            f"any_nn_runpod session listening on {host}:{listener.port}",
            file=sys.stderr,
            flush=True,
        )
        channel = listener.accept(timeout=wait)
        if channel is None:
            print(
                f"No local side connected within {wait:.0f}s -- running "
                "standalone.",
                file=sys.stderr,
                flush=True,
            )
        else:
            link = Link(channel, role="remote").start(initiate=False)

    workdir = os.path.dirname(os.path.abspath(entry))
    session = Session(link, output_dir=output_dir, workdir=workdir)
    _publish(session)

    # The script's own directory comes first, so `import helpers` finds what
    # you shipped next to it rather than something on the system path.
    sys.path.insert(0, workdir)

    code = 0
    try:
        runpy.run_path(entry, run_name="__main__")
    except SystemExit as exit_request:
        code = int(exit_request.code or 0)
    except BaseException as exc:  # noqa: BLE001 -- reported home, then re-raised
        code = 1
        text = traceback.format_exc()
        print(text, file=sys.stderr, flush=True)
        if link.connected:
            link.notify(
                "status",
                {
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": text,
                },
            )
    finally:
        if link.connected:
            # Not a ping: a ping is answered on arrival, while the "finished"
            # status is still queued behind however many log messages this run
            # produced. Hanging up then loses it, and the local side reports a
            # run that succeeded as one that died. This waits for the far side
            # to have actually handled everything.
            link.flush(timeout=120)
            link.close("session over")
        if listener is not None:
            listener.close()
    return code


def _publish(new_session: Session):
    """Make ``from any_nn_runpod import session`` see this one."""
    import any_nn_runpod

    any_nn_runpod.session = new_session
    sys.modules[__name__].session = new_session


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m any_nn_runpod.run_session")
    parser.add_argument("--entry", required=True, help="the training script to run")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7778)
    parser.add_argument("--output-dir", default="./out")
    parser.add_argument(
        "--wait",
        type=float,
        default=120.0,
        help="seconds to wait for the local side; 0 means do not wait at all",
    )
    parser.add_argument(
        "--no-link",
        action="store_true",
        help="run standalone: never listen, never expect a local side",
    )
    args = parser.parse_args(argv)

    return serve(
        entry=os.path.abspath(args.entry),
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        wait=None if args.no_link else args.wait,
    )
