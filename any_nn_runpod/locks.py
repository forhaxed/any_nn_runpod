"""
Lock-file control, kept exactly as it works in any_nn -- except the files sit
wherever the *output* is, which with a link means your own machine.

    touch out/pause.lock            training pauses
    rm    out/pause.lock            it resumes
    touch out/do_eval.lock          one evaluation, then the file is removed
    touch out/save_checkpoint.lock  one checkpoint, then the file is removed
    touch out/stop.lock             finish the current step and wind up
"""

from __future__ import annotations

import os
import threading

PAUSE = "pause.lock"
EVAL = "do_eval.lock"
SAVE = "save_checkpoint.lock"
STOP = "stop.lock"

#: Files that fire once and are deleted, paired with the action they mean.
ONE_SHOT = ((EVAL, "eval"), (SAVE, "save"), (STOP, "stop"))


def poll_locks(output_dir: str, state) -> list[str]:
    """Read the lock files once and return the actions they ask for.

    ``state`` is any object; a ``_paused`` attribute is kept on it so that
    pause and resume are reported on the edge rather than every poll.
    """
    actions = []
    try:
        paused = os.path.isfile(os.path.join(output_dir, PAUSE))
        was_paused = getattr(state, "_paused_flag", False)
        if paused and not was_paused:
            actions.append("pause")
        elif not paused and was_paused:
            actions.append("resume")
        state._paused_flag = paused

        for filename, action in ONE_SHOT:
            path = os.path.join(output_dir, filename)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    continue
                actions.append(action)
    except OSError:
        pass
    return actions


class LockWatcher(threading.Thread):
    """Polls the lock files and forwards what they ask for."""

    def __init__(self, output_dir: str, send_control, interval: float = 0.5):
        super().__init__(name="anr-locks", daemon=True)
        self.output_dir = output_dir
        self.send_control = send_control
        self.interval = interval
        self._stop = threading.Event()
        self._paused_flag = False

    def run(self):
        while not self._stop.wait(self.interval):
            for action in poll_locks(self.output_dir, self):
                self.send_control(action)

    def stop(self):
        self._stop.set()
