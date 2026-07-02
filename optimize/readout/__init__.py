"""Readout optimization package configuration.

The optimizer generates figures from task and analysis callbacks, so it must use a
non-interactive Matplotlib backend. GUI backends such as TkAgg are not thread-safe
and can abort the process when their objects are finalized off the main thread.
"""

import matplotlib


matplotlib.use("Agg", force=True)
