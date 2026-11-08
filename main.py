"""Cosmos Collection launcher. The app lives in the cosmoscollection package.

Starts the command line instead (cosmoscollection/cli) for the
CosmosCollection-CLI executable, or when a command is given
(python main.py dso search M31) - before anything of the app's own loads, so
the command line stays quick and never loads its windows."""

import sys

from cosmoscollection.cli import Main as _cli

if __name__ == "__main__" and _cli.wants_cli(sys.argv):
    sys.exit(_cli.main())

# First, before anything imports numpy or Qt: thread caps, SSL certificates,
# Chromium flags, stderr log redirect and logging setup.
import cosmoscollection.app.Bootstrap  # noqa: E402,F401

from cosmoscollection.app.Startup import main  # noqa: E402

if __name__ == "__main__":
    main()
