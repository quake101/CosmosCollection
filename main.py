"""Cosmos Collection launcher. The app lives in the cosmoscollection package."""

# First, before anything imports numpy or Qt: thread caps, SSL certificates,
# Chromium flags, stderr log redirect and logging setup.
import cosmoscollection.app.Bootstrap  # noqa: F401

from cosmoscollection.app.Startup import main

if __name__ == "__main__":
    main()
