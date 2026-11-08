"""Cosmos Collection's command line.

    CosmosCollection-CLI <command> [options]       (packaged app)
    python main.py <command> [options]             (from source)
    python -m cosmoscollection.cli <command> ...

Lightweight on purpose: it never loads the app's windows (no QtGui, QtWidgets,
QtWebEngine or matplotlib). PySide6.QtCore is allowed - QSettings, so the
command line reads and writes the same settings as the app. Each command's
module imports what it needs only when that command runs, so --help is
instant. See docs/CLI.md.
"""

import argparse
import logging
import os
import sys

# Top-level command names, for main.py's "is this the command line?" check -
# kept here so that check imports nothing else
COMMANDS = frozenset({"dso", "images", "sessions", "targets", "stats", "sites", "library", "solve", "quality",
                      "tonight", "visibility", "weather", "help"})

# The old command line's options (app/CommandLine.py), still accepted
LEGACY_FLAGS = frozenset({"--add-image", "--list-dsos", "--search-dso"})

# Options that may come before the command name (python main.py --json dso ...)
_GLOBAL_FLAGS = frozenset({"--json", "-v", "--verbose", "-q", "--quiet", "--yes", "-y", "--no-input",
                           "-h", "--help", "--version"})
_GLOBAL_OPTIONS_WITH_VALUE = frozenset({"--site"})


def is_cli_executable():
    """Whether this is the packaged CosmosCollection-CLI executable."""
    if not getattr(sys, "frozen", False):
        return False
    name = os.path.splitext(os.path.basename(sys.executable))[0].lower()
    return name.endswith("-cli")


def wants_cli(argv):
    """Whether argv (sys.argv) asks for the command line rather than the app:
    the CLI executable, a command name (after any global options), --help, or
    one of the old command line's options."""
    if is_cli_executable():
        return True
    args = argv[1:]
    if any(arg.split("=", 1)[0] in LEGACY_FLAGS for arg in args):
        return True
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in COMMANDS:
            return True
        if arg in _GLOBAL_FLAGS:
            if arg in ("-h", "--help", "--version"):
                return True
            i += 1
        elif arg.split("=", 1)[0] in _GLOBAL_OPTIONS_WITH_VALUE:
            i += 1 if "=" in arg else 2
        else:
            return False
    return False


def prog_name():
    if getattr(sys, "frozen", False):
        return os.path.basename(sys.executable)
    script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    return "python main.py" if script == "main.py" else "python -m cosmoscollection.cli"


class Context:
    """What a command runs with: its parsed arguments and the global options."""

    def __init__(self, args):
        self.args = args
        self.json = args.json
        self.yes = args.yes
        self.no_input = args.no_input
        self.site_name = args.site

    def connection(self):
        """The database connection (context manager), as the app opens it - set
        up and migrated by DatabaseManager. Exceptions inside the with block
        aren't logged as database errors (a command's "No object named..." isn't
        one); they reach main() as they are."""
        from contextlib import contextmanager
        from cosmoscollection.core.DatabaseManager import DatabaseManager

        @contextmanager
        def opened():
            manager = DatabaseManager().get_connection()
            conn = manager.__enter__()
            try:
                yield conn
            finally:
                manager.__exit__(None, None, None)

        return opened()

    def require_online(self, feature):
        """Raise the offline error when offline mode is on (Settings > Application Settings)."""
        from cosmoscollection.core import OnlineLimits
        if OnlineLimits.is_offline():
            from cosmoscollection.cli.Output import EXIT_OFFLINE, CliError
            raise CliError(f"{feature} needs the internet.", EXIT_OFFLINE, hint=OnlineLimits.OFFLINE_NOTE)

    def site(self):
        """The observing location: --site NAME, else the active saved location."""
        from cosmoscollection.cli.Output import CliError
        from cosmoscollection.planning import Observer
        with self.connection() as conn:
            if self.site_name:
                site = Observer.find_site(conn, self.site_name)
                if site is None:
                    names = ", ".join(s.name for s in Observer.load_sites(conn)) or "none saved"
                    raise CliError(f"No saved location named '{self.site_name}'.", hint=f"Saved locations: {names}")
                return site
            site = Observer.active_site(conn)
        if site is None:
            raise CliError("No observing location is set up.",
                           hint="Add one in the app (Settings > Location), or pass --site NAME.")
        return site


def _common_options():
    """Global options, accepted before or after the command name."""
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("global options")
    group.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                       help="print the result as JSON (for scripts)")
    group.add_argument("-v", "--verbose", action="count", default=argparse.SUPPRESS,
                       help="show progress details (-vv for debug logging)")
    group.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                       help="only print the result and errors")
    group.add_argument("-y", "--yes", action="store_true", default=argparse.SUPPRESS,
                       help="answer yes to confirmations (e.g. deleting files)")
    group.add_argument("--no-input", action="store_true", default=argparse.SUPPRESS,
                       help="never ask a question; fail instead (for scripts)")
    group.add_argument("--site", metavar="NAME", default=argparse.SUPPRESS,
                       help="saved observing location to use (default: the active one)")
    return common


_GLOBAL_DEFAULTS = {"json": False, "verbose": 0, "quiet": False, "yes": False, "no_input": False, "site": None}


def build_parser():
    # The commands' modules: each has register(...), adding its commands with
    # set_defaults(func=run_function); run_function(ctx) returns an exit code
    # (None = 0). Imported by name, not importlib, so PyInstaller finds them;
    # they import what their commands need only when one runs.
    from cosmoscollection.cli import CollectionCommands, ImagingCommands, LibraryCommands, PlanningCommands

    common = _common_options()
    prog = prog_name()
    parser = argparse.ArgumentParser(
        prog=prog, parents=[common],
        description="Cosmos Collection - your deep sky collection from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""examples:
  {prog} dso search "M 31"
  {prog} dso show NGC 7000
  {prog} sessions list --since 2026-01-01
  {prog} images add M31 "C:/Astro/M31.jpg" --integration 3600 --favorite
  {prog} library scan "D:/Astro" --import
  {prog} sessions check --relink
  {prog} quality 37 --move
  {prog} solve "C:/Astro/M31.fits"
  {prog} tonight --type galaxy
  {prog} visibility M31 --date 2026-12-01
  {prog} weather
  {prog} --json stats

Run '{prog} <command> --help' for a command's options.""")
    parser.add_argument("--version", action="store_true", help="show the version and exit")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    groups = CollectionCommands.register(subparsers, common)
    LibraryCommands.register(subparsers, common, groups)
    ImagingCommands.register(subparsers, common)
    PlanningCommands.register(subparsers, common)
    help_parser = subparsers.add_parser("help", help="show help for a command")
    help_parser.add_argument("topic", nargs="*", help="command to show help for")
    return parser


def _setup_logging(args):
    """Warnings and errors only, unless -v (info) or -vv (debug); to stderr."""
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr, force=True)


def main(argv=None):
    """Run a command; its exit code."""
    from cosmoscollection.cli.Output import EXIT_ERROR, EXIT_OK, EXIT_USAGE, CliError, configure_streams, error

    configure_streams()
    argv = list(sys.argv[1:] if argv is None else argv)

    if any(arg.split("=", 1)[0] in LEGACY_FLAGS for arg in argv):
        from cosmoscollection.app.CommandLine import translate_legacy_arguments
        argv = translate_legacy_arguments(argv)

    parser = build_parser()
    args = parser.parse_args(argv)
    for name, value in _GLOBAL_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, value)

    if args.version:
        from cosmoscollection.version import get_version
        print(f"Cosmos Collection {get_version()}")
        return EXIT_OK
    if args.command in (None, "help"):
        topic = getattr(args, "topic", None)
        if topic:
            return main(topic + ["--help"])
        parser.print_help()
        return EXIT_OK if args.command == "help" else EXIT_USAGE
    if not hasattr(args, "func"):
        # A command group without its action ("dso" alone)
        parser.parse_args([args.command, "--help"])
        return EXIT_USAGE

    _setup_logging(args)
    # OpenBLAS thread cap and SSL setup, before numpy or a network module loads
    import cosmoscollection.core.ProcessSetup  # noqa: F401

    from cosmoscollection.core.OnlineLimits import OfflineError
    try:
        result = args.func(Context(args))
        return EXIT_OK if result is None else result
    except CliError as e:
        error(str(e), e.hint)
        return e.code
    except OfflineError as e:
        from cosmoscollection.cli.Output import EXIT_OFFLINE
        error(str(e))
        return EXIT_OFFLINE
    except KeyboardInterrupt:
        error("Stopped.")
        return 130
    except BrokenPipeError:
        # Output piped into something that stopped reading (e.g. | head)
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
        return EXIT_OK
    except Exception as e:
        logging.getLogger(__name__).debug("Command failed", exc_info=True)
        error(str(e) or e.__class__.__name__, "Run again with -vv for details." if args.verbose < 2 else None)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
