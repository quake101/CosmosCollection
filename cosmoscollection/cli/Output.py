"""Printing results for the command line: tables for people, JSON (--json) for
scripts, notes and errors on stderr so stdout stays clean for piping."""

import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Callable

# Exit codes
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_OFFLINE = 3  # The command needs the internet and offline mode is on


class CliError(Exception):
    """A failure to report as 'Error: message' (plus an optional hint line)."""

    def __init__(self, message, code=EXIT_ERROR, hint=None):
        super().__init__(message)
        self.code = code
        self.hint = hint


def configure_streams():
    """Never fail on a character the console's code page can't show (e.g. '→'
    with output redirected to a file on Windows). The app's own executable has
    no console (no stdout at all), so a command given to it prints nowhere
    rather than failing."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _stdout_is_utf8():
    return (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") == "utf8"


def print_json(data):
    """data as indented JSON - ASCII escapes unless stdout is UTF-8, so it's
    always valid JSON whatever the console's code page."""
    json.dump(data, sys.stdout, indent=2, ensure_ascii=not _stdout_is_utf8(), default=str)
    sys.stdout.write("\n")


def note(message):
    """A line for the person running the command (not part of the result)."""
    sys.stdout.flush()  # After what's been printed so far
    print(message, file=sys.stderr)


def error(message, hint=None):
    sys.stdout.flush()  # After what's been printed so far
    print(f"Error: {message}", file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)


@dataclass
class Column:
    """One table column: its header, how to get the cell text from a row, and
    alignment ('<' or '>'). flex columns are shortened first when the table is
    wider than the terminal."""
    header: str
    value: Callable[[Any], Any]
    align: str = "<"
    flex: bool = False


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def print_table(rows, columns, indent=""):
    """rows as an aligned table. On a terminal, long cells in flex columns are
    cut to fit its width; redirected output keeps every character."""
    cells = [[_cell(column.value(row)) for column in columns] for row in rows]
    widths = [max([len(column.header)] + [len(r[i]) for r in cells]) for i, column in enumerate(columns)]

    if sys.stdout.isatty():
        available = shutil.get_terminal_size((120, 24)).columns - len(indent) - 2 * (len(columns) - 1) - 1
        flex = [i for i, column in enumerate(columns) if column.flex]
        while sum(widths) > available and flex:
            widest = max(flex, key=lambda i: widths[i])
            if widths[widest] <= max(12, len(columns[widest].header)):
                break
            widths[widest] -= 1

    def line(texts):
        parts = []
        for text, width, column in zip(texts, widths, columns):
            if len(text) > width:
                text = text[:max(width - 1, 0)] + "…"
            parts.append(text.rjust(width) if column.align == ">" else text.ljust(width))
        return (indent + "  ".join(parts)).rstrip()

    print(line([column.header for column in columns]))
    print(indent + "  ".join("-" * width for width in widths))
    for texts in cells:
        print(line(texts))


def print_fields(pairs, indent=""):
    """'Label:  value' lines, labels aligned; pairs with a None value are skipped."""
    pairs = [(label, value) for label, value in pairs if value not in (None, "")]
    if not pairs:
        return
    width = max(len(label) for label, _ in pairs) + 1
    for label, value in pairs:
        print(f"{indent}{(label + ':').ljust(width)}  {value}")

