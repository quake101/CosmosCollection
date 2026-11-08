"""Questions for the person at the terminal. --yes answers confirmations;
--no-input (or no terminal, e.g. a script or scheduled task) never asks and
fails with a hint instead, so a script can't hang waiting for an answer."""

import sys

from cosmoscollection.cli.Output import EXIT_USAGE, CliError


def can_ask(ctx):
    return not ctx.no_input and sys.stdin is not None and sys.stdin.isatty()


def confirm(ctx, question, default=False):
    """Whether the person agrees to question ("Delete 12 files?")."""
    if ctx.yes:
        return True
    if not can_ask(ctx):
        raise CliError(f"{question} Not asking when running without a terminal (or with --no-input).",
                       EXIT_USAGE, hint="Pass --yes to go ahead.")
    choices = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            answer = input(f"{question} {choices} ").strip().lower()
        except EOFError:
            return False
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def choose(ctx, question, options, default=None, flag_hint=None):
    """One of options [(key, label)] chosen by number; default (a key) when the
    answer is empty. Without a terminal, default if given, else an error that
    names flag_hint (the option a script should pass instead)."""
    if not can_ask(ctx):
        if default is not None:
            return default
        raise CliError(f"{question} Not asking when running without a terminal (or with --no-input).",
                       EXIT_USAGE, hint=f"Pass {flag_hint} to choose." if flag_hint else None)
    print(question)
    for number, (key, label) in enumerate(options, 1):
        marker = " (default)" if key == default else ""
        print(f"  {number}. {label}{marker}")
    while True:
        try:
            answer = input("Choice: ").strip()
        except EOFError:
            answer = ""
        if not answer and default is not None:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
