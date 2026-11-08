"""The old command line's options, still accepted: translated to the commands
that replaced them (cosmoscollection/cli, docs/CLI.md).

    --list-dsos                      -> dso list
    --search-dso NAME                -> dso search NAME
    --add-image --dso D --image P    -> images add D P
        --equipment / --integration / --date / --notes (same names)
        --set-favorite               -> --favorite
"""

import argparse
import sys


def translate_legacy_arguments(argv):
    """argv (without the program name) in the new commands' form."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--add-image', action='store_true')
    parser.add_argument('--list-dsos', action='store_true')
    parser.add_argument('--search-dso')
    parser.add_argument('--dso')
    parser.add_argument('--image')
    parser.add_argument('--equipment', default='')
    parser.add_argument('--integration', default='')
    parser.add_argument('--date', default='')
    parser.add_argument('--notes', default='')
    parser.add_argument('--set-favorite', action='store_true')
    args, rest = parser.parse_known_args(argv)  # rest: global options such as --json

    if args.add_image:
        if not args.dso or not args.image:
            print("Error: --add-image needs --dso NAME and --image PATH", file=sys.stderr)
            sys.exit(2)
        new = ['images', 'add', args.dso, args.image]
        for option in ('equipment', 'integration', 'date', 'notes'):
            value = getattr(args, option)
            if value:
                new += [f'--{option}', value]
        if args.set_favorite:
            new.append('--favorite')
    elif args.search_dso:
        new = ['dso', 'search', args.search_dso]
    elif args.list_dsos:
        new = ['dso', 'list']
    else:
        return argv
    print(f"Note: the old options still work, but the command is now: {' '.join(_quoted(new))}", file=sys.stderr)
    return new + rest


def _quoted(args):
    return [f'"{a}"' if (' ' in a or not a) else a for a in args]
