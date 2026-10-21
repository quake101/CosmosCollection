"""
Readable names for catalog objects.

The common name of an object comes from its aliases in the NAME catalog
("Andromeda Galaxy" for M 31). Window titles and new Target List entries and
sessions show it after the designation - "M 31 (Andromeda Galaxy)" - the way
names are written by hand in the Target List.

No UI imports, so any window (or a background thread) can use it.
"""

import logging
import re

logger = logging.getLogger(__name__)

CONSTELLATIONS = {
    "AND": "Andromeda", "ANT": "Antlia", "APS": "Apus", "AQR": "Aquarius", "AQL": "Aquila", "ARA": "Ara",
    "ARI": "Aries", "AUR": "Auriga", "BOO": "Boötes", "CAE": "Caelum", "CAM": "Camelopardalis",
    "CNC": "Cancer", "CVN": "Canes Venatici", "CMA": "Canis Major", "CMI": "Canis Minor",
    "CAP": "Capricornus", "CAR": "Carina", "CAS": "Cassiopeia", "CEN": "Centaurus", "CEP": "Cepheus",
    "CET": "Cetus", "CHA": "Chamaeleon", "CIR": "Circinus", "COL": "Columba", "COM": "Coma Berenices",
    "CRA": "Corona Australis", "CRB": "Corona Borealis", "CRV": "Corvus", "CRT": "Crater", "CRU": "Crux",
    "CYG": "Cygnus", "DEL": "Delphinus", "DOR": "Dorado", "DRA": "Draco", "EQU": "Equuleus",
    "ERI": "Eridanus", "FOR": "Fornax", "GEM": "Gemini", "GRU": "Grus", "HER": "Hercules",
    "HOR": "Horologium", "HYA": "Hydra", "HYI": "Hydrus", "IND": "Indus", "LAC": "Lacerta", "LEO": "Leo",
    "LMI": "Leo Minor", "LEP": "Lepus", "LIB": "Libra", "LUP": "Lupus", "LYN": "Lynx", "LYR": "Lyra",
    "MEN": "Mensa", "MIC": "Microscopium", "MON": "Monoceros", "MUS": "Musca", "NOR": "Norma",
    "OCT": "Octans", "OPH": "Ophiuchus", "ORI": "Orion", "PAV": "Pavo", "PEG": "Pegasus", "PER": "Perseus",
    "PHE": "Phoenix", "PIC": "Pictor", "PSC": "Pisces", "PSA": "Piscis Austrinus", "PUP": "Puppis",
    "PYX": "Pyxis", "RET": "Reticulum", "SGE": "Sagitta", "SGR": "Sagittarius", "SCO": "Scorpius",
    "SCL": "Sculptor", "SCT": "Scutum", "SER": "Serpens", "SEX": "Sextans", "TAU": "Taurus",
    "TEL": "Telescopium", "TRI": "Triangulum", "TRA": "Triangulum Australe", "TUC": "Tucana",
    "UMA": "Ursa Major", "UMI": "Ursa Minor", "VEL": "Vela", "VIR": "Virgo", "VOL": "Volans",
    "VUL": "Vulpecula",
}

# Better-known catalogs first, for an object's main designation ("M 31", not "NGC 224")
CATALOG_RANK = {"M": 0, "NGC": 1, "IC": 2, "Sh2": 3, "Barnard": 3, "LDN": 3, "vdB": 3, "LBN": 4,
                "Caldwell": 5, "Abell": 5, "Arp": 5, "Collinder": 5, "Melotte": 5, "Ced": 5}

COMMON_NAME_WORDS = {"nebula", "galaxy", "cluster", "remnant", "loop", "veil", "cloud", "pillars", "pinwheel",
                     "whirlpool", "sombrero", "pleiades", "beehive", "rosette", "lagoon", "trifid", "eagle"}

# "New General Catalogue 6913", "Messier Object 106", "NGC 2244 Satellite
# Cluster": a catalog number in words, not a name
CATALOG_NUMBER_NAME = re.compile(
    r"(?i)(?=.*\d)(?=.*\b(?:catalog|catalogue|messier|object|ngc|ic|index|melotte|collinder|caldwell|"
    r"sharpless|sh2|lbn|ldn|barnard|ugc|pgc|abell|cr|mel)\b)"
    r"|\b(?:M|NGC|IC|Sh2|C)[\s-]?\d+\b")  # ...or the designation itself ("Spiral Galaxy M106")

# "(Squid Nebula)" or "[notes]" after a name
_ANNOTATION = re.compile(r'\([^)]*\)|\[[^\]]*\]')

# QSettings key (bool, default True): sessions made from scanned files - the
# import wizard and the Session Manager's scans - are named with the common name
SESSION_COMMON_NAMES_SETTING = "sessions/add_common_names"


def pick_common_name(aliases):
    """The most readable of an object's NAME aliases: "Orion Nebula", not
    "Ori Nebula" or "CRAB NEB"."""
    def score(name):
        words = name.split()
        abbreviation = any(word.upper() in CONSTELLATIONS and len(word) == 3 for word in words)
        descriptive = words[-1].lower() in COMMON_NAME_WORDS
        return (abbreviation, not descriptive, len(words) == 1, len(name))
    readable = [name for name in aliases if name and not name.isupper() and len(name) > 3
                and not CATALOG_NUMBER_NAME.search(name)]
    return min(readable, key=score) if readable else None


def preferred_designation(designations):
    """The best-known of (catalogue, designation) pairs as text ("M 31"), or
    None: M, NGC, IC ... and '7318A' over '7318-1'."""
    if not designations:
        return None
    catalogue, designation = min(designations, key=lambda d: (CATALOG_RANK.get(d[0], 9), "-" in d[1], d[0], d[1]))
    return f"{catalogue} {designation}"


def base_name(name):
    """A name without its parenthetical common name or bracketed notes."""
    return _ANNOTATION.sub('', name or '').strip()


def designation_candidates(name):
    """(catalogue, designation) pairs a name could be ('Sh2-129' -> ('Sh2', '129');
    'vdB 152 (Wolf's Cave Nebula)' -> ('vdB', '152'))."""
    base = base_name(name)
    if not base or '&' in base or ',' in base:
        return []  # Pairs/groups ('NGC 4038 & NGC 4039') aren't one catalog object
    candidates = []
    parts = base.split()
    if len(parts) >= 2:
        candidates.append((parts[0], " ".join(parts[1:])))
    for pattern in (r'([A-Za-z]+)[-_ ]?(\d[\w.+\-]*)', r'([A-Za-z]+\d)[-_ ]+(\d[\w.+\-]*)'):
        match = re.fullmatch(pattern, base)
        if match:
            candidates.append((match.group(1), match.group(2)))
    return list(dict.fromkeys(candidates))


def with_common_name(name, common):
    """'M 31' + 'Andromeda Galaxy' -> 'M 31 (Andromeda Galaxy)'; the name as it
    is when there's no common name or it already shows one."""
    name = (name or "").strip()
    if not common or not name or "(" in name or common.lower() in name.lower():
        return name
    return f"{name} ({common})"


def common_name_from_designations(designations):
    """The common name in a designations list as the main window keeps it
    ("M 31, NGC 224, NAME Andromeda Galaxy"), or None."""
    aliases = [d.strip()[5:] for d in (designations or "").split(",") if d.strip().startswith("NAME ")]
    return pick_common_name(aliases)


def load_common_names(conn, dsodetailids):
    """{dsodetailid: readable common name} ("Bode's Galaxy") from the NAME catalog."""
    ids = sorted({str(i) for i in dsodetailids if i})
    aliases = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        for dso, name in conn.execute(f"SELECT dsodetailid, designation FROM cataloguenr WHERE catalogue = 'NAME' "
                                      f"AND dsodetailid IN ({','.join('?' * len(chunk))})", chunk).fetchall():
            aliases.setdefault(str(dso), []).append(str(name))
    return {dso: name for dso, names in aliases.items() if (name := pick_common_name(names))}


def catalog_ids(conn, name):
    """The catalog objects (dsodetail ids) a name designates - a designation
    ('Sh2-129', 'm31') or a common name ('Andromeda Galaxy'); empty if none."""
    ids = set()
    for catalogue, designation in designation_candidates(name):
        ids.update(row[0] for row in conn.execute(
            "SELECT dsodetailid FROM cataloguenr WHERE catalogue = ? COLLATE NOCASE AND designation = ? COLLATE NOCASE",
            (catalogue, designation)))
    if not ids and base_name(name):
        ids.update(row[0] for row in conn.execute(
            "SELECT dsodetailid FROM cataloguenr WHERE catalogue = 'NAME' AND designation = ? COLLATE NOCASE",
            (base_name(name),)))
    return frozenset(ids)


def common_name(conn, name=None, dsodetailid=None):
    """The object's common name, by its id or what it's called; None if it has
    none or the name doesn't pick out one object."""
    if not dsodetailid:
        ids = catalog_ids(conn, name)
        if len(ids) != 1:
            return None
        dsodetailid = next(iter(ids))
    return load_common_names(conn, [dsodetailid]).get(str(dsodetailid))


def display_name(name, dsodetailid=None, designations=None, conn=None):
    """The name with its common name added - "M 31 (Andromeda Galaxy)" - for
    window titles and the names of new targets and sessions. Unchanged if it
    already has one in parentheses or none is known; never raises.

    designations: the main window's list for the object - when given, the
    database isn't needed; conn: an open connection, else one is opened."""
    name = (name or "").strip()
    if not name or "(" in name:
        return name
    try:
        if designations:
            common = common_name_from_designations(designations)
        else:
            if conn is None:
                from DatabaseManager import DatabaseManager
                with DatabaseManager().get_connection() as own_conn:
                    common = common_name(own_conn, name, dsodetailid)
            else:
                common = common_name(conn, name, dsodetailid)
        return with_common_name(name, common)
    except Exception as e:
        logger.debug(f"No common name for {name!r}: {e}")
        return name


def suggestion_names(conn, rows, typed=""):
    """Names for catalog search results, in order: [(label, dsodetailid)] from
    [(dsodetailid, catalogue, designation)] rows.

    A designation gets the common name ("NGC 224 (Andromeda Galaxy)"); common
    names that matched give one entry per object, with its main designation
    ("M 31 (Andromeda Galaxy)") - the alias shown is one containing 'typed',
    since a completer that filters on it would hide the others."""
    ids = list(dict.fromkeys(str(row[0]) for row in rows))
    commons = load_common_names(conn, ids)
    main_names = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        pairs = {}
        for dso, catalogue, designation in conn.execute(
                f"SELECT dsodetailid, catalogue, designation FROM cataloguenr WHERE catalogue <> 'NAME' "
                f"AND dsodetailid IN ({','.join('?' * len(chunk))})", chunk):
            pairs.setdefault(str(dso), []).append((catalogue, str(designation)))
        main_names.update({dso: preferred_designation(p) for dso, p in pairs.items()})
    # The aliases each object matched by, for one entry per object
    matched = {}
    for dso, catalogue, designation in rows:
        if catalogue == "NAME":
            matched.setdefault(str(dso), []).append(str(designation))
    typed = (typed or "").strip().lower()
    labels, seen = [], set()
    for dso, catalogue, designation in rows:
        dso = str(dso)
        if catalogue == "NAME":
            if ("name", dso) in seen:
                continue
            seen.add(("name", dso))
            common = commons.get(dso)
            if not common or typed not in common.lower():
                aliases = matched[dso]
                common = pick_common_name(aliases) or min(aliases, key=len)
            main = main_names.get(dso)
            label = f"{main} ({common})" if main else common
        else:
            label = with_common_name(f"{catalogue} {designation}", commons.get(dso))
        if label not in seen:
            seen.add(label)
            labels.append((label, dso))
    return labels
