"""Unify CamelCase citation keys -> lowercase in Participatory_it_from_bit.tex.

Targets only \cite-family commands' key arguments; does NOT touch the .bib
file or other manuscripts (scope-respecting fix).
"""
import re

FP = "Attention/Participatory_it_from_bit.tex"

REPL = {
    "Friston2010": "friston2010free",
    "Vaswani2017": "vaswani2017attention",
    "Parr2022": "parr2022active",
    "Amari2016": "amari2016information",
    "Nakahara2003": "nakahara2003geometry",
    "Frankel2011": "frankel2011geometry",
}

# Match \cite, \citep, \citet, \citealt, \citeyear, \citenum, \nocite,
# optionally followed by * and 0-2 [..] optional args, then {keys}
CITE_CMD_RE = re.compile(
    r"(\\(?:cite[a-z]*|nocite)\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{)"
    r"([^}]+)"
    r"(\})"
)

count = 0

def replace_keys(match):
    global count
    prefix, keys_str, suffix = match.group(1), match.group(2), match.group(3)
    new_keys = []
    for key in keys_str.split(","):
        ks = key.strip()
        if ks in REPL:
            new_keys.append(REPL[ks])
            count += 1
        else:
            new_keys.append(ks)
    return prefix + ", ".join(new_keys) + suffix


with open(FP, "r", encoding="utf-8") as f:
    src = f.read()
new_src = CITE_CMD_RE.sub(replace_keys, src)
with open(FP, "w", encoding="utf-8") as f:
    f.write(new_src)
print(f"Replacements applied: {count}")
