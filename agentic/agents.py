# -*- coding: utf-8 -*-
"""
Agenci LLM do klastrowania czesci (styl Rudolfa).

Trzech wspolpracujacych agentow:
  1. AgentOdkrywajacy (Discovery)  - z probki czesci proponuje liste klastrow
  2. AgentKonsolidujacy (Consolidation) - scala synonimy w czysta taksonomie
  3. AgentKlasyfikujacy (Classifier) - przypisuje kazda czesc do klastra
                                       (moze zaproponowac NOWY, jesli wlaczone)

Filozofia (z notatek Rudolfa):
  - klaster = TYP FUNKCJONALNY czesci (BALL, SENSOR, SEAL, MOTOR...),
  - glowny sygnal = OPIS materialu (+ wiedza dziedzinowa),
  - HS / Material Field to informacja o MATERIALE - pomocnicza, nie decydujaca,
  - przy watpliwosci wybieraj GRUBSZY typ funkcjonalny (nie mnoz podtypow).
"""
from __future__ import annotations

from llm_client import LLMClient

FILOZOFIA = (
    "You cluster automotive/industrial parts for a customs & logistics team at Bosch. "
    "A cluster = the FUNCTIONAL PART TYPE (what the part IS), e.g. BALL, SENSOR, SEAL, "
    "SCREW, MOTOR, SPRING, FILTER, PISTON, BEARING. "
    "Judge mainly from the material DESCRIPTION using real-world/product knowledge "
    "(e.g. 'wheel speed sensor' -> SENSOR, 'input rod tip; ball' -> BALL, 'rubber gasket' -> SEAL). "
    "HS code and material_field describe the MATERIAL (rubber/steel/aluminium) - use them only "
    "as weak hints, they are often noisy. Prefer COARSE functional types; do NOT create many "
    "sub-types. Use SHORT, UPPERCASE, consistent cluster names."
)


# ============================================================
#  1. Agent odkrywajacy taksonomie
# ============================================================

def odkryj_taksonomie(client: LLMClient, karty: list[str]) -> list[str]:
    system = FILOZOFIA
    lista = "\n".join(f"- {k}" for k in karty)
    user = (
        "Propose a concise list of functional part-type CLUSTERS that cover the parts below. "
        "Aim for coarse, non-overlapping categories (target 40-90 clusters for the whole dataset, "
        "here propose what this sample needs). Return JSON: {\"clusters\": [\"NAME1\", ...]}.\n\n"
        f"PARTS SAMPLE:\n{lista}"
    )
    obj = client.chat_json(system, user)
    return _oczysc_liste(obj.get("clusters", []))


# ============================================================
#  2. Agent konsolidujacy
# ============================================================

def skonsoliduj_taksonomie(client: LLMClient, propozycje: list[str]) -> list[str]:
    system = FILOZOFIA
    lista = "\n".join(f"- {k}" for k in propozycje)
    user = (
        "Consolidate/merge this list of proposed cluster names into a single clean taxonomy: "
        "merge synonyms and near-duplicates (e.g. 'BOLTS' + 'SCREWS' -> one), keep names short "
        "and UPPERCASE, remove redundancy, keep coarse functional types. "
        "Return JSON: {\"clusters\": [\"NAME1\", ...]}.\n\n"
        f"PROPOSED NAMES:\n{lista}"
    )
    obj = client.chat_json(system, user)
    wynik = _oczysc_liste(obj.get("clusters", []))
    return wynik or _oczysc_liste(propozycje)


# ============================================================
#  3. Agent klasyfikujacy (partiami)
# ============================================================

def klasyfikuj_partie(client: LLMClient, partia: list[tuple[int, str]],
                      taksonomia: list[str], allow_new: bool) -> dict[int, str]:
    """partia: lista (id, part_card). Zwraca {id: nazwa_klastra}."""
    system = FILOZOFIA
    tax = "\n".join(f"- {k}" for k in taksonomia)
    reguly_new = (
        "If a part truly fits none, use \"NEW: <short name>\" to propose a new cluster."
        if allow_new else
        "You MUST pick exactly one cluster from the taxonomy (choose the closest)."
    )
    # part_card zawsze zwraca gotowe pola (desc="..." | hs6=... | ...) - renderujemy 1:1
    linie = "\n".join(f"[{i}] {card}" for i, card in partia)
    user = (
        "Classify each part into ONE cluster from the taxonomy below. "
        f"{reguly_new}\n"
        "Return JSON: {\"assignments\": [{\"id\": <int>, \"cluster\": \"<NAME>\"}, ...]} "
        "with one entry per input id.\n\n"
        f"TAXONOMY:\n{tax}\n\n"
        f"PARTS:\n{linie}"
    )
    obj = client.chat_json(system, user)
    wynik: dict[int, str] = {}
    for a in obj.get("assignments", []):
        try:
            wynik[int(a["id"])] = str(a["cluster"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
    return wynik


# ============================================================
#  3b. Klasyfikator z PEWNOSCIA i "czego brakuje" (analiza roznic)
# ============================================================

def klasyfikuj_z_pewnoscia(client: LLMClient, partia: list[tuple[int, str]],
                           taksonomia: list[str]) -> dict[int, dict]:
    """Jak klasyfikuj_partie, ale kazda czesc dostaje tez pewnosc i - gdy model
    nie jest pewny - krotkie 'czego brakuje, by zdecydowac'.

    Zwraca {id: {"cluster": str, "confidence": float, "missing": str}}.
    """
    system = FILOZOFIA
    tax = "\n".join(f"- {k}" for k in taksonomia)
    # part_card zawsze zwraca gotowe pola (desc="..." | hs6=... | ...) - renderujemy 1:1
    linie = "\n".join(f"[{i}] {card}" for i, card in partia)
    user = (
        "Classify each part into ONE cluster from the taxonomy below (pick the CLOSEST). "
        "For EACH part also report how sure you are and, if unsure, what is missing.\n"
        "Fields per part:\n"
        '- "cluster": exactly one name from the taxonomy;\n'
        '- "confidence": number 0.0-1.0 (how sure the assignment is);\n'
        '- "missing": if confidence < 0.7, a SHORT phrase naming the ONE piece of info that '
        "would make you sure. Prefer these canonical reasons when they fit: "
        '"function unclear", "material vs function conflict", "description too generic/only a code", '
        '"ambiguous between two clusters", "need drawing/photo", "need dimensions", '
        '"taxonomy has no good fit". Otherwise "".\n'
        'Return JSON: {"assignments":[{"id":<int>,"cluster":"<NAME>","confidence":<float>,'
        '"missing":"<text>"}]} with exactly one entry per input id.\n\n'
        f"TAXONOMY:\n{tax}\n\n"
        f"PARTS:\n{linie}"
    )
    obj = client.chat_json(system, user)
    wynik: dict[int, dict] = {}
    for a in obj.get("assignments", []):
        try:
            i = int(a["id"])
        except (KeyError, ValueError, TypeError):
            continue
        try:
            conf = float(a.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf = 0.0
        wynik[i] = {
            "cluster": str(a.get("cluster", "")).strip(),
            "confidence": max(0.0, min(1.0, conf)),
            "missing": str(a.get("missing", "") or "").strip(),
        }
    return wynik


# ============================================================
#  Pomocnicze
# ============================================================

def _oczysc_liste(lst) -> list[str]:
    out, widziane = [], set()
    for x in lst or []:
        s = str(x).strip().strip('"').upper()
        if s and s not in widziane:
            widziane.add(s)
            out.append(s)
    return out
