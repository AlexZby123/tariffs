# -*- coding: utf-8 -*-
"""
Klient LLM dla Bosch Model Farm.

Obsluguje dwa tryby (config: provider):
  - bosch : Bosch Model Farm (endpoint w stylu Azure OpenAI, auth: Bearer token).
            base_url + /openai/deployments/{deployment}/chat/completions?api-version=...
  - mock  : offline, bez tokenu - heurystyka slow kluczowych (test pipeline)

Jedyny publiczny interfejs uzywany przez agentow:
    client = LLMClient.from_config(cfg)
    obj = client.chat_json(system_prompt, user_prompt)   # zwraca dict (JSON)
    txt = client.chat_text(system_prompt, user_prompt)   # zwraca str
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

KATALOG = Path(__file__).parent


# ============================================================
#  Wczytanie configu
# ============================================================

def wczytaj_config(sciezka: Path | None = None) -> dict:
    sciezka = sciezka or (KATALOG / "config.yaml")
    if not sciezka.exists():
        raise FileNotFoundError(
            f"Brak {sciezka.name}. Skopiuj config.example.yaml -> config.yaml "
            f"i wklej token Model Farm."
        )
    with open(sciezka, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
#  Wspolny parser JSON (LLM czasem owija w ```json ... ```)
# ============================================================

def _skanuj_obiekty(txt: str) -> list[str]:
    """Zwraca wszystkie KOMPLETNE, zbalansowane obiekty {...} (pomijajac nawiasy
    w stringach). Uzywane do ratowania UCIETEJ odpowiedzi (gadatliwe/rozumujace
    modele przekraczaja budzet tokenow) - odzyskujemy wszystkie pelne pozycje."""
    out, depth, start = [], 0, None
    instr = esc = False
    for k, ch in enumerate(txt):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            if depth == 0:
                start = k
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(txt[start:k + 1])
                start = None
    return out


def wyciagnij_json(txt: str) -> Any:
    txt = txt.strip()
    # usun ewentualne ogrodzenie ```json
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
        txt = re.sub(r"\n?```$", "", txt).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # sprobuj wyciac najwiekszy blok { ... } lub [ ... ]
        for otw, zam in (("{", "}"), ("[", "]")):
            i, j = txt.find(otw), txt.rfind(zam)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(txt[i:j + 1])
                except json.JSONDecodeError:
                    continue
        # RATUNEK: odzyskaj kompletne obiekty z ucietej listy (assignments)
        parsed = []
        for o in _skanuj_obiekty(txt):
            try:
                parsed.append(json.loads(o))
            except json.JSONDecodeError:
                continue
        # tylko obiekty z 'id' (pozycje assignments) - unikamy smieci z odkrywania
        poz = [p for p in parsed if isinstance(p, dict) and "id" in p]
        if poz:
            return {"assignments": poz}
        raise ValueError(f"Nie udalo sie sparsowac JSON z odpowiedzi LLM:\n{txt[:500]}")


# ============================================================
#  Klient
# ============================================================

class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = cfg.get("provider", "mock")
        self.model = cfg.get("model", "gpt-4o")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 3000))
        self.reasoning_effort = cfg.get("reasoning_effort")  # GPT-5/o-series: minimal|low|medium|high
        self.max_retries = int(cfg.get("max_retries", 5))
        self.timeout = float(cfg.get("request_timeout", 90))
        self._client = None
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        if self.provider != "mock":
            self._init_sdk()

    @classmethod
    def from_config(cls, sciezka: Path | None = None) -> "LLMClient":
        return cls(wczytaj_config(sciezka))

    # --- inicjalizacja SDK ---
    def _init_sdk(self) -> None:
        """Bosch Model Farm: endpoint w stylu Azure OpenAI, ale autoryzacja
        naglowkiem 'Authorization: Bearer <token>' (nie 'api-key').
        Dlatego uzywamy zwyklego klienta OpenAI, ktory wysyla Bearer,
        z base_url zawierajacym deployment i api-version jako default_query -
        1:1 z dokumentacyjnym curl.
        """
        key = self.cfg.get("api_key", "")
        if not key or "PASTE" in key:
            raise ValueError("Ustaw api_key (token Bosch Model Farm) w config.yaml")

        from openai import OpenAI
        base = str(self.cfg.get("base_url", "https://aoai-farm.bosch-temp.com/api")).rstrip("/")
        deployment = self.cfg.get("deployment") or self.cfg.get("model")
        if not deployment:
            raise ValueError("Ustaw 'deployment' (nazwa deploymentu w Model Farm) w config.yaml")
        api_version = self.cfg.get("api_version", "2025-04-01-preview")

        # 'model' w ciele zapytania: dla OpenAI = deployment; dla Gemini np. 'google/gemini-2.5-flash-lite'
        self.model = self.cfg.get("model") or deployment
        self._client = OpenAI(
            api_key=key,
            base_url=f"{base}/openai/deployments/{deployment}",
            default_query={"api-version": api_version},
            timeout=self.timeout,
            max_retries=0,  # retry z backoffem obslugujemy sami
        )

    # --- polaczenie testowe ---
    def check(self) -> str:
        if self.provider == "mock":
            return "mock OK (offline, bez tokenu)"
        odp = self.chat_text("You are a healthcheck.", "Reply with the single word: OK")
        return f"{self.provider} / {self.model} -> {odp.strip()[:40]}"

    def chat_text(self, system: str, user: Any) -> str:
        if self.provider == "mock":
            return _mock_odpowiedz(system, user if isinstance(user, str) else str(user))
        return self._chat_with_retry(system, user, json_mode=False)

    def chat_json(self, system: str, user: Any) -> Any:
        if self.provider == "mock":
            return wyciagnij_json(_mock_odpowiedz(system, user if isinstance(user, str) else str(user), json_mode=True))
        txt = self._chat_with_retry(system, user, json_mode=True)
        return wyciagnij_json(txt)

    def _chat_with_retry(self, system: str, user: Any, json_mode: bool) -> str:
        # flagi adaptowane przy bledach parametrow (rozne modele maja rozne wymogi:
        # GPT-5/o-series: 'max_completion_tokens' zamiast 'max_tokens', temperatura=1)
        use_json = json_mode
        use_temp = True
        token_param = "max_tokens"
        use_reasoning = bool(self.reasoning_effort)
        max_tok = self.max_tokens
        bumped = False

        ostatni_blad = None
        proba = 0
        while proba < self.max_retries:
            kwargs: dict = dict(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            kwargs[token_param] = max_tok
            if use_temp:
                kwargs["temperature"] = self.temperature
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            if use_reasoning:
                kwargs["reasoning_effort"] = self.reasoning_effort
            try:
                resp = self._client.chat.completions.create(**kwargs)
                self.calls += 1
                if getattr(resp, "usage", None):
                    self.tokens_in += resp.usage.prompt_tokens or 0
                    self.tokens_out += resp.usage.completion_tokens or 0
                content = resp.choices[0].message.content or ""
                # modele rozumujace (GPT-5/o) potrafia zjesc caly budzet na reasoning
                # i zwrocic pusta tresc -> zwieksz budzet raz i ponow (bez kosztu proby).
                # Jesli po bumpie nadal pusto, zwracamy "" -> wyciagnij_json rzuci blad,
                # ktory run_clustering lapie per-partia (te PN dostana INNE, reszta biegu OK).
                if not content.strip() and not bumped:
                    max_tok = max(max_tok * 4, 16000); bumped = True; continue
                return content
            except Exception as e:  # noqa: BLE001
                ostatni_blad = e
                msg = str(e).lower()
                # adaptacje parametrow - bez zuzywania proby (natychmiast ponow)
                if use_json and "response_format" in msg:
                    use_json = False; continue
                if token_param == "max_tokens" and "max_completion_tokens" in msg:
                    token_param = "max_completion_tokens"; continue
                if use_temp and "temperature" in msg:
                    use_temp = False; continue
                if use_reasoning and "reasoning_effort" in msg:
                    use_reasoning = False; continue
                proba += 1
                czekaj = min(2 ** proba, 30)
                print(f"    [retry {proba}/{self.max_retries}] {str(e)[:120]} -> czekam {czekaj}s")
                time.sleep(czekaj)
        raise RuntimeError(f"LLM nieosiagalny po {self.max_retries} probach: {ostatni_blad}")

    def podsumowanie_kosztow(self) -> str:
        return (f"zapytan={self.calls}  tokeny_wej={self.tokens_in}  "
                f"tokeny_wyj={self.tokens_out}")


# ============================================================
#  MOCK - offline heurystyka (do testow bez tokenu)
# ============================================================

# slowo w opisie -> nazwa klastra (styl Rudolfa)
_MOCK_MAPA = [
    ("o-ring", "O-RING"), ("o ring", "O-RING"),
    ("ball joint", "BALL JOINT"), ("ball", "BALL"),
    ("wheel speed sensor", "SENSOR"), ("pressure sensor", "SENSOR"), ("sensor", "SENSOR"),
    ("screw", "SCREWs & NUTs & BOLTs & PINs & STUDs"), ("bolt", "SCREWs & NUTs & BOLTs & PINs & STUDs"),
    ("nut", "SCREWs & NUTs & BOLTs & PINs & STUDs"), ("stud", "SCREWs & NUTs & BOLTs & PINs & STUDs"),
    ("rivet", "SCREWs & NUTs & BOLTs & PINs & STUDs"), ("pin", "SCREWs & NUTs & BOLTs & PINs & STUDs"),
    ("gasket", "SEAL"), ("seal", "SEAL"),
    ("bearing", "BEARING, BRACKET, SUPPORT, NEEDLE, GUIDE RING"),
    ("bracket", "BEARING, BRACKET, SUPPORT, NEEDLE, GUIDE RING"),
    ("bushing", "BUSHING"), ("bush", "BUSHING"),
    ("spring", "SPRING"),
    ("filter", "FILTER"),
    ("clip", "CLIP & CLAMP"), ("clamp", "CLIP & CLAMP"),
    ("piston", "PISTON"),
    ("damper", "DAMPER"),
    ("solenoid", "SOLENOID VALVE"),
    ("valve", "VALVE BODY"),
    ("motor", "MOTOR"),
    ("pump", "PUMP ELEMENT"),
    ("washer", "WASHER"),
    ("ecu", "ECU"),
    ("label", "ADHESIVE FILM, LABEL, STICKER"), ("sticker", "ADHESIVE FILM, LABEL, STICKER"),
    ("cap", "PROTECTION CAP / SLEEVE"), ("sleeve", "PROTECTION CAP / SLEEVE"),
    ("rod", "ROD"),
    ("plug", "SCREW PLUG GROMMET"), ("grommet", "SCREW PLUG GROMMET"),
    ("reservoir", "RESERVOIR CAP"),
    ("armature", "ARMATURE"),
    ("tappet", "TAPPET"),
    ("plate", "PLATE/DISC"), ("disc", "PLATE/DISC"),
    ("housing", "HOUSING"),
    ("hydraulic", "HYDRAULIC UNIT"),
    ("coil", "COIL / MAGNET"), ("magnet", "COIL / MAGNET"),
]
_MOCK_TAXONOMIA = sorted({n for _, n in _MOCK_MAPA} | {"INNE"})


def _mock_klasyfikuj_opis(opis: str) -> str:
    o = opis.lower()
    for slowo, klaster in _MOCK_MAPA:
        if slowo in o:
            return klaster
    return "INNE"


def _mock_odpowiedz(system: str, user: str, json_mode: bool = True) -> str:
    """Rozpoznaje typ zadania po tresci promptu i zwraca sensowny JSON.

    UWAGA kolejnosc: najpierw sprawdzamy wzorzec KLASYFIKACJI ([id] desc="..."),
    bo prompt klasyfikujacy zawiera slowo 'propose' (dla NEW:) i wpadlby
    w galaz odkrywania taksonomii.
    """
    low = user.lower()
    # zadanie: klasyfikacja partii - linie [id] ... desc="..."
    linie = re.findall(r"\[(\d+)\][^\n]*?desc=\"(.*?)\"", user, flags=re.S)
    if linie:
        przypisania = [{"id": int(i), "cluster": _mock_klasyfikuj_opis(opis)}
                       for i, opis in linie]
        return json.dumps({"assignments": przypisania})
    # zadanie: odkrycie / konsolidacja taksonomii
    if "propose" in low or "consolidat" in low or "merge" in low or "cluster" in low:
        return json.dumps({"clusters": _MOCK_TAXONOMIA})
    # healthcheck / inne
    return json.dumps({"ok": True}) if json_mode else "OK"
