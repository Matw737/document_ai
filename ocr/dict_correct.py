"""Konserwatywna korekta słownikowa wyjścia OCR (poziom słowa).

Zasada: lepiej NIE poprawić niż zepsuć. Poprawiamy tylko, gdy:
  - token składa się wyłącznie z liter (żadnych cyfr, kodów, IBAN-ów, NIP-ów),
  - nie ma go w słowniku,
  - istnieje DOKŁADNIE JEDEN kandydat w odległości edycyjnej <= max_dist,
  - nie wygląda na skrót (wersaliki <= 4 znaki) ani na nazwę własną
    (słowo z wielkiej litery poprawiamy tylko do słowa, które w treningu
    też bywało pisane z wielkiej litery).
Słownik budujesz z etykiet treningowych (czysty tekst z fitz).
"""
import re
from collections import Counter, defaultdict

try:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    _HAS_RF = True
except ImportError:          # działa też bez rapidfuzz (wolniej)
    _HAS_RF = False


def _lev(a, b, cutoff):
    """Odległość Levenshteina z odcięciem; zwraca cutoff+1, gdy większa."""
    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cutoff:
            return cutoff + 1
        prev = cur
    return prev[-1]


def _nearest(low, pool, d):
    """Dwa najbliższe słowa z puli w odległości <= d: [(słowo, dystans), ...]."""
    if _HAS_RF:
        res = process.extract(low, pool, scorer=Levenshtein.distance,
                              limit=2, score_cutoff=d)
        return [(c, int(dist)) for c, dist, _ in res]
    scored = [(w, _lev(low, w, d)) for w in pool]
    scored = sorted((s for s in scored if s[1] <= d), key=lambda s: s[1])
    return scored[:2]


_EDGE = re.compile(r"^([^\w§]*)(.*?)([^\w§]*)$", re.S)
_NINE_IN_WORD = re.compile(r"(?<=[^\W\d_])9(?=[^\W\d_])")
_PARAGRAPH_L = re.compile(r"^§l(?=[.\s]|$)")

# --- początek zdania (ta sama logika jest w decoder.py: fix_sentence_case) ---
_ABBREV = {"ul.", "al.", "pl.", "os.", "im.", "sp.", "ust.", "pkt.", "art.", "tel.", "np.",
           "tzw.", "godz.", "ks.", "dr", "inż.", "mgr", "prof.", "św.", "tj.", "tzn.",
           "ww.", "ok.", "zob.", "por.", "poz.", "str.", "zał.", "wg.", "ew.", "m.in.",
           "itd.", "itp."}
_DOTTED_ABBR = re.compile(r"^(?:[^\W\d_]\.){2,}$")        # o.o., s.a.


def is_sentence_start(prev):
    if prev is None:
        return True
    p = prev.lower().rstrip('"”»)]')
    if not p or p[-1] not in ".!?:":
        return False
    return not (p in _ABBREV or _DOTTED_ABBR.match(p))


def fix_misc(word):
    """Tanie reguły stałe: 9 między literami -> g, '§l' -> '§1'."""
    word = _NINE_IN_WORD.sub("g", word)
    word = _PARAGRAPH_L.sub("§1", word)
    return word


class DictCorrector:
    def __init__(self, texts=(), min_len=5, short_min_freq=30,
                 max_dist=1, long_len=9, protected=(),
                 freq=None, cap_seen=None,
                 low_mid=None, cap_mid=None, defined=None):
        """
        texts          iterowalny zbiór tekstów (np. r["text"] z train_records)
        min_len        poniżej tej długości poprawiamy tylko do bardzo częstych słów
        short_min_freq minimalna częstość kandydata dla słów krótkich (3-4 litery)
        max_dist       dopuszczalna odległość dla słów < long_len liter
        long_len       od tej długości dopuszczamy odległość max_dist + 1
        protected      słowa (małymi literami), których nigdy nie ruszamy
        """
        self.min_len = min_len
        self.short_min_freq = short_min_freq
        self.max_dist = max_dist
        self.long_len = long_len
        self.protected = {p.lower() for p in protected}

        self.freq = Counter(freq or {})
        self.cap_seen = set(cap_seen or ())
        # low_mid / cap_mid: ile razy słowo stało w ŚRODKU zdania małą / wielką literą
        # defined: słowa pisane wielką literą w środku zdania w umowach (terminy
        #          zdefiniowane: Zamawiający, Wykonawca, Strony...) - nigdy ich nie zmniejszamy
        self.low_mid = Counter(low_mid or {})
        self.cap_mid = Counter(cap_mid or {})
        self.defined = set(defined or ())
        for t in texts:
            prev = None
            for tok in t.split():
                core = _EDGE.match(tok).group(2)
                if core.isalpha():
                    low = core.lower()
                    self.freq[low] += 1
                    if core[0].isupper() and core[1:].islower():
                        self.cap_seen.add(low)
                    if len(core) >= 2 and not is_sentence_start(prev):
                        if core.islower():
                            self.low_mid[low] += 1
                        elif core[0].isupper() and core[1:].islower():
                            self.cap_mid[low] += 1
                prev = tok

        self.by_len = defaultdict(list)
        for w in self.freq:
            self.by_len[len(w)].append(w)

        self.log = []   # (przed, po) - do audytu

    def _case_like(self, core, cand):
        if core.isupper():
            return cand.upper()
        if core[0].isupper():
            return cand.capitalize()
        return cand

    def _correct_core(self, core):
        if not core or not core.isalpha():
            return core
        low = core.lower()
        n = len(core)
        if low in self.freq or low in self.protected:
            return core
        is_upper = core.isupper()
        is_cap = core[0].isupper() and core[1:].islower()
        if is_upper and n <= 4:          # skróty: NIP, KRS, ABC, ODU...
            return core
        if n < 3:
            return core

        d = self.max_dist + (1 if n >= self.long_len else 0)
        pool = []
        for L in range(n - d, n + d + 1):
            pool.extend(self.by_len.get(L, ()))
        if not pool:
            return core

        res = _nearest(low, pool, d)
        if not res:
            return core
        if len(res) > 1 and res[1][1] == res[0][1]:   # remis -> nie zgadujemy
            return core
        cand, dist = res[0]

        if n < self.min_len and self.freq[cand] < self.short_min_freq:
            return core
        if is_cap and cand not in self.cap_seen:       # ochrona nazw własnych
            return core
        return self._case_like(core, cand)

    def correct(self, word):
        m = _EDGE.match(word)
        pre, core, post = m.group(1), m.group(2), m.group(3)
        new = self._correct_core(core)
        if new != core:
            self.log.append((core, new))
        return pre + new + post

    # --- słownik jako plik: budujesz w notatniku, ładujesz w aplikacji ---
    def to_json(self, path):
        import json
        keep = {w for w, c in self.low_mid.items() if c >= 3}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"freq": dict(self.freq),
                       "cap_seen": sorted(self.cap_seen),
                       "low_mid": {w: self.low_mid[w] for w in keep},
                       "cap_mid": {w: self.cap_mid[w] for w in keep if self.cap_mid.get(w)},
                       "defined": sorted(self.defined)},
                      f, ensure_ascii=False)

    @classmethod
    def from_json(cls, path, **kwargs):
        import json
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(freq=d["freq"], cap_seen=d["cap_seen"],
                   low_mid=d.get("low_mid"), cap_mid=d.get("cap_mid"),
                   defined=d.get("defined"), **kwargs)