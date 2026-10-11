"""Dekoder CTC + poprawki po dekodowaniu (wielkość liter, cyfry, słownik).

Pliki obok tego modułu:
  alphabet.py   (idx_to_char_v3)
  slownik.json  (opcjonalny; bez niego dekoder działa bez korekty słownikowej)

Kolejność poprawek na każdym tokenie:
  fix_word  ->  fix_misc  ->  słownik
"""
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import torch

from .alphabet import idx_to_char_v3

try:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    _HAS_RF = True
except ImportError:
    _HAS_RF = False


# ----------------------------------------------------------------------
# 1. Wielkość liter i cyfry
# ----------------------------------------------------------------------
# litery, których mała i wielka forma wyglądają prawie tak samo
AMBIG = set("oOsScCkKpPuUvVwWxXzZżŻśŚźŹćĆ")
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"})


def fix_case(word):
    """Ujednolica wielkość liter w słowie na podstawie liter jednoznacznych.
    Pierwsza LITERA (nie pierwszy znak: '.Zamawiający"') zostaje bez zmian."""
    first = next((i for i, c in enumerate(word) if c.isalpha()), None)
    if first is None:
        return word
    head, tail = word[:first + 1], word[first + 1:]
    body = [c for c in tail if c.isalpha() and c not in AMBIG]
    n = len(body)
    if n < 2:
        return word
    ratio = sum(c.isupper() for c in body) / n
    if ratio <= 0.2 or (n >= 3 and ratio <= 0.3):          # słowo małe / Capitalized
        return head + tail.lower()
    if ratio >= 0.8 or (n >= 3 and ratio >= 0.7):          # słowo wersalikami
        return head + "".join(c.upper() if c in AMBIG else c for c in tail)
    return word


def fix_digits(word):
    digits = sum(c.isdigit() for c in word)
    # litery, które NIE są typowymi pomyłkami cyfr
    other_letters = sum(c.isalpha() and c not in "OolI" for c in word)
    if digits >= 2 and other_letters == 0:
        return word.translate(DIGIT_FIX)
    return word


def fix_word(word):
    if not word:
        return word
    digits = sum(c.isdigit() for c in word)
    letters = sum(c.isalpha() for c in word)
    if digits >= 2 and digits >= letters:
        return fix_digits(word)
    return fix_case(word)


# ----------------------------------------------------------------------
# 2. Tanie reguły stałe
# ----------------------------------------------------------------------
_EDGE = re.compile(r"^([^\w§]*)(.*?)([^\w§]*)$", re.S)
_NINE_IN_WORD = re.compile(r"(?<=[^\W\d_])9(?=[^\W\d_])")
_PARAGRAPH_L = re.compile(r"^§l(?=[.\s]|$)")


def _fix_zero_in_word(word):
    """'0' przy literach w tokenie bez innych cyfr: BISTR0 -> BISTRO, P0NOWNIE -> PONOWNIE."""
    if "0" not in word:
        return word
    letters = sum(c.isalpha() for c in word)
    other_digits = sum(c.isdigit() and c != "0" for c in word)
    if letters < 3 or other_digits:
        return word
    chars = list(word)
    for i, c in enumerate(chars):
        if c != "0":
            continue
        nb = [chars[j] for j in (i - 1, i + 1) if 0 <= j < len(chars) and chars[j].isalpha()]
        if not nb:
            continue
        chars[i] = "O" if all(x.isupper() for x in nb) else "o"
    return "".join(chars)


def fix_misc(word):
    """9 między literami -> g;  '§l' -> '§1';  0 w słowie z literami -> o/O."""
    word = _NINE_IN_WORD.sub("g", word)
    word = _PARAGRAPH_L.sub("§1", word)
    word = _fix_zero_in_word(word)
    return word


# ----------------------------------------------------------------------
# 3. Korekta słownikowa (konserwatywna)
# ----------------------------------------------------------------------
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


_NEAR_LIMIT = 12


def _nearest(low, pool, d, limit=_NEAR_LIMIT):
    """Najbliższe słowa z puli w odległości <= d, posortowane: [(słowo, dystans), ...]."""
    if _HAS_RF:
        res = process.extract(low, pool, scorer=Levenshtein.distance,
                              limit=limit, score_cutoff=d)
        return [(c, int(dist)) for c, dist, _ in res]
    scored = [(w, _lev(low, w, d)) for w in pool]
    scored = sorted((s for s in scored if s[1] <= d), key=lambda s: s[1])
    return scored[:limit]


_VOW = set("aąeęioóuy")


def _only_inflection_change(core, cand):
    """Długie słowo różniące się tylko samogłoską w ostatnich 1-2 znakach
    (kondygnacja/kondygnacji) to najpewniej inna poprawna forma, nie błąd OCR."""
    if len(core) < 5 or len(core) != len(cand):
        return False
    c = core.lower()
    diff = [i for i, (a, b) in enumerate(zip(c, cand)) if a != b]
    return bool(diff) and all(i >= len(c) - 2 and c[i] in _VOW and cand[i] in _VOW
                              for i in diff)


def _suffix_variant(low, cand):
    """Jedno słowo to drugie z dopisaną/ucinaną końcówką (zakończy/zakończyć)."""
    return (len(low) >= 6 and abs(len(low) - len(cand)) <= 2
            and (cand.startswith(low) or low.startswith(cand)))


_COMPOUND = re.compile(r"^[^\W\d_]+(?:[/\-–—][^\W\d_]+)+$")
_COMPOUND_SPLIT = re.compile(r"([/\-–—])")


class _Dictionary:
    def __init__(self, freq, cap_seen, min_len=5, short_min_freq=30,
                 max_dist=1, long_len=9, tie_ratio=10, protected=(),
                 low_mid=None, cap_mid=None, defined=None):
        self.freq = freq
        self.cap_seen = set(cap_seen)
        self.low_mid = low_mid or {}        # słowo -> ile razy małą literą w środku zdania
        self.cap_mid = cap_mid or {}        # słowo -> ile razy wielką literą w środku zdania
        self.defined = set(defined or ())   # terminy zdefiniowane (Zamawiający, Strony...)
        self.min_len = min_len
        self.short_min_freq = short_min_freq
        self.max_dist = max_dist
        self.long_len = long_len
        self.tie_ratio = tie_ratio
        self.protected = {p.lower() for p in protected}
        self.by_len = defaultdict(list)
        for w in freq:
            self.by_len[len(w)].append(w)
        self._cache = {}

    @classmethod
    def from_json(cls, path, **kwargs):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["freq"], d["cap_seen"], low_mid=d.get("low_mid"),
                   cap_mid=d.get("cap_mid"), defined=d.get("defined"), **kwargs)

    @staticmethod
    def _case_like(core, cand):
        if core.isupper():
            return cand.upper()
        if core[0].isupper():
            return cand.capitalize()
        return cand

    @staticmethod
    def _normalize_known_case(core):
        """Znane słowo z mieszaną wielkością (OrAZ) -> pierwsza litera + małe."""
        if core.islower() or core.isupper():
            return core
        if core[0].isupper() and core[1:].islower():
            return core
        return core[0] + core[1:].lower()

    def _compute(self, core):
        if not core or not core.isalpha():
            return core
        low = core.lower()
        n = len(core)
        if low in self.protected:
            return core
        if low in self.freq:
            return self._normalize_known_case(core)
        is_upper = core.isupper()
        is_cap = core[0].isupper() and core[1:].islower()
        if is_upper and n <= 4:          # skróty: NIP, KRS, ABC...
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
        best = res[0][1]
        if len(res) == _NEAR_LIMIT and res[-1][1] == best:
            return core                                # zbyt wielu kandydatów
        tied = sorted((c for c, dist in res if dist == best),
                      key=lambda w: -self.freq[w])
        if len(tied) > 1 and self.freq[tied[0]] < self.tie_ratio * self.freq[tied[1]]:
            return core                                # remis bez wyraźnego faworyta
        cand = tied[0]

        if n < self.min_len and self.freq[cand] < self.short_min_freq:
            return core
        if is_cap and cand not in self.cap_seen:       # ochrona nazw własnych
            return core
        if _only_inflection_change(core, cand):        # inna poprawna forma
            return core
        if _suffix_variant(low, cand):                 # inna poprawna forma
            return core
        return self._case_like(core, cand)

    def _correct_core(self, core):
        hit = self._cache.get(core)
        if hit is None:
            if len(self._cache) > 100_000:
                self._cache.clear()
            hit = self._compute(core)
            self._cache[core] = hit
        return hit

    def correct(self, word):
        m = _EDGE.match(word)
        pre, core, post = m.group(1), m.group(2), m.group(3)
        if _COMPOUND.match(core):                      # towaru/usugi, biało-czerwony
            parts = _COMPOUND_SPLIT.split(core)
            core = "".join(p if i % 2 else self._correct_core(p)
                           for i, p in enumerate(parts))
        else:
            core = self._correct_core(core)
        return pre + core + post


_DICT_PATH = Path(__file__).with_name("slownik.json")
_dictionary = None        # None = jeszcze nie wczytany, False = niedostępny


def _get_dictionary():
    global _dictionary
    if _dictionary is None:
        if _DICT_PATH.exists():
            _dictionary = _Dictionary.from_json(_DICT_PATH)
        else:
            warnings.warn(f"Brak {_DICT_PATH.name} - dekoder działa bez korekty słownikowej.")
            _dictionary = False
    return _dictionary or None


# ----------------------------------------------------------------------
# 4. Wielka pierwsza litera w środku zdania (poziom tekstu strony)
#    Użycie: text = fix_sentence_case(text)   po złożeniu linii strony
# ----------------------------------------------------------------------
# skróty, po których kropka NIE kończy zdania (ta sama lista jest w dict_correct.py)
_ABBREV = {"ul.", "al.", "pl.", "os.", "im.", "sp.", "ust.", "pkt.", "art.", "tel.", "np.",
           "tzw.", "godz.", "ks.", "dr", "inż.", "mgr", "prof.", "św.", "tj.", "tzn.",
           "ww.", "ok.", "zob.", "por.", "poz.", "str.", "zał.", "wg.", "ew.", "m.in.",
           "itd.", "itp."}
# po tych skrótach idzie nazwa własna (ul. Zielona, dr Kos): nie ruszamy
_NAME_PREFIX = {"ul.", "al.", "pl.", "os.", "im.", "dr", "prof.", "inż.", "mgr", "ks.", "św."}
_DOTTED_ABBR = re.compile(r"^(?:[^\W\d_]\.){2,}$")        # o.o., s.a.
_OPENERS = set('„"“«([')
_SINGLE_LOWER = set("WZOUI")                              # W Krakowie, Z siedzibą, O podatek...
# słowa funkcyjne: wielka litera w środku zdania jest w nich zawsze błędem
_FUNCTION_LOWER = {"za", "ze", "we", "do", "od", "po", "pod", "przez", "przy", "przed",
                   "poza", "pomiędzy", "między", "nad", "bez", "dla", "oraz", "wraz",
                   "wobec", "wśród", "zgodnie", "według", "poprzez", "pomimo", "podczas",
                   "lub", "albo", "ani", "jeżeli", "jeśli", "chyba", "który", "która",
                   "które", "których", "której", "którego", "którym"}
_WS = re.compile(r"(\s+)")
_MIN_LINE_TOKENS = 6      # krótsza linia (nagłówek, pozycja listy) kończy zdanie


def _is_sentence_start(prev):
    if prev is None:
        return True
    p = prev.lower().rstrip('"”»)]')
    if not p or p[-1] not in ".!?:":
        return False
    return not (p in _ABBREV or _DOTTED_ABBR.match(p))


def _lower_first(tok, prev, d):
    m = _EDGE.match(tok)
    pre, core, post = m.group(1), m.group(2), m.group(3)
    if not core or not core.isalpha() or not core[0].isupper():
        return tok
    if any(ch in _OPENERS for ch in pre) or _is_sentence_start(prev):
        return tok
    low = core.lower()
    lowered = pre + low + post
    if len(core) == 1:
        return lowered if core in _SINGLE_LOWER else tok
    if not core[1:].islower():                 # WERSALIKI / mieszane: nie ruszamy
        return tok
    if low in _FUNCTION_LOWER:
        return lowered
    if d is None or core[0] not in AMBIG:
        return tok
    if prev is not None:
        if prev.lower() in _NAME_PREFIX:       # ul. Zielona
            return tok
        pcore = _EDGE.match(prev).group(2)
        pl = pcore.lower()
        if (pcore.isalpha() and pcore[0].isupper() and pcore[1:].islower()
                and pl not in d.defined and d.low_mid.get(pl, 0) < 3):
            return tok                         # po imieniu: Jan Kos, Anna Piekarz
    if low in d.defined or low not in d.freq:  # termin zdefiniowany / nieznane słowo
        return tok
    lm = d.low_mid.get(low, 0)
    if lm >= 3 and d.cap_mid.get(low, 0) * 10 <= lm:
        return lowered
    return tok


def fix_sentence_case(text):
    """Zamienia na małą pierwszą literę słów wielką w środku zdania
    (Wynosi, Za, W, O, Odpowiada...), z ochroną nazw własnych i terminów
    zdefiniowanych (Zamawiający, Wykonawca, Strony, Umowy)."""
    d = _get_dictionary()
    out = []
    prev_last, prev_len = None, 0
    for line in text.split("\n"):
        parts = _WS.split(line)
        prev = prev_last if prev_len >= _MIN_LINE_TOKENS else None
        n_tok = 0
        for i in range(0, len(parts), 2):          # tokeny na indeksach parzystych
            tok = parts[i]
            if not tok:
                continue
            parts[i] = _lower_first(tok, prev, d)
            prev = parts[i]          # kontekst z już poprawionego słowa
            n_tok += 1
        if n_tok:
            prev_last, prev_len = prev, n_tok
        else:                                      # pusta linia = koniec akapitu
            prev_last, prev_len = None, 0
        out.append("".join(parts))
    return "\n".join(out)


# ----------------------------------------------------------------------
# 5. Dekodowanie
# ----------------------------------------------------------------------
def postprocess_word(word):
    word = fix_misc(fix_word(word))
    d = _get_dictionary()
    return d.correct(word) if d else word


def postprocess_text(text):
    # crop może zawierać kilka słów rozdzielonych spacją
    return " ".join(postprocess_word(t) for t in text.split(" "))


def greedy_decode(logits, input_lengths, postprocess=True):
    predictions = torch.argmax(logits, dim=2).cpu().tolist()
    lengths = [int(x) for x in input_lengths]

    texts = []
    for seq, n in zip(predictions, lengths):
        out, prev = [], -1
        for s in seq[:n]:
            if s != prev and s != 0:
                out.append(idx_to_char_v3[s])
            prev = s
        raw = "".join(out)
        texts.append(postprocess_text(raw) if postprocess else raw)
    return texts