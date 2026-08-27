from collections import Counter
from pathlib import Path

import torch

# ============================================================================
# Definición de alfabetos
# ============================================================================
#
# Alfabeto bilingüe explícito para tu dataset:
# - espacio
# - letras minúsculas y mayúsculas
# - vocales acentuadas españolas + diéresis (ü/Ü)
# - ñ/Ñ
# - dígitos
# - puntuación observada en los splits y signos españoles comunes
#
# El orden define los índices de los caracteres (0-indexado). Este alfabeto
# debe ser coherente con:
#   - build_hdf5.py (codificación de labels en HDF5)
#   - configs/*.yml  (training.n_class, GenModel.n_class, DiscModel.n_class,
#                     HFDiscModel.n_class, OcrModel.n_class)
# ============================================================================

def _unique_chars(*chunks):
    seen = set()
    ordered = []
    for chunk in chunks:
        for char in chunk:
            if char in seen:
                continue
            seen.add(char)
            ordered.append(char)
    return "".join(ordered)


SPANISH_ALPHABET = _unique_chars(
    " ",  # espacio
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "áéíóúüÁÉÍÓÚÜ",
    "ñÑ",
    "0123456789",
    ".,;:¿?¡!\"'#&()*+-/",
)

Alphabets = {
    'custom': SPANISH_ALPHABET
}


def get_alphabet_stats(alphabet_key="custom"):
    alphabet = Alphabets[alphabet_key]
    return {
        "alphabet": alphabet,
        "vocab_size": len(alphabet),
        "unique_chars": len(set(alphabet)),
        "has_duplicates": len(alphabet) != len(set(alphabet)),
    }


def collect_characters_from_split(split_file):
    chars = Counter()
    split_path = Path(split_file)
    with split_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                _, label = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Línea inválida en {split_file}:{line_no}: {line!r}") from exc
            chars.update(label)
    return chars


def validate_split_alphabet(split_file, alphabet_key="custom"):
    alphabet = set(Alphabets[alphabet_key])
    observed = collect_characters_from_split(split_file)
    unknown = {ch: count for ch, count in observed.items() if ch not in alphabet}
    return {
        "unknown": unknown,
        "observed": observed,
    }


class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        El `blank` de CTC NO forma parte de este alfabeto.
        Convención del repo:
          - caracteres: índices 0..len(alphabet)-1
          - blank (CTC): índice len(alphabet)  (p.ej. OcrModel.n_class = len(alphabet)+1)
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet_key, ignore_case=False):
        alphabet = Alphabets[alphabet_key]
        # print(alphabet)
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i

    def encode(self, text, max_len=None):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            encoded = []
            for char in text:
                c = char.lower() if self._ignore_case else char
                if c not in self.dict:
                    raise ValueError(f"Caracter fuera de alfabeto: {repr(char)}")
                encoded.append(self.dict[c])
            return encoded

        if not isinstance(text, (list, tuple)):
            raise TypeError(f"text must be str, list, or tuple. Got: {type(text)}")

        length = []
        results = []
        for item in text:
            encoded_word = []
            for char in item:
                c = char.lower() if self._ignore_case else char
                if c not in self.dict:
                    raise ValueError(f"Caracter fuera de alfabeto en texto {repr(item)}: {repr(char)}")
                encoded_word.append(self.dict[c])
            if not encoded_word:
                encoded_word = [self.dict.get(' ', 0)]
            results.append(encoded_word)
            length.append(len(encoded_word))

        labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(r) for r in results], batch_first=True)
        lengths = torch.IntTensor(length)

        if max_len is not None and max_len > labels.size(-1):
            pad_labels = torch.zeros((labels.size(0), max_len)).long()
            pad_labels[:, :labels.size(-1)] = labels
            labels = pad_labels

        return labels, lengths

    def decode(self, t, length=None, raw=True):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Returns:
            text (str or list of str): texts to convert.
        """
        def nonzero_count(x):
            return len(x.nonzero())

        if isinstance(t, list):
            t = torch.IntTensor(t)
            length = torch.IntTensor([len(t)])
        elif length is None:
            length = torch.IntTensor([nonzero_count(t)])

        if length.numel() == 1:
            length = length[0]
            if not raw:
                assert nonzero_count(t) == length, "{} text with length: {} does not match declared length: {}".\
                                                    format(t, nonzero_count(t), length)
            if raw:
                # Decodificar por longitud (el índice 0 puede ser un carácter válido, ej. espacio)
                l = int(length.item() if hasattr(length, 'item') else length)
                if t.dim() == 2:
                    t = t[0]
                seq = t[:l]
                return ''.join([self.alphabet[int(i)] for i in seq if 0 <= int(i) < len(self.alphabet)])
            else:
                char_list = []
                if t.dim() == 2:
                    t = t[0]
                blank_idx = len(self.alphabet)
                for i in range(length):
                    ti = int(t[i])
                    if ti == blank_idx:
                        continue
                    if i > 0 and int(t[i - 1]) == ti:
                        continue
                    if 0 <= ti < len(self.alphabet):
                        char_list.append(self.alphabet[ti])
                return ''.join(char_list)
        else:
            # batch mode: usar longitudes proporcionadas (0 puede ser espacio)
            texts = []
            for i in range(length.numel()):
                l = length[i]
                l = int(l.item() if hasattr(l, 'item') else l)
                texts.append(self.decode(t[i, :l], torch.IntTensor([l]), raw=raw))
            return texts


def get_true_alphabet(name):
    tag = '_'.join(name.split('_')[:2])
    return Alphabets[tag]


def sanitize_word(word, true_alphabet, max_length=20, ignore_case=True):
    """Deja solo caracteres del alfabeto; si queda vacía o demasiado larga, devuelve None."""
    if not word or not isinstance(word, str):
        return None
    if ignore_case:
        word = word.lower()
    cleaned = ''.join(c for c in word if c in true_alphabet)
    if len(cleaned) < 1 or len(cleaned) >= max_length:
        return None
    return cleaned


def get_lexicon(path, true_alphabet, max_length=20, ignore_case=True):
    words = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 2:
                    continue

                word = ''.join(ch for ch in line if ch in true_alphabet)
                if len(word) != len(line) or len(word) >= max_length:
                    continue
                if ignore_case:
                    word = word.lower()
                words.append(word)
    except FileNotFoundError as e:
        print(e)
    return words


def word_capitalize(word):
    word = list(word)
    # Preserve Spanish diacritics when capitalizing (e.g., á->Á, ñ->Ñ).
    word[0] = word[0].upper()
    word = ''.join(word)
    return word
