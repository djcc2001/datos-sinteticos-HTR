SPANISH_LOWER = list("abcdefghijklmnopqrstuvwxyz")
SPANISH_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPANISH_ACCENTS = [
    "á", "é", "í", "ó", "ú",
    "Á", "É", "Í", "Ó", "Ú",
    "ñ", "Ñ",
    "ü", "Ü",
]
SPANISH_DIGITS = list("0123456789")
SPANISH_PUNCT = [
    ".", ",", ";", ":", "¿", "?", "¡", "!",
    "(", ")", "[", "]", "{", "}", "-", "_",
    "/", "\\", "'", '"', "@", "#", "$", "%", "&",
    "*", "+", "=",
]

# Space is included to support multi-token lines and padding behavior.
SPANISH_CHARSET = (
    SPANISH_LOWER
    + SPANISH_UPPER
    + SPANISH_ACCENTS
    + SPANISH_DIGITS
    + SPANISH_PUNCT
    + [" "]
)

SPANISH_VOCAB_SIZE = len(SPANISH_CHARSET)
