from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b\S+\.com\b\S*")
EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
TOKEN_RE = re.compile(r"<url>|<email>|<num>|[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    name: str
    lowercase: bool = True
    normalize_unicode: bool = True
    replace_urls: bool = False
    replace_emails: bool = False
    replace_numbers: bool = False
    remove_stopwords: bool = False
    stemming: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PREPROCESSING_VARIANTS: tuple[PreprocessConfig, ...] = (
    PreprocessConfig(name="minimal"),
    PreprocessConfig(
        name="normalized_placeholders",
        replace_urls=True,
        replace_emails=True,
        replace_numbers=True,
    ),
    PreprocessConfig(
        name="normalized_stopwords",
        replace_urls=True,
        replace_emails=True,
        replace_numbers=True,
        remove_stopwords=True,
    ),
    PreprocessConfig(
        name="normalized_stopwords_stem",
        replace_urls=True,
        replace_emails=True,
        replace_numbers=True,
        remove_stopwords=True,
        stemming=True,
    ),
)


class TextPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self._stemmer = PorterStemmer() if config.stemming else None

    def tokenize(self, text: str) -> list[str]:
        value = str(text)
        if self.config.normalize_unicode:
            value = unicodedata.normalize("NFKC", value)
        if self.config.lowercase:
            value = value.lower()
        if self.config.replace_emails:
            value = EMAIL_RE.sub(" <email> ", value)
        if self.config.replace_urls:
            value = URL_RE.sub(" <url> ", value)
        if self.config.replace_numbers:
            value = NUMBER_RE.sub(" <num> ", value)

        tokens = TOKEN_RE.findall(value)
        if self.config.remove_stopwords:
            tokens = [
                token
                for token in tokens
                if token.startswith("<") or token not in ENGLISH_STOP_WORDS
            ]
        if self._stemmer is not None:
            tokens = [
                token if token.startswith("<") else self._stemmer.stem(token)
                for token in tokens
            ]
        return tokens

    def transform(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.tokenize(text) for text in texts]
