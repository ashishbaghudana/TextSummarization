from __future__ import division, unicode_literals, print_function
from glob import glob
from io import open
from utils import normalize_string
import os

from language import Language

FULL_TEXT_EXTENSION = ".text"
SUMMARY_EXTENSION = ".summary"


class DataLoader(object):
    def __init__(self, full_text_directory, summary_directory=None):
        self.full_text_directory = full_text_directory
        self.summary_directory = summary_directory
        if self.summary_directory is None:
            self.summary_directory = self.full_text_directory

    def load(self):
        full_text_lang = Language(type='full_text')
        summary_text_lang = Language(type='summary_text')
        pairs = []
        for doc, summary in self:
            full_text_lang.add_text(doc)
            summary_text_lang.add_text(summary)
            pairs.append((doc, summary))
        return full_text_lang, summary_text_lang, pairs

    def _read(self, filename):
        with open(filename) as fp:
            content = fp.read()
        return normalize_string(content)

    def _get_basename(self, filename):
        return os.path.splitext(os.path.basename(filename))[0]

    def _get_summary_filename(self, filename):
        return os.path.join(
            self.summary_directory,
            f"{self._get_basename(filename)}{SUMMARY_EXTENSION}")

    def __iter__(self):
        full_texts = glob(
            os.path.join(self.full_text_directory, f"*{FULL_TEXT_EXTENSION}"))
        for filename in full_texts:
            yield (self._read(filename),
                   self._read(self._get_summary_filename(filename)))
