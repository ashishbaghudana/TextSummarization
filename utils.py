from __future__ import division, unicode_literals, print_function
import unicodedata
import re
import time
import math


def as_minutes(s):
    """
    Convert minutes to seconds
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent, remaining=False):
    """
    Get time since a given time
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    if remaining:
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
    else:
        return '%s' % as_minutes(s)


def time_string():
    """Get time as a well formatted string"""
    now = time.localtime(time.time())
    return time.strftime("%Y-%m-%d %H:%M:%S", now)


def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII thanks to
    http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    """
    Normalize the string by converting to lowercase and replacing any
    characters that are punctuations or digits with an empty space
    """
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
