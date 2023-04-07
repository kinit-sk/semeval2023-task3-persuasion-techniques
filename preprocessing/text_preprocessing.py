"""
Functions for preprocessing of the text.

The main strategy used is finding patterns using regular expressions and replacing them with a placeholder
in a format of {pattern} in cases of urls, emojis, hashtags etc. or to delete subsequent repeating characters.
"""

import re
import advertools as adv
from advertools.regex import URL


def clean_emojis(text, replace):
    emojis_present = set(adv.extract_emoji([text])["emoji"][0])
    if len(emojis_present) > 0:
        text_replace = "{emoji} " if replace else ""
        text = re.sub(r"|".join(emojis_present), text_replace, text)
    return text


def clean_urls(text, replace):
    text_replace = "{url}" if replace else ""
    return URL.sub(text_replace, text)


def clean_hashtags(text, replace):
    text_replace = "{hashtag}" if replace else ""
    return re.sub(r"#[^\s]+", text_replace, text)


def clean_mentions(text, replace):
    text_replace = "{mention}" if replace else ""
    return re.sub(r"\B\(?@[^\s]+", text_replace, text)


def clean_emails(text, replace):
    text_replace = "{email}" if replace else ""
    return re.sub(r"[^\s]+[\w]+@[^\s]+[\w]", text_replace, text)


def normalize_punctuation(text):
    return re.sub(r"([?!.])\1+", lambda m: m.string[m.span()[0]], text)


def normalize_whitespaces(text):
    return re.sub(r"[\s]+", " ", text).strip()


def clean_repeating_non_word_characters(text):
    return re.sub(r"([\W_])\1+", "", text).strip()


def clean_all_non_word_non_whitespace_characters(text):
    return re.sub(r"[^\s\w{}]", "", text).strip()


def lower_text(text):
    return text.lower()


def upper_text(text):
    return text.upper()
