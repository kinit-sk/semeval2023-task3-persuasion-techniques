import logging
#import nltk
import os
import pandas as pd

from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, TextIO
from tqdm import tqdm
from argparse import Namespace
from preprocessing.google_translate import GoogleTranslate
from preprocessing.args import parser
from preprocessing.namedtuples import Article, Span, Line
import preprocessing.text_preprocessing as tp
import sys
sys.path.append('..')



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('init_data')


def init_raw_data(train_set=True, dev_set=True, test_set=True) -> Tuple[Dict[int, Line], Dict[int, Article]]:
    path = '_input_data/data'
    sentences = []
    labels = []
    logger.info("Loading _input_data...")

    def append_formatted_lines(article_file: str, language: str, label_file: str = None) -> List[Tuple[int, int, int, str, str, List[str]]]:
        
        with open(article_file, 'r', encoding='utf-8') as article_file:
            articles = [line.split('\t') for line in article_file.read().split('\n')]
            for sentence in articles:
                if len(sentence) != 1: # there are blank lines, for some reason
                    if len(sentence) == 3:
                        sentences.append((int(sentence[0]), int(sentence[1]), sentence[2], language))
                    if len(sentence) == 5:
                        c = sentence[2][-1]
                        new_id = ''
                        idx = -1
                        while c.isdigit():
                            new_id += c
                            c = sentence[2][idx-1]
                            idx -= 1
                        new_id = new_id[::-1]
                        sentences.append((int(sentence[0]), int(sentence[1]), sentence[2][:idx], language)) # the original line
                        sentences.append((int(new_id), int(sentence[3]), sentence[4], language)) # the corrupted line


        if label_file:
            with open(label_file, 'r', encoding='utf-8') as label_file:
                l = [line.split('\t') for line in label_file.read().split('\n')]
                for label_line in l:
                    if len(label_line) != 1: # just in case, as articles exhibited odd behaviour with some empty lines of just ['']
                        assert len(label_line) == 3
                        labels.append((int(label_line[0]), int(label_line[1]), label_line[2]))

    for language in tqdm(parser.get_default('article_languages')):
        for _set in [(train_set, 'train'), (test_set, 'test'), (dev_set, 'dev')]:
            if _set[0]:
                line_file = f'{path}/{language}/{_set[1]}-labels-subtask-3.template'
                if os.path.exists(line_file): # some languages do not have all _input_data available
                    label_file = None if _set[1] == 'test' else f'{path}/{language}/{_set[1]}-labels-subtask-3.txt'

                    append_formatted_lines(
                        line_file,
                        language,
                        label_file=label_file
                    )

    """
    Assign labels where appropriate
    """
    labels_dict = {(l[0],l[1]):l[2] for l in labels}
    sentences = [(*s, labels_dict.get((s[0], s[1]), 'unknown').split(',')) for s in sentences] # test set will have unknown labels

    # convert to namedtuple
    sentences = [Line(*s) for s in sentences]
    """
    Create id mappings
    """
    sentence_id_to_sentence = {i:v for i,v in enumerate(sentences)}

    article_id_to_sentences = defaultdict(list)
    for line in sentences:
        article_id_to_sentences[line.article_id].append(line)

    return sentence_id_to_sentence, article_id_to_sentences


def init_data(args):
    print(args)
    train = False
    dev = False
    test = False
    if args.evaluate_as_test == 'train': train = True
    elif args.evaluate_as_test == 'dev': dev = True
    elif args.evaluate_as_test == 'test': test = True

    sentence_id_to_sentence, article_id_to_sentences = init_raw_data(
        train_set=train,
        dev_set=dev,
        test_set=test
    )
    
    logger.info(f'Number of lines: {(l := len(sentence_id_to_sentence))}')
    logger.info(f'Number of articles: {(a := len(article_id_to_sentences))}')

    """
    Language Filtering
    """
    logger.info('Beginning Language Filtering.')
    sentence_id_to_sentence = {k:v 
        for k,v
        in sentence_id_to_sentence.items()
        if v.language in args.article_languages}
    logger.info(f'Number of lines deleted after language filtering: {l - (l := len(sentence_id_to_sentence))}, Left: {l}')
    logger.info(f'Number of articles deleted after language filtering: {a - (a := len(article_id_to_sentences))}, Left: {a}')


    """
    Translations
    """
    if args.translate_to_english:
        logger.info(f'Initializing GoogleTranslate')
        translator = GoogleTranslate().load() # load dictionary of translations
        logger.info(f'Done')

        logger.info(f'Translating texts')
        sentence_id_to_sentence = {
            k: Line(s.article_id, s.line_number, translate_text(s.text, translator), s.language, s.techniques)
            for k, s
            in sentence_id_to_sentence.items()
        }
        logger.info(f'Done')


    """
    Text Pre-processing
    """
    logger.info('Beginning text preprocessing.')
    sentence_id_to_sentence = {
        k: Line(s.article_id, s.line_number, clean_text(s.text, args), s.language, s.techniques)
        for k, s
        in sentence_id_to_sentence.items()
    }

    logger.info(f'Number of lines deleted after text preprocessing: {l - (l := len(sentence_id_to_sentence))}, Left: {l}')

    return sentence_id_to_sentence, article_id_to_sentences



def translate_text(text, translator):
    en_text, lang = translator.dict.get(
        text, # if present, return (english_text, detected_language)
        (text, 'en') # if not present, return original (i.e. already in 'en')
    )


    return en_text


def clean_text(text: str, args: Namespace):
    replace = args.add_placeholders

    if args.clean_emojis:
        text = tp.clean_emojis(text, replace)

    if args.clean_mentions:
        text = tp.clean_mentions(text, replace)

    if args.clean_hashtags:
        text = tp.clean_hashtags(text, replace)

    if args.clean_urls:
        text = tp.clean_urls(text, replace)

    if args.clean_emails:
        text = tp.clean_emails(text, replace)

    if args.normalize_punctuation:
        text = tp.normalize_punctuation(text)

    if args.normalize_whitespaces:
        text = tp.normalize_whitespaces(text)

    if args.clean_repeating_non_word:
        text = tp.clean_repeating_non_word_characters(text)

    if args.clean_all_non_word_non_whitespace:
        text = tp.clean_all_non_word_non_whitespace_characters(text)

    if args.lower_text:
        text = tp.lower_text(text)

    if args.upper_text:
        text = tp.upper_text(text)

    return text
