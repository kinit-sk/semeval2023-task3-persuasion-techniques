"""
This is dummy method that predicts 'Appeal_to_Fear-Prejudice' and 'Doubt' for every sentence.
These two techniques have just randomly been chosen.
"""
import logging

from preprocessing.namedtuples import Result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dummy_method')


def run_dummy_method(sentence_id_to_sentence, article_id_to_sentences, args):   
    results = []

    logger.info(f'len of sentence dict is: {len(sentence_id_to_sentence.values())}')
    for sentence in sentence_id_to_sentence.values():
        results.append(Result(
            article_id=sentence.article_id,
            paragraph_id=sentence.line_number,
            techniques=['Loaded_Language'],
            language=sentence.language
        ))
        
    return results
