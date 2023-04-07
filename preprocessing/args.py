import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Multilingual Persuasion Technique Detection on a Paragraph Level.')

parser.add_argument(
    '--method',
    type=str,
    default='dummy',
    choices=['dummy', 'classification'],
    required=True,
    help='Decide the method of how to evaluate a given a model. Make sure the model chosen in model-name can support it.',
)

parser.add_argument(
    '--evaluate-as-test',
    default='dev',
    required=True,
    choices=['train', 'dev', 'test'],
    help='Decide which set(s) will be used for evaluation of the chosen method/model.',
)

parser.add_argument(
    '--article-languages',
    type=str,
    nargs='*',
    choices=['en', 'fr', 'ge', 'it', 'po', 'ru', 'ka', 'gr', 'es'],
    default=['en', 'fr', 'ge', 'it', 'po', 'ru', 'ka', 'gr', 'es'],
    help='filter for fact checks based on language',
)

parser.add_argument(
    '--model-name',
    type=str,
    default='na',
    help='handle for sentence_transformers model to be used for fine-tuning. Also name of model to be used for labelling.'
)

parser.add_argument(
    '--wandb-run-name',
    type=str,
    default=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    help='name for wandb run defaults to timestamp',
)

parser.add_argument(
    '--wandb-project',
    type=str,
    help='name for wandb project where the results will be logged',
)

parser.add_argument(
    '--wandb-offline',
    type=str,
    default='online',
    help='Select offline if you wish to run locally. Leave blank, or select "online"',
)

# Text preprocessing args

parser.add_argument(
    '--add-placeholders',
    action='store_true',
    help=""""Replace removed text with placeholders.
        urls - '{url}'
        emojis - '{emoji}'
        hashtags - '{hashtag}'
        mentions - '{mention}'
        emails - '{email}'
        """"",
)

parser.add_argument(
    '--clean-emojis',
    action='store_true',
    help='Remove emojis.',
)

parser.add_argument(
    '--clean-urls',
    action='store_true',
    help='Remove urls.',
)

parser.add_argument(
    '--clean-hashtags',
    action='store_true',
    help='Remove hashtags.',
)

parser.add_argument(
    '--clean-mentions',
    action='store_true',
    help='Remove mentions.',
)

parser.add_argument(
    '--clean-emails',
    action='store_true',
    help='Remove emails.',
)

parser.add_argument(
    '--normalize-punctuation',
    action='store_true',
    help='Replace 2+ consecutive "?", "!" or "." with a single matching character.',
)

parser.add_argument(
    '--normalize-whitespaces',
    action='store_true',
    help='Replace 2+ whitespace characters with a single whitespace.',
)

parser.add_argument(
    '--clean-repeating-non-word',
    action='store_true',
    help='Remove repeating non-word characters.',
)

parser.add_argument(
    '--clean-all-non-word-non-whitespace',
    action='store_true',
    help='Remove all non-word, non-whitespace characters apart from the "{" and "}" characters which are '
         'reserved as placeholders',
)

parser.add_argument(
    '--lower-text',
    action='store_true',
    help='Lower all word characters.',
)

parser.add_argument(
    '--upper-text',
    action='store_true',
    help='Capitalize all word characters.',
)

parser.add_argument(
    '--translate-to-english',
    action='store_true',
    help='Translate all _input_data to english.',
)