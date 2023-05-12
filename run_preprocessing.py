import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from preprocessing.init_data import init_raw_data
from preprocessing.args import parser

"""
If you wish to get data that are processed in some way (preprocessing, translations etc.) uncomment the
two lines below and comment the line after that. Then run the programs with appropriate arguments using
the argparser. 
"""


#args = parser.parse_args()
#sentence_id_to_sentence, article_id_to_sentences = init_data(args)

for name in ('train_set', 'dev_set', 'test_set'):

    #Choose a file name
    FILE_NAME = f"{name}.csv"

    # Set which data to pull
    sentence_id_to_sentence, article_id_to_sentences = init_raw_data(train_set=name == 'train_set', dev_set=name == 'dev_set', test_set=name == 'test_set')


    df = pd.DataFrame(data=sentence_id_to_sentence.values())

    # name file as desired
    df.to_csv(f"./_processed_data/{FILE_NAME}", index=False)
