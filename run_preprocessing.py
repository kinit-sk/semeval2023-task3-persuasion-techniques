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



# Set which data to pull
sentence_id_to_sentence, article_id_to_sentences = init_raw_data(train_set=True, dev_set=False, test_set=False)


df = pd.DataFrame(data=sentence_id_to_sentence.values())
# change the location as desired
df.to_csv(f"./_processed_data/data.csv", index=False)