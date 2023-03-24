from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
import pandas as pd
import ast
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.techniques_encoded
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


def create_dataset_training(MODEL_NAME:str, PATHS_TO_TRAIN:list[str], PATHS_TO_VALIDATION:list[str], TRAINING_LANGUAGES:list[str] ) -> tuple[CustomDataset, CustomDataset]:
    """
    Function to load and process the _input_data into suitable format. Includes setting up tokenizer, one hot encoding of
    techniques, concatenation of csv files if necessary, language filtering and loading _input_data into CustomDataset class.

    :param MODEL_NAME: name of the model as per Hugging face to use for tokenizer.
    :param PATHS_TO_TRAIN: List of paths to csv files we want to use for training.
    :param PATHS_TO_VALIDATION: List of paths to csv files we want to use for validation.
    :param TRAINING_LANGUAGES: List of languages we are using.
    :return: Tuple of training and testing set in Dataset format.
    """

    #Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, return_dict=False)
    MAX_LEN = tokenizer.model_max_length
    if MAX_LEN > 1024:
        print(f'hmm, seems the tokenizer is saying too much: {MAX_LEN}, setting down to 512.')
        MAX_LEN = 512
    print(f'max length of tokens for < {MODEL_NAME} > is: < {MAX_LEN} >')

    #Concatenate csv files
    df_train = pd.DataFrame()
    for train_path in PATHS_TO_TRAIN:
        df_train = pd.concat([df_train, pd.read_csv(train_path)], ignore_index=True)

    #One hot encoding of techniques
    df_train['techniques'] = df_train['techniques'].apply(ast.literal_eval)
    one_hot = MultiLabelBinarizer()
    df_train['techniques_encoded'] = one_hot.fit_transform(df_train['techniques'])[:, 1:].tolist()

    #Load validation dataset and encode
    if PATHS_TO_VALIDATION:
        df_dev = pd.DataFrame()
        for dev_path in PATHS_TO_VALIDATION:
            df_dev = pd.concat([df_dev, pd.read_csv(dev_path)], ignore_index=True)

        df_dev['techniques'] = df_dev['techniques'].apply(ast.literal_eval)
        # use fitted one-hot encoder transform in case some techniques are present in training but not in dev
        df_dev['techniques_encoded'] = one_hot.transform(df_dev['techniques'])[:, 1:].tolist()
    else:
        df_dev = pd.DataFrame(
            columns=[['article_id', 'line_number', 'text', 'language', 'techniques', 'techniques_encoded']])

    #Language filtering
    df_train = df_train[df_train['language'].isin(TRAINING_LANGUAGES)]
    df_dev = df_dev[df_dev['language'].isin(TRAINING_LANGUAGES)]
    train_dataset = df_train
    test_dataset = df_dev

    print(f"TRAIN Dataset: {train_dataset.shape}")
    print(f"TEST Dataset: {test_dataset.shape}")

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    return training_set, testing_set


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=False)
    MAX_LEN = tokenizer.model_max_length
    if MAX_LEN > 1024:
        print(f'hmm, seems the tokenizer is saying too much: {MAX_LEN}, setting down to 512.')
        MAX_LEN = 512
    print(f'max length of tokens for < {MAX_LEN} > is: < {MAX_LEN} >')
    return tokenizer, MAX_LEN


def load_dataframe(args):
    test_df = pd.read_csv(args["EVALUATION_SET"])

    language_to_texts = {lan: list(text) for lan, text in test_df.groupby('language')['text']}
    for language, texts in language_to_texts.items():
        print(f'language {language} has {len(texts)} lines of _input_data.')

    if not args['ONLY_TEST']:
        print('Encoding techniques into one-hot-encoded lists!\n')
        # Create sentence and label lists
        test_sentences = test_df.text.values

        test_df.techniques = test_df.techniques.apply(lambda x: ast.literal_eval(x))
        one_hot = MultiLabelBinarizer()
        test_df['techniques_encoded'] = one_hot.fit_transform(test_df['techniques'])[:, 1:].tolist()
        test_labels = test_df.techniques_encoded.values
        test_labels = np.array([labels for labels in test_labels])

        mappings = one_hot.classes_[1:]
    else:
        print('No targets to create one-hot-encodings from! Loading from pre-set list instead...\n')
        mappings = np.array(['Appeal_to_Authority', 'Appeal_to_Fear-Prejudice', 'Appeal_to_Hypocrisy',
                             'Appeal_to_Popularity', 'Appeal_to_Time', 'Appeal_to_Values',
                             'Causal_Oversimplification', 'Consequential_Oversimplification',
                             'Conversation_Killer', 'Doubt', 'Exaggeration-Minimisation',
                             'False_Dilemma-No_Choice', 'Flag_Waving', 'Guilt_by_Association',
                             'Loaded_Language', 'Name_Calling-Labeling',
                             'Obfuscation-Vagueness-Confusion', 'Questioning_the_Reputation',
                             'Red_Herring', 'Repetition', 'Slogans', 'Straw_Man', 'Whataboutism'
                             ])
    return test_df, mappings

def create_dataloader(args, test_df, language, language_counts):
    print(
        f'testing on {language_counts[language]} _input_data points (around {language_counts[language] / args["BATCH_SIZE"]} iterations)')
    filtered_test_df = test_df.copy()
    filtered_test_df = filtered_test_df[filtered_test_df['language'] == language]
    filtered_test_df = filtered_test_df.reset_index()
    tokenizer, MAX_LEN = get_tokenizer(args['MODEL_NAME'])

    testing_set = CustomDataset(
        filtered_test_df,
        tokenizer,
        MAX_LEN
    )

    testing_loader = DataLoader(
        testing_set,
        batch_size=args['BATCH_SIZE'],
        shuffle=args['SHUFFLE'],
        num_workers=args['NUM_WORKERS']
    )

    return testing_loader, filtered_test_df