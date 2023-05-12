from training_evaluation.training import run_training
from training_evaluation.evaluating import run_eval
from training_evaluation.models import BERTBase, EnglishRobertaBase, EnglishRobertaLarge, mBERTBase, XLMRobertaBase, XLMRobertaLarge


args_train = {
    # general
    'MODEL_NAME_PREFIX': '',                                # so that model names don't get overwritten, write a UNIQUE name prefix to distinguish this current model
    'EPOCHS': 5,                                            # Make sure to set the EXACT amount of epochs for non-validating runs. Typically BERT-like models won't need to go over 4 epochs
    'LEARNING_RATE': 1e-05,
    'MODEL_NAME': 'xlm-roberta-large',                      # model name as per hugging face library, currently supported model names are: roberta-large, roberta-base, bert-base-uncased, bert-base-multilingual-cased, xlm-roberta-base and xlm-roberta-large
    'DROPOUT_RATE': 0.3,
    'MODELS_DIR': '',                                       # name of subdirectory within 'models' folder where the trained models will be saved, e.g., 'distilbert_base_threshold_0_4'
    'PATHS_TO_TRAIN': ['_processed_data/train_set.csv', '_processed_data/dev_set.csv'],    # useful when training on more sets (e.g. train and dev), list of paths to the csv files
    'PATHS_TO_VALIDATION': [],                              # leave empty if no validation required

    # for training
    'TRAIN_BATCH_SIZE': 8,
    'TRAIN_SHUFFLE': True,
    'TRAIN_NUM_WORKERS': 0,
    'CURRENT_EPOCH': 0,
    'CURRENT_TRAIN_LOSS': 0,
    'TRAINING_LANGUAGES': ['en', 'fr', 'ge', 'it', 'po', 'ru'], # full list is ['en', 'fr', 'ge', 'it', 'po', 'ru']
    'FREEZE': False,                                        # If True first 80% of the layers will be frozen during training

    # for validation
    'VALIDATION_BATCH_SIZE': 8,
    'VALIDATION_SHUFFLE': False,
    'VALIDATION_NUM_WORKERS': 0,
    'BEST_LOSS': float('inf'),
    'CURRENT_VALIDATION_LOSS': 0
}

model_file = run_training(args_train)


args_eval = {
    'MODEL_FILE_NAME': model_file,                          # path to saved model
    'RESULT_DIR_NAME': '',                                  # name of subdirectory within 'results' folder where to store the txt result files (for submission), e.g., 'distilbert_base_threshold_0_4'
    'EVALUATION_THRESHOLD': 0.5,                            # select the confidence threshold for when to assign a label. Default is 0.5
    'MODEL_NAME': args_train['MODEL_NAME'],                 # change in case you are not evaluating the same model that you are training
    'ONLY_TEST': True,                                      # set to `True` ONLY IF running on test set (not dev set or anything else)
    'EVALUATION_SET': '_processed_data/test_set.csv',       # path to csv with evaluation set
    'LANGUAGES': ['en', 'fr', 'ge', 'it', 'po', 'ru', 'ka', 'gr', 'es'],    # select which languages to evaluate, packaged in a list. # e.g. ['en', 'fr', 'ge', 'it', 'po', 'ru', 'ka', 'gr', 'es'],

    # test params
    'BATCH_SIZE': 32,
    'SHUFFLE': False,  # no need to shuffle during evaluation
    'NUM_WORKERS': 0
}


run_eval(args_eval)



