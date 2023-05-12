import numpy as np
from sklearn import metrics
import torch
import pandas as pd

from training_evaluation.model_data_preparation import load_dataframe, create_dataloader
from collections import Counter
from training_evaluation.models import BERTBase, EnglishRobertaBase, EnglishRobertaLarge, mBERTBase, XLMRobertaBase, XLMRobertaLarge

from tqdm import tqdm


def test(model, testing_loader, only_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            # if there are no targets available
            if not only_test:
                targets = data['targets'].to(device, dtype=torch.float)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())

            outputs = model(ids, mask, token_type_ids)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def run_tests(model, testing_loader, evaluation_threshold, only_test, lang, results_list):
    outputs, targets = test(model, testing_loader, only_test)
    outputs = np.array(outputs) >= evaluation_threshold

    if targets:
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        results_list.append([lang, accuracy, f1_score_micro, f1_score_macro])
        print(results_list)

    return outputs, results_list


def create_submission_txt(outputs, df, path, mappings):
    df = df[['article_id', 'line_number']]
    outputs = [
        ",".join([technique for technique, keep in zip(mappings, output) if keep])
        for output in outputs
    ]
    df['techniques'] = outputs
    df.to_csv(path, sep='\t', header=None, index=None)


def run_eval(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args["MODEL_FILE_NAME"])
    model.to(device)

    test_df, mappings = load_dataframe(args)
    language_counts = Counter(test_df.language)

    results_list = []
    for language in args['LANGUAGES']:
        print(f'\n\n=========== Now evaluating language: {language} ===========\n')

        if language_counts.get(language, 'na') == 'na':
            print('no _input_data points to be tested')

        else:
            testing_loader, filtered_test_df = create_dataloader(args, test_df, language, language_counts)

            outputs, results_list = run_tests(model, testing_loader, args['EVALUATION_THRESHOLD'], args['ONLY_TEST'], language, results_list)

            path = f'results/{args["RESULT_DIR_NAME"]}/{language}.txt'
            create_submission_txt(outputs, filtered_test_df, path, mappings)

    df_results = pd.DataFrame(results_list, columns=['language', 'accuracy', 'F1_micro', 'F1_macro'])
    df_results.to_csv(f'results_analysis/results_table/{args["RESULT_DIR_NAME"]}.csv', sep='\t', header=True, index=False)
    print('Done!')
