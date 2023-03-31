import ast
import os
import pandas as pd
from _input_data.scorers.scorer_subtask_3 import main_as_function as get_scorer_micro_macro

args = {
    # PLEASE MAKE SURE THAT 'RESULTS_DIR_1' and 'RESULTS_DIR_2' are well listed and in PARALLEL ORDER!!!
    # Ensembling is primarily meant for original - english model pairs.

    'RESULTS_DIR_1': [

        '',


        '',

    ],

    'RESULTS_DIR_2': [

        '',


        '',

    ],

    'ENSEMBLE_METHOD': 'union',  # choices: ['union']
    'LANGUAGES': ['en', 'fr', 'ge', 'it', 'po', 'ru'],
    # choices: ['en', 'fr', 'ge', 'it', 'po', 'ru', 'ka', 'gr', 'es']
}

for dir1, dir2 in zip(args['RESULTS_DIR_1'], args['RESULTS_DIR_2']):
    print('\n\n============================================================================================')
    print(f'Now ensembling models: 1.{dir1} and 2.{dir2} with ensemble method: {args["ENSEMBLE_METHOD"]}')

    result_dir_name = f'{dir1}_and_{dir2}'
    result_dir = f'results/{result_dir_name}'
    os.makedirs(result_dir, exist_ok=True)

    """
    Loading Files
    """
    print('Loading Files...')
    results_list = []

    for language in args['LANGUAGES']:
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()

        file1 = f'results/{dir1}/{language}.txt'
        file2 = f'results/{dir2}/{language}.txt'

        if os.path.exists(file1) and os.path.exists(file2):
            file1_df = pd.read_csv(file1, delimiter="\t", keep_default_na=False, names=['article_id', 'line_number', 'techniques'])
            file1_df['language'] = language
            file2_df = pd.read_csv(file2, delimiter="\t", keep_default_na=False, names=['article_id', 'line_number', 'techniques'])
            file2_df['language'] = language
            df1 = pd.concat([df1, file1_df], axis=0)
            df2 = pd.concat([df2, file2_df], axis=0)
        else:
            print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(f'Could not find file: {file1} or {file2}')
            print('Please make sure you have typed in the correct results directories and/or languages')
            print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!')
            break
        print('Done')

        """
        Ensembling .txt Files
        """
        print('Ensembling .txt Files...')
        merged_df = pd.merge(df1, df2, on=['article_id', 'line_number'], how='outer')

        def techniques_union(row):
            techniques = [tech.strip() for tech in row[0].split(',')] + [tech.strip() for tech in row[1].split(',')]
            techniques = [tech for tech in techniques if tech]
            return ','.join(set(techniques))

        merged_df['techniques'] = merged_df.apply(lambda row: techniques_union(row[['techniques_x', 'techniques_y']]), axis=1)

        merged_df.drop(['techniques_x', 'techniques_y', 'language_x'], axis=1, inplace=True)

        print(f'Saving ensembled txt file for language: {language} into {result_dir}/{language}.txt')
        filtered_result = merged_df[merged_df['language_y'] == language]
        filtered_result = filtered_result.drop('language_y', axis=1)
        filtered_result.to_csv(f'{result_dir}/{language}.txt', sep='\t', header=None, index=None)

        print('Done.')


        micro_f1, macro_f1 = get_scorer_micro_macro(
            f'_input_data/{language}-dev-labels-subtask-3.txt',
            f'{result_dir}/{language}.txt',
            'techniques_subtask3.txt'
        )
        results_list.append((language, 0, micro_f1, macro_f1))
        print(results_list)

        print(f'Recording results to < results_table/{result_dir_name}.csv >')
        results_df = pd.DataFrame(results_list, columns=[['language', 'accuracy', 'F1_micro', 'F1_macro']])
        results_df.to_csv(f'results_analysis/results_table/ensemble/{result_dir_name}.csv', sep='\t', header=True, index=False)