# KInITVeraAI at SemEval-2023 Task 3: Simple yet Powerful Multilingual Fine-Tuning for Persuasion Techniques Detection

This repository contains supplementary material for the paper *KInITVeraAI at SemEval-2023 Task 3: Simple yet Powerful Multilingual Fine-Tuning for Persuasion Techniques Detection*.

### Citing the Paper:
If you make use of any data or modules in this repository, please cite the following paper:

*Hromadka, T., Smolen, T., Remis, T., Pecher, B., & Srba, I. (2023). KInITVeraAI at SemEval-2023 Task 3: Simple yet Powerful Multilingual Fine-Tuning for Persuasion Techniques Detection. In Proceedings of the 17th International Workshop on Semantic Evaluation.*

## Abstract

This paper presents the best-performing solution to the SemEval 2023 Task 3 on the subtask 3 dedicated to persuasion techniques detection. Due to a high multilingual character of the input data and a large number of 23 predicted labels (causing a lack of labelled data for some language-label combinations), we opted for fine-tuning pre-trained transformer-based language models. Conducting multiple experiments, we find the best configuration, which consists of large multilingual model (XLM-RoBERTa large) trained jointly on all input data, with carefully calibrated confidence thresholds. Our final system performed the best on 6 out of 9 languages (including two surprise languages) and achieved highly competitive results on the remaining three languages.

## Structure of code

This repository is structured in three folders:

1. [Input data](_input_data) – raw data and tools provided by task organizer
2. [Processed data](_processed_data) - processed data in csv format
3. [Exploratory Data Analysis](eda) - notebook with exploratory data analysis of the dataset
4. [Preprocessing](preprocessing) - python files to process data into csv format
5. [Results](results) - folder where your results will be saved
6. [Analysis of results](results_analysis) - data of our results and notebook with their analysis
7. [Training and evaluation](training_evaluation) - python files for training and evaluation of models

## Data

To get the original data of the task, please contact the task organizers. You can find all the necessary information as well as contact to the organizers at [this page](https://propaganda.math.unipd.it/semeval2023task3/). Once you have access to the data, please put it into the _input_data folder in order to run the code (it should have folders baselines, data and scorers inside). To process the data into .csv format use [run_preprocessing.py](run_preprocessing.py). Afterwards, you can just use these files to train the models. You can either create csv with raw data, or if desired you can translate or do further preprocessing by using the init_data function. To do this you need to uncomment lines in [text preprocessing](text_preprocessing.py) and run it with appropriate [arguments](/preprocessing/args.py).


## Running the code
To run the training and evaluation of the model, you need to run [run_persuasion_detection.py](run_persuasion_detection.py). This files trains and evaluates the model. Before running, it is important to set the arguments to the values as per your request. You do this by changing the values in *args_train* and *args_eval* for the arguments for training and evaluation respectively. The purpose of the fields are explained in the comments within the files. After running the code, the results will be displayed. Furthermore, the model will be saved in the location set in arguments as well as the calculated results, which will be saved in [results_analysis/results_table](results_analysis/results_table) folder.

In case you want to get ensembled results, you can use the [run_ensemble.py](run_ensemble.py) and set the arguments accordingly. The calculated metrics will be saved in the same location. The riles with results which ensemble method uses are expected to be in the results folder, to which the evaluation method should save them by default.

To analyse the results you can use the [create_results.ipynb](create_results.ipynb) notebook, which leverages the tables with calculated results during evaluation.

## Results Replication

For replication of Figures 2 and 4 from the paper, please see the plots as well as their code in [create_results.ipynb](create_results.ipynb). To replicate table 1, first obtain results for each language in submission format. Afterwards you will need to concatenate all of these files together (manually or using code) to form a single file containing results for all of the languages. These results are then to be fed to scorer_subtask_3.py, which is provided by the task organizers.