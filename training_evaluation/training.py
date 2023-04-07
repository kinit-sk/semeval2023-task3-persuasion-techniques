from training_evaluation.models import BERTBase, EnglishRobertaBase, EnglishRobertaLarge, mBERTBase, XLMRobertaBase, XLMRobertaLarge
from torch.utils.data import DataLoader
import torch
from training_evaluation.model_data_preparation import create_dataset_training
from tqdm import tqdm

def run_training(args):
    """
    Function creates a dataset based on given arguments. Afterwards it trains and validates a model.

    :param args: dictionary of arguments needed to create the model
    """
    training_set, testing_set = create_dataset_training(MODEL_NAME=args['MODEL_NAME'], PATHS_TO_TRAIN=args['PATHS_TO_TRAIN'], PATHS_TO_VALIDATION=args['PATHS_TO_VALIDATION'], TRAINING_LANGUAGES=args['TRAINING_LANGUAGES'])

    training_loader = DataLoader(
        training_set,
        batch_size=args['TRAIN_BATCH_SIZE'],
        shuffle=args['TRAIN_SHUFFLE'],
        num_workers=args['TRAIN_NUM_WORKERS']
    )

    validation_loader = DataLoader(
        testing_set,
        batch_size=args['VALIDATION_BATCH_SIZE'],
        shuffle=args['VALIDATION_SHUFFLE'],
        num_workers=args['VALIDATION_NUM_WORKERS']
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lm = LanguageModel(training_loader=training_loader, validation_loader=validation_loader, device=device, args=args)
    return lm.train_over_epochs()


class LanguageModel:
    """
    Class to hold the model as well as train and validate it.
    """

    def __init__(self, training_loader, validation_loader, device, args):

        self.args = args
        self.device = device
        self.validation_loader = validation_loader
        self.training_loader = training_loader
        if args['MODEL_NAME'] == 'roberta-large':
            self.model = EnglishRobertaLarge()
        elif args['MODEL_NAME'] == 'roberta-base':
            self.model = EnglishRobertaBase()
        elif args['MODEL_NAME'] == 'bert-base-uncased':
            self.model = BERTBase()
        elif args['MODEL_NAME'] == 'bert-base-multilingual-cased':
            self.model = mBERTBase()
        elif args['MODEL_NAME'] == 'xlm-roberta-base':
            self.model = XLMRobertaBase()
        elif args['MODEL_NAME'] == 'xlm-roberta-large':
            self.model = XLMRobertaLarge()

        self.model = EnglishRobertaBase()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args['LEARNING_RATE'])
        if args['FREEZE']:
            counter = 0
            no_layers = len([param for param in self.model.parameters()])
            for param in self.model.parameters():
                counter += 1
                param.requires_grad = False
                if counter == (no_layers//5) * 4:
                    break


    def save_model_and_weights(self, epoch):
        print(f'\n\n --- Currently saving model! --- \n')
        model_path = f'{self.args["MODELS_DIR"]}{self.args["MODEL_NAME_PREFIX"]}_model_{epoch}epoch_batch{self.args["TRAIN_BATCH_SIZE"]}_lr{self.args["LEARNING_RATE"]}.pth'
        model_weights_path = f'{self.args["MODELS_DIR"]}{self.args["MODEL_NAME_PREFIX"]}_model_weights_{epoch}epoch_batch{self.args["TRAIN_BATCH_SIZE"]}_lr{self.args["LEARNING_RATE"]}.pth'
        torch.save(self.model.state_dict(), model_weights_path)
        torch.save(self.model, model_path)
        return model_path

    def train(self, epoch):
        print(f'\nNow training on Epoch {epoch}.')
        print(
            f'Testing on {len(self.training_loader) * self.training_loader.batch_size - 1} data points (around {len(self.training_loader)} iterations)')

        self.args['CURRENT_TRAIN_LOSS'] = 0
        self.model.train()
        for itr, data in tqdm(enumerate(self.training_loader, 0)):
            ids = data['ids'].to(self.device, dtype=torch.long)
            mask = data['mask'].to(self.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
            targets = data['targets'].to(self.device, dtype=torch.float)

            outputs = self.model(ids, mask, token_type_ids)

            loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
            if itr % 200 == 0:
                print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            self.args['CURRENT_TRAIN_LOSS'] += loss

        print(f'Accumulated Training Loss after Epoch {epoch} is: {self.args["CURRENT_TRAIN_LOSS"]}')

    def validate(self, epoch):
        print(f'\nNow validating on Epoch {epoch}.')
        print(
            f'Validating on {len(self.validation_loader) * self.validation_loader.batch_size} data points (around {len(self.validation_loader)} iterations)')
        self.model.eval()
        self.args['CURRENT_VALIDATION_LOSS'] = 0
        with torch.no_grad():
            for val_itr, data in tqdm(enumerate(self.validation_loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)
                loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)

                if val_itr % 200 == 0:
                    print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')

                # accumulate the loss across all batches
                self.args['CURRENT_VALIDATION_LOSS'] += loss

                # clear cache
                torch.cuda.empty_cache()

        print(f'Accumulated Validation Loss after Epoch {epoch} is: {self.args["CURRENT_VALIDATION_LOSS"]}')

        # Early stopping if necessary
        if self.args['CURRENT_VALIDATION_LOSS'] < self.args['BEST_LOSS']:
            self.args['BEST_LOSS'] = self.args['CURRENT_VALIDATION_LOSS']
            self.save_model_and_weights(epoch)

            return True
        else:
            # stop training
            print('Validation loss increased! Stopping training...')
            print(f'Last validation loss was: {self.args["BEST_LOSS"]}')
            print(f'Current validation loss is: {self.args["CURRENT_VALIDATION_LOSS"]}')

            return False

    def train_over_epochs(self):
        for epoch in range(self.args['EPOCHS']):
            self.train(epoch)
            if self.args['PATHS_TO_VALIDATION'] and not self.validate(epoch):
                break  # the last epoch was NOT counted due to early stopping
            else:
                # store latest version
                path = self.save_model_and_weights(self.model)
        print("Done.")
        return path
