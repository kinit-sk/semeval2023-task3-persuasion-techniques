# Creating the customized models, by adding a drop out and a dense layer on top of BERT-based model to get the final output for the model.
import torch
import transformers

class BERTBase(torch.nn.Module):
    def __init__(self):
        super(BERTBase, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 23)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class EnglishRobertaBase(torch.nn.Module):
    def __init__(self):
        super(EnglishRobertaBase, self).__init__()
        self.l1 = transformers.RobertaModel.from_pretrained('roberta-base', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 23)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class EnglishRobertaLarge(torch.nn.Module):
    def __init__(self):
        super(EnglishRobertaLarge, self).__init__()
        self.l1 = transformers.RobertaModel.from_pretrained('roberta-large', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(1024, 23)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class mBERTBase(torch.nn.Module):
    def __init__(self):
        super(mBERTBase, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-multilingual-cased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 23)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class XLMRobertaBase(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaBase, self).__init__()
        self.l1 = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 23)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class XLMRobertaLarge(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaLarge, self).__init__()
        self.l1 = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-large', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(1024, 23)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output