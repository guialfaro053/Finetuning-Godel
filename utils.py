from torch.utils.data import DataLoader, Dataset, SequentialSampler
import os, glob, json, pandas as pd, numpy as np
from nltk.tokenize import TweetTokenizer
import re
from transformers import AutoTokenizer, InputExample

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def postprocess_text(preds, labels):
        preds = [normalize_answer(pred.strip()) for pred in preds]
        labels = [normalize_answer(label.strip()) for label in labels]
        return preds, labels

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s

def clean_str(txt):
    """
    Lower text, remove url, illegal char, etc.
    """
    txt = txt.lower()
    txt = re.sub('^', ' ', txt)
    txt = re.sub('$', ' ', txt)
    # url and tag
    words = []
    for word in txt.split():
        i = word.find('http')
        if i >= 0:
            word = word[:i] + ' ' + '__url__'
        words.append(word.strip())
    txt = ' '.join(words)
    # remove markdown URL
    txt = re.sub(r'\[([^\]]*)\] \( *__url__ *\)', r'\1', txt)
    # remove illegal char
    txt = re.sub('__url__', 'URL', txt)
    txt = re.sub(r"[^A-Za-z0-9():,.!?\"\']", " ", txt)
    txt = re.sub('URL', '__url__', txt)
    # contraction
    add_space = ["'s", "'m", "'re", "n't", "'ll", "'ve", "'d", "'em"]
    tokenizer = TweetTokenizer(preserve_case=False)
    txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '
    txt = txt.replace(" won't ", " will n't ")
    txt = txt.replace(" can't ", " can n't ")
    for a in add_space:
        txt = txt.replace(a+' ', ' '+a+' ')
    txt = re.sub(r'^\s+', '', txt)
    txt = re.sub(r'\s+$', '', txt)
    txt = re.sub(r'\s+', ' ', txt)  # remove extra spaces
    return txt

class SkyeDataset(Dataset):
    """
    Data Initialization according to datafile 
    """
    def __init__(self, dataset_file, tokenizer, max_length, instruction = None, knowledge = None):
        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.max_length = max_length

        extension = dataset_file.split('.')[-1]
        
        if extension == 'json':
            context = []
            label = []

            for index, db in enumerate(json.load(open(dataset_file, 'r'))):
                text_snippet = []
                for d_index in range(0, len(db['Context']), 2):
                    dialog = db['Context'][d_index]
                    instruction = db['Instruction'][d_index]
                    knowledge = db['Knowledge'][d_index]
                    text_snippet.append(dialog)
                    convo = ' EOS '.join(text_snippet)
                    query = f"{instruction } [CONTEXT] {str(convo)} [KNOWLEDGE] {knowledge }"
                    context.append(query)
                    label += [db['Context'][d_index + 1]]
                    text_snippet.append(label[-1])
        
        self.context = context
        self.label = label

    def __len__(self):
        """
        Returns length of dataset
        """
        return len(self.label)

    def __getitem__(self, index):
        """
        Returns a dictionary of input_ids, attention_masks, and label input_ids 
        """
        _context = self.context[index]
        _label = self.label[index]
        context_encoding = self.tokenizer.encode_plus(_context, max_length = self.max_length, 
                                                        padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)
        with self.tokenizer.as_target_tokenizer():
            label_encoding = self.tokenizer.encode_plus(_label, max_length = self.max_length, 
                                                            padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)

        label_encoding["labels"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in label_encoding["input_ids"]]

        return {
                'input_ids': context_encoding.input_ids.flatten(), 
                'attention_mask': context_encoding.attention_mask.flatten(), 
                'labels': label_encoding.input_ids.flatten()
                }

def createDataLoader(ds, batch_size, eval=False):
    if eval:
        eval_sampler = SequentialSampler(ds)
        return DataLoader(ds, batch_size, sampler=eval_sampler)
    return DataLoader(ds, batch_size, shuffle=True)

