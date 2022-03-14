import pandas as pd
from transformers import BertTokenizer
import numpy as np
from config import MAX_LEN

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def generate_label(polarity):
    if polarity == 'positive':
        return 0
    if polarity =='negative':
        return 1
    return 2

def tokenize(row):
    return tokenizer.encode_plus(
        row.Text,
        max_length = MAX_LEN,
        pad_to_max_length=True,
        add_special_tokens=True,
        return_token_type_ids=False,
        truncation=True
        )

def term_mask(row):
    tok = tokenizer.encode_plus(
        row.Aspect_term,
        max_length = MAX_LEN,
        pad_to_max_length=True,
        add_special_tokens=True,
        return_token_type_ids=False,
        truncation=True
        ).input_ids
    tok = np.array(tok)
    tok = tok[tok>0][1:-1]
    len_term = tok.shape[0]
    mask = np.zeros((len(row.input_ids)))
    for i in range(len(row.input_ids) - len_term + 1):
        if (np.array(row.input_ids[i:i+len_term]) == tok).all():
            mask[i:i+len_term] = 1
        if row.input_ids[i] == 0:
          break
    return mask

def preprocess(filename = '\\data\\traindata.csv'):
    df = pd.read_csv(filename, sep='\t',header=None)
    df.columns = ['Polarity','Aspect_category','Aspect_term','Character_offset','Text']

    aspects = df.Aspect_category.unique()
    aspects.sort()
    df['label'] = df.Polarity.apply(generate_label)

    df['tokenize'] = df.apply(tokenize,axis=1)
    df['input_ids'] = df.tokenize.apply(lambda dico : dico['input_ids'])
    df['attention_mask'] = df.tokenize.apply(lambda dico : dico['attention_mask'])
    df['Term_mask'] = df.apply(term_mask, axis=1)
    df["Aspect_id"] = df.Aspect_category.apply(lambda x: np.argwhere(aspects == x)[0,0])

    return df
