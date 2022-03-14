import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import preprocess

class My_Dataset(Dataset):
    def __init__(self, df, transform = None):
    
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids = self.df['input_ids'].iloc[index]
        attention_mask = self.df['attention_mask'].iloc[index]
        label = self.df['label'].iloc[index]
        aspect_id = self.df['Aspect_id'].iloc[index]
        term_mask = self.df['Term_mask'].iloc[index]
        

        return {'term_mask':torch.tensor(term_mask), 
                'aspect_id':torch.tensor(aspect_id), 
                'input_ids': torch.tensor(input_ids), 
                'attention_mask':torch.tensor(attention_mask),
                'label':torch.tensor(label)}

def create_data_loader(filename, train=True):

    df = preprocess(filename)
    dataset = My_Dataset(df)
    batch_size = 8 if train else 4
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader
