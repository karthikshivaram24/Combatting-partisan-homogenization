import torch
from Scripts.utils.bert_utils import load_tokenizer

class CPDatasetMT(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = load_tokenizer()
        print(self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        x1 = self.df["text"].iloc[index]
        x2 = self.df["context_word"].iloc[index]
        y1 = self.df["class_label"].iloc[index]
        y2 = self.df["word_label"].iloc[index]
        wc = self.df["which_cluster"].iloc[index]
        
        # replace context word from positive samples
        x1 = x1.replace(x2,"")
        x1 = self.tokenizer.encode(x1,truncation=True,padding="max_length",max_length=500, add_special_tokens=True)
        x2 = self.tokenizer.encode(x2,truncation=True,add_special_tokens=False,max_length=1)
        
        x1 = torch.LongTensor(x1)
        x2 = torch.LongTensor(x2)
        y1 = torch.Tensor([y1])
        y2 = torch.Tensor([y2])
        wc = torch.Tensor([wc])
        return x1,x2, y1,y2, wc

class CPDatasetST(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = load_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        x1 = self.df["text"].iloc[index]
        y1 = self.df["binary_ps"].iloc[index]
        wc = self.df["which_cluster"].iloc[index]
        
        x1 = self.tokenizer.encode(x1,truncation=True,padding="max_length",max_length=500, add_special_tokens=True)
        
        x1 = torch.LongTensor(x1)
        y1 = torch.Tensor([y1])
        wc = torch.Tensor([wc])
        

        return x1, y1, wc