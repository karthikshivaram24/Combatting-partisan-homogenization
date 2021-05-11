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
        encoded_x1 = self.tokenizer.encode_plus(text=x1, add_special_tokens=True, max_length = 500, pad_to_max_length=True, return_attention_mask = True,truncation=True, return_tensors = 'pt')
        encoded_x2 = self.tokenizer.encode_plus(text=x2, add_special_tokens=False, max_length = 1, pad_to_max_length=True, return_attention_mask = False,truncation=True, return_tensors = 'pt')
        
        x1 = encoded_x1["input_ids"].flatten()
        am1 = encoded_x1["attention_mask"].flatten()
        x2 = encoded_x2["input_ids"].flatten()
        y1 = torch.Tensor([y1])
        y2 = torch.Tensor([y2])
        wc = torch.Tensor([wc])
        return x1,am1,x2,y1,y2, wc

class CPDatasetMT_with_Text(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = load_tokenizer()
        print(self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        text = self.df["text"].iloc[index]
        x1 = self.df["text"].iloc[index]
        x2 = self.df["context_word"].iloc[index]
        y1 = self.df["class_label"].iloc[index]
        y2 = self.df["word_label"].iloc[index]
        wc = self.df["which_cluster"].iloc[index]
        
        # replace context word from positive samples
        x1 = x1.replace(x2,"")
        encoded_x1 = self.tokenizer.encode_plus(text=x1, add_special_tokens=True, max_length = 500, pad_to_max_length=True, return_attention_mask = True,truncation=True, return_tensors = 'pt')
        encoded_x2 = self.tokenizer.encode_plus(text=x2, add_special_tokens=False, max_length = 1, pad_to_max_length=True, return_attention_mask = False,truncation=True, return_tensors = 'pt')
        
        x1 = encoded_x1["input_ids"].flatten()
        am1 = encoded_x1["attention_mask"].flatten()
        x2 = encoded_x2["input_ids"].flatten()
        y1 = torch.Tensor([y1])
        y2 = torch.Tensor([y2])
        wc = torch.Tensor([wc])
        return x1,am1,x2,y1,y2, wc,text

class CPDatasetST(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = load_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        x1 = self.df["processed_all"].iloc[index]
        y1 = self.df["binary_ps"].iloc[index]
        wc = self.df["which_cluster"].iloc[index]
        
#         x1 = self.tokenizer.encode(x1,truncation=True,padding="max_length",max_length=500, add_special_tokens=True)
        
        encoded = self.tokenizer.encode_plus(text=x1, add_special_tokens=True, max_length = 500, pad_to_max_length=True, return_attention_mask = True,truncation=True, return_tensors = 'pt')
        
        x1 = encoded["input_ids"].flatten()
        am1 = encoded["attention_mask"].flatten()
        y1 = torch.Tensor([y1])
        wc = torch.Tensor([wc])
        

        return x1,am1, y1, wc
    
class CPDatasetST_with_Text(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = load_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        text = self.df["processed_all"].iloc[index]
        x1 = self.df["processed_all"].iloc[index]
        y1 = self.df["binary_ps"].iloc[index]
        wc = self.df["which_cluster"].iloc[index]
        
#         x1 = self.tokenizer.encode(x1,truncation=True,padding="max_length",max_length=500, add_special_tokens=True)
        
        encoded = self.tokenizer.encode_plus(text=x1, add_special_tokens=True, max_length = 500, pad_to_max_length=True, return_attention_mask = True,truncation=True, return_tensors = 'pt')
        
        x1 = encoded["input_ids"].flatten()
        am1 = encoded["attention_mask"].flatten()
        y1 = torch.Tensor([y1])
        wc = torch.Tensor([wc])

        return x1,am1, y1, wc, text