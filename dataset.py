import pandas as pd
import numpy as np
import transformers
import torch
from torch.utils.data import Dataset


# create a dataset class that has three function __init__, __len__, __item__
class RadiologyLabeledDataset(Dataset):
    def __init__(self, tokenizer, df, max_length, target, text_col):
        super(RadiologyLabeledDataset, self).__init__()
        self.text_col = text_col
        self.df = df
        self.tokenizer = tokenizer
        self.target = self.df.loc[:, target].to_numpy()
        self.max_length = max_length
        # self.target_df = self.df[['mass_label', 'aggressive_label', 'met_label_x']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.loc[index, self.text_col]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        ids = inputs["input_ids"]
        # token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "target": torch.tensor(self.target[index], dtype=torch.long),
            "file": [self.df.loc[index, "File Name"]],
            # "target_labels": torch.tensor(self.target_df.iloc[index,:], dtype=torch.long),
        }


# create a dataset class that has three function __init__, __len__, __item__
# For unlabeled data, we don't need to have a target
class RadiologyUnlabeledDataset(Dataset):
    def __init__(self, tokenizer, df, max_length, text_col, text2_col):
        super(RadiologyUnlabeledDataset, self).__init__()
        self.text_col = text_col
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text2_col = text2_col  # this is the text of the other view

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.loc[index, self.text_col]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        ids = inputs["input_ids"]
        # token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "file": [self.df.loc[index, "File Name"]],
            "other_view": [self.df.loc[index, self.text2_col]],
        }
