import torch
import pandas as pd
class Dataset(torch.utils.data.Dataset):

    def __init__(self, hist_path, funcs_path):
        self.hist, self.funcs = pd.read_csv(hist_path, header=None).astype('float'), pd.read_csv(funcs_path, header=None)
        self.init_alph()
        a = self.get_alphabet()
        self.funcs = self.funcs.applymap(lambda x: a[x])

    def __len__(self):
        return len(self.hist)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return torch.tensor(self.hist.iloc[idx].values, dtype=torch.float).T, torch.tensor(self.funcs.iloc[idx].values, dtype=torch.long).T
        else:
            return torch.tensor(self.hist.iloc[idx].values, dtype=torch.float).unsqueeze(0).T, torch.tensor(self.funcs.iloc[idx].values, dtype=torch.long).unsqueeze(0).T
    def init_alph(self):
        alph = {'<PAD>'}
        for rowIndex, row in self.funcs.iterrows():
            for columnIndex, value in row.items():
                alph.add(value)
        self.alph = {value:id for id,value in enumerate(alph)}
        k = ''
        for key, value in self.alph.items():
            if value == 0:
                k = key
                break
        tmp = self.alph['<PAD>']
        self.alph['<PAD>'] = 0
        self.alph[k] = tmp
    def get_alphabet(self):
        return self.alph


