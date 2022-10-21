import pandas as pd
import torch
import numpy as np

from model import *
import torch
from dataset import *
from tqdm import tqdm
import argparse


"""hist, funcs = pd.read_csv('./hist.csv', header = None).astype('float'), pd.read_csv('./funcs.csv', header = None)

alph = get_alphabet()
print(alph)
print(len(alph))
model = TransformerModel(num_encoder_layers=3, nhead=8, num_decoder_layers=3,
                 emb_size=128, hist_bins=5, tgt_vocab_size=len(alph),
                 dim_feedforward= 512, dropout = 0.1)
print(hist.iloc[0].values)
src = torch.tensor(hist.iloc[0].values, dtype=torch.float, device=device).unsqueeze(0).T
tgt = torch.tensor([alph[el] for el in funcs.iloc[0].values], dtype=torch.long, device=device).unsqueeze(0).T

src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

logits = model(src, tgt, tgt_mask,
                tgt_padding_mask)
print(logits.shape)

"""

def train_epoch_transformer(model, train_loader, optimizer, criterion, batch_size):
  model.train()
  total_loss = 0
  num_correct = 0
  total_items = 0
  for src, tgt in tqdm(train_loader):
      src = src.to(device).squeeze().T
      tgt = tgt.to(device).squeeze().T
      #print(src.shape, tgt.shape)
      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

      logits = model(src, src_mask, src_padding_mask, tgt, tgt_mask,
                     tgt_padding_mask)

      #print(logits)

      optimizer.zero_grad()

      loss = criterion(logits.flatten(start_dim=0, end_dim=1), tgt.flatten())

      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      total_items += (tgt != 0).sum(dim=(0,1))

      num_correct += (torch.logical_and((logits.argmax(dim=2) == tgt), (tgt != 0))).sum(dim=(0,1))
  return total_loss / len(train_loader), num_correct / total_items

def test_epoch_transformer(model, test_loader, criterion, batch_size):
  model.eval()
  total_loss = 0
  num_correct = 0
  total_items = 0
  for src, tgt in tqdm(test_loader):
      src = src.to(device).squeeze().T
      tgt = tgt.to(device).squeeze().T
      #print(src.shape, tgt.shape)
      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

      logits = model(src, src_mask, src_padding_mask, tgt, tgt_mask,
                     tgt_padding_mask)

      #print(logits)


      loss = criterion(logits.flatten(start_dim=0, end_dim=1), tgt.flatten())

      """if torch.isnan(loss):
          print(logits)
          print(tgt)
          print(loss)"""
      total_loss += loss.item()
      total_items += (tgt != 0).sum(dim=(0,1))

      num_correct += (torch.logical_and((logits.argmax(dim=2) == tgt), (tgt != 0))).sum(dim=(0,1))
  return total_loss / len(test_loader), num_correct / total_items

def train(model, train_dataset, test_dataset, batch_size=8, epochs=100):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
    for e in range(epochs):
        train_loss, train_acc = train_epoch_transformer(model, train_loader, optim, criterion, batch_size=batch_size)
        test_loss, test_acc = test_epoch_transformer(model, test_loader, criterion, batch_size=batch_size)
        print(
            f'Epoch: {e + 1} Training Loss: {train_loss} Training Accuracy: {train_acc} Test Loss: {test_loss} Test Accuracy: {test_acc}')

parser = argparse.ArgumentParser(description='Trains a transformer on histogram and expression data')
parser.add_argument('--path_hist', type=str, default='hist.csv')
parser.add_argument('--path_funcs', type=str, default='funcs.csv')
parser.add_argument('--encoder_layers', type=int, default=3)
parser.add_argument('--decoder_layers', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=32)
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--dim_feedforward', type=int, default=2048)
parser.add_argument('--dropout', type=float, default=0.1)
args = parser.parse_args()

ds = Dataset(args.path_hist, args.path_funcs)
train_idx = list(range(0, int(9 * len(ds) / 10)))
test_idx = list(range(int(9 * len(ds) / 10), len(ds)))
train_dataset = torch.utils.data.Subset(ds, train_idx)
test_dataset = torch.utils.data.Subset(ds, test_idx)
model = TransformerModel(num_encoder_layers=args.encoder_layers, nhead=args.num_heads, num_decoder_layers=args.decoder_layers,
                 emb_size=args.emb_size, hist_bins=5, tgt_vocab_size=len(ds.get_alphabet()),
                 dim_feedforward=args.dim_feedforward, dropout=args.dropout)
train(model, train_dataset, test_dataset)

