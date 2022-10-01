from model import *
import torch
from dataset import *
from tqdm import tqdm

ds = Dataset('../hist.csv', '../funcs.csv')
alph = ds.get_alphabet()
print(len(alph))
print(alph)
model = TransformerModel(num_encoder_layers=3, nhead=8, num_decoder_layers=3,
                 emb_size=256, hist_bins=5, tgt_vocab_size=len(alph) + 1,
                 dim_feedforward= 512, dropout = 0.1)
src, tgt = ds[0]
src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

print(tgt)
logits = model(src, src_mask, src_padding_mask, tgt, tgt_mask,
                tgt_padding_mask)
print(alph)
print(logits.shape, tgt.shape)
print(tgt)
print('dim:', logits.flatten(start_dim=0, end_dim=1).shape, tgt.flatten().shape)
print(tgt.flatten())
criterion = nn.CrossEntropyLoss()
loss = criterion(logits.flatten(start_dim=0, end_dim=1), tgt.flatten())
print(loss)