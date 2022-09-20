import torch
import math
from torch import nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#adapted from https://torchtutorialstaging.z5.web.core.windows.net/beginner/translation_transformer.html
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
  src_seq_len = src.shape[0]
  tgt_seq_len = tgt.shape[0]

  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
  src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

  src_padding_mask = (src == 0).transpose(0, 1)
  tgt_padding_mask = (tgt == 0).transpose(0, 1)
  return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class TransformerModel(nn.Module):
    def __init__(self, num_encoder_layers:int, nhead:int, num_decoder_layers:int,
                 emb_size:int, hist_bins:int, tgt_vocab_size:int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        super(TransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.emb_size = emb_size
        self.src_tok_emb = nn.Linear(1, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, src_mask, src_padding_mask,
                trg,
                tgt_mask,
                tgt_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src.unsqueeze(2)) * math.sqrt(self.emb_size))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg)* math.sqrt(self.emb_size))
        #print('s',src_emb)
        memory = self.transformer_encoder(src_emb, src_mask, None)
        #print('m',memory)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask)
        out = self.generator(outs)
        #print('o',out)
        return out
