import tensorflow as tf

from .positional_encoding import PositionalEmbedding
from .attention import GlobalSelfAttention
from .feedforward import FeedForward
from .config import EncoderConfig


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x
  
class Encoder(tf.keras.layers.Layer):
  def __init__(self, cfg: EncoderConfig, vocab_size=1000):
    super().__init__()

    self.d_model = cfg.d_model
    self.num_layers = cfg.num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=cfg.d_model)

    self.enc_layers = [
        EncoderLayer(d_model=cfg.d_model,
                     num_heads=cfg.num_heads,
                     dff=cfg.dff,
                     dropout_rate=cfg.dropout_rate)
        for _ in range(cfg.num_layers)]
    self.dropout = tf.keras.layers.Dropout(cfg.dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.