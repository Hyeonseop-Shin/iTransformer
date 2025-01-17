import torch
from torch import nn
from torch.nn import functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(c_in=configs.seq_len,
                                                    d_model=configs.d_model,
                                                    dropout=configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        FullAttention(mask_flag=False,
                                      scale=configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False)
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            conv_layers=None,
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
    
    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).transpose(-1, 1)[:, :, :N]

        return dec_out
    
    def forward(self, x_enc, x_mark_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
