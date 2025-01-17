from torch import nn
from torch.nn import functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,    # query, key, value
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)   # [B, in_num_query, d_model]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # [B, d_ff, in_num_query]
        y = self.dropout(self.conv2(y).transpose(-1, 1))    # [B, in_num_query, d_model]

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()

        self.attn_layers = nn.ModuleList(attn_layers)   # Composed with EncoderLayers
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x: [B, L, d_model]

        attentions = list()
        if self.conv_layers is not None:    # conv_layers will be None
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attentions.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attentions.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attentions.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attentions
