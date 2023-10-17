import torch
import torch.nn as nn
from einops import repeat, rearrange
import math
from pytorch_lightning.core import LightningModule
import yaml


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len, dropout_p=0.5):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]
        )


class linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, seq, seq1, seq2):
        a = rearrange(seq, "a b -> a b 1")
        b = rearrange(seq1, "a b -> a b 1")
        c = rearrange(seq2, "a b -> a b 1")
        seq = torch.cat((a, b, c), dim=2)
        out = self.linear(seq)
        out = rearrange(out, "a b 1 -> a b")
        return out


class skeleTransLayer(nn.Module):
    def __init__(self, num_classes, d_model, nhead, seq_len, nlayers, mask=True):
        super(skeleTransLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.mask = mask
        self.hidden = 8

        encoder_layer = nn.TransformerEncoderLayer(2 * self.d_model, nhead, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, 2 * self.d_model))
        self.pos_encoding = PositionalEncoding(2 * self.d_model, max_len=seq_len)
        self.linear = nn.Linear(2 * self.d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 2 * self.d_model))
        self.linear_pre = nn.Linear(2 * self.d_model, 512)

    def forward(self, seq, mask=None):
        b, f, v, c = seq.shape
        out = seq.view(b, f, v * c)
        k, n, m = out.shape
        cls_tokens = repeat(self.cls_token, "1 1 m -> k 1 m", k=k)
        h = torch.cat((cls_tokens, out), dim=1)
        pos_embeddings = repeat(self.pos_embedding, "1 n m -> k n m", k=k)
        h = pos_embeddings[:, : n + 1, :] + h
        if self.mask == True:
            h = self.encoder(src=h, src_key_padding_mask=mask)
        else:
            h = self.encoder(src=h)
        h = h.mean(dim=1)
        res = self.linear(h)
        return res


class InferModel(LightningModule):
    def __init__(self):
        super(InferModel, self).__init__()
        with open("configs/default.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        num_classes = cfg["num_classes"]
        self.model1 = skeleTransLayer(num_classes, 21, 1, 60, 1, mask=False)
        self.model2 = skeleTransLayer(num_classes, 21, 1, 60, 1, mask=False)
        self.model3 = skeleTransLayer(num_classes, 42, 1, 60, 1, mask=False)

    def forward(self, x):
        z = self.model1(x[:, :, :21, :])
        z2 = self.model2(x[:, :, 21:, :])
        z3 = self.model3(x)
        z = (z + z2 + z3) / 3
        return z
