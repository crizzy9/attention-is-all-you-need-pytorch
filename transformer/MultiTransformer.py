import torch
import torch.nn as nn
import numpy as np
from transformer.Models import Encoder, Decoder

class MultiTransformer(nn.Module):
    """
    Class that holds several transformer modules
    with global attention
    """
    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_modules=4, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super(MultiTransformer, self).__init__()

        self.n_modules = n_modules
        self.encoders = nn.ModuleList([Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout) for _ in range(n_modules)])

        self.decoders = nn.ModuleList([Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout) for _ in range(n_modules)])

        self.tgt_word_prjs = nn.ModuleList([nn.Linear(d_model, n_tgt_vocab, bias=False) for _ in range(n_modules)])
        for tgt_word_prj in self.tgt_word_prjs:
            nn.init.xavier_normal_(tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            for i, tgt_word_prj in enumerate(self.tgt_word_prjs):
                tgt_word_prj.weight = self.decoders[i].tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            for i in range(n_modules):
                self.encoders[i].src_word_emb.weight = self.decoders[i].tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        """
        src_seq dim: n_modules x batch_size x max_seq_len (4x1x1600)
        """

        tgt_seq, tgt_pos = tgt_seq[:, :, :-1], tgt_pos[:, :, :-1]
        enc_outputs = [self.encoders[i](src_seq[i], src_pos[i])[0] for i in range(self.n_modules)]
        dec_outputs = [self.decoders[i](tgt_seq[i], tgt_pos[i], src_seq, enc_outputs)[0] for i in range(self.n_modules)]

        seq_logits = []
        for i in range(self.n_modules):
            seq_logits.append(self.tgt_word_prjs[i](dec_outputs[i]) * self.x_logit_scale)

        print("seq logits")
        print(seq_logits.size())
        print(seq_logits)
        print(seq_logits[0])
        # might be a problem?
        return seq_logits.view(-1, seq_logit.size(3))
