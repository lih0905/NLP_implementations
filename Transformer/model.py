"""
Transformer 구현
"""

import copy

import torch
import torch.nn as nn

from .layers import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, EncoderDecoder, Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator


def make_model(src_vocab, tgt_vocab, dev, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "하이퍼파라미터들로부터 모델 생성"
    c = copy.deepcopy # 클래스 생성 후 deepcopy하여 사용
    
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, dev)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout),N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

