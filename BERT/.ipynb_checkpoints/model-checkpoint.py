"""
Transformer 구현
"""

import copy

import torch
import torch.nn as nn

from .layers import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, BERT, Encoder, EncoderLayer, Embeddings, Generator


def make_model(src_vocab, dev, N=24,
               d_model=1024, d_ff=1024*4, h=16, dropout=0.1):
    "하이퍼파라미터들로부터 모델 생성"
    c = copy.deepcopy # 클래스 생성 후 deepcopy하여 사용
    
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, dev)
    
    model = BERT(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        Generator(d_model, src_vocab)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

