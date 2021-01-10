import copy
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def clones(module, N):
    "N개의 동일한 레이어를 생성한다."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "마스킹 생성"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 # 대각선 아래만 True, 위에는 False


class SublayerConnection(nn.Module):
    """
    임의의 sublayer에 대해 residual connection과 layer normalization을 적용한다.
    이때 드랍아웃 또한 적용하여 오버피팅을 방지한다.
    구현의 편의를 위해 normalization은 입력에서 적용한다.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # 입력으로 x 뿐 아니라 sublayer를 받아야함!
        return x + self.dropout(sublayer(self.norm(x)))
            

class PositionwiseFeedForward(nn.Module):
    """
    FFN 구현. d_model 차원의 입력값을 d_ff 차원의 공간에 임베딩한 후 
    relu 함수를 취한 후 다시 d_model 차원으로 임베딩한다.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
            
          
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "h : head 갯수, d_model : 모델 차원."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 # 입력 벡터를 h개로 나눠야하므로 체크 필요
        self.d_k = d_model // h # 정수로 처리하기 위해 // 사용
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 위 그림에 linear module이 4개 필요
        self.attn = None 
        self.dropout = nn.Dropout(p=dropout)

    def attention(query, key, value, mask=None, dropout=None):
        "Scaled Dot Product Attention을 구현한다."
        d_k = query.size(-1) # query size = (length, d_k)
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
        # scores size = (length, length)
        if mask is not None:
            # 마스크가 False(=0)인 점은 -10^9 로 채워 학습되는 것을 방지
            scores = scores.masked_fill(mask ==0 , -1e9) 
        p_attn = F.softmax(scores, dim = -1)
        # p_attn size = (length)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn        
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 마스크를 모든 head에 동일하게 적용한다.
            mask = mask.unsqueeze(1) # 마스크 벡터의 head 부분에 해당하는 값을 1로 채움
        nbatches = query.size(0) # query size : (nbatches, length, d_model)
        
        # 1) Q, K, V 에 linear projection을 취한 후 사이즈 변환 => h x d_k
        query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            for l, x in zip(self.linears, (query, key, value))]
        # size : (nbatches, h, length, d_k)
        
        # 2) (length, d_k) 사이즈의 벡터들에 attention을 h번 적용
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x size : (nbatches, h, length, d_k)
        
        # 3) x를 이어붙인 후 마지막 linear projection 적용
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x) # size : (nbatches, length, d_model)


class EncoderLayer(nn.Module):
    "인코더 레이어는 셀프어텐션과 FFN으로 이루어져 있다."
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        """
        sublayer는 입력값 x뿐 아니라 적용할 layer도 입력으로 받아야함.
        sublayer[0]은 self_attn, sublayer[1]은 FFN을 적용 layer로 사용한다.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        # 입력값도 패딩으로 인한 마스킹 필요
        return self.sublayer[1](x, self.feed_forward)
    
    
class Encoder(nn.Module):
    "인코더는 N개의 레이어로 구성된다."
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "입력값(과 마스크)를 각각의 레이어에 통과시킨다."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)    

    
class DecoderLayer(nn.Module):
    "셀프어텐션, 소스어텐션, FFN으로 구성한다."
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # 입력값을 연산
        self.src_attn = src_attn # 쿼리는 입력값, 키와 밸류는 인코더의 출력값을 입력
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        sublayer[0]은 self_attn, sublayer[1]은 src_attn,
        sublayer[2]는 FFN을 적용 layer로 사용한다. 
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
class Decoder(nn.Module):
    "N개의 디코더 레이어로 구성"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        "memory는 인코더의 출력값으로, 디코더 레이어의 두번째 self_attn의 입력에 사용"
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "입력값에 PE를 더하는 모듈 구성."
    def __init__(self, d_model, dropout, dev, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.pe = torch.zeros([max_len, d_model],requires_grad=False).to(dev)
        position = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                            - (math.log(10000.0) / d_model)) # (d_model/2)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = (self.pe).unsqueeze(0) # (1, max_len, d_model)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] # x의 length까지만 필요함
        return self.dropout(x)
    

class EncoderDecoder(nn.Module):
    """
    표준적인 인코더-디코더 모델 구현
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self, src, src_mask):
        "입력값(단어 index)을 받아서 인코딩"
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        "결과값(index)과 인코더의 어텐션의 출력을 받아서 디코딩"
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "인코더의 출력값을 메모리 입력값으로 받아서 순전파"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)    
    
    
class Generator(nn.Module):
    "일반적인 linear + softmax 계층 구현."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        # 벡터를 단어로 변환
        
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    