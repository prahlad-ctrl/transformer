import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''INPUT EMBEDDING'''

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, dmodel): # number of unique words and dim of vector
        super().__init__()
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dmodel)
        
    def forward(self, x):
        return self.embedding(x) *math.sqrt(self.dmodel) # stabilizes variance
    
'''POSITIONAL ENCODING'''

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, dropout, dmodel): # added the max sequence length and regularization
        super().__init__()
        self.dmodel = dmodel
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, dmodel) # pe dim=(seq_len, dmodel)
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1) #make it as a column (each row as position index)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float()* (-math.log(10000.0)/dmodel)) #computes pos encoding formula- 10000 to the power −2i/dmodel​
        pe[:, 0::2] = torch.sin(pos *div_term) # to even pos
        pe[:, 1::2] = torch.cos(pos *div_term) # to add pos
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]].to(x.device)
        return self.dropout(x)
    
'''LAYER NORMALIZATION'''

class LayerNorm(nn.Module):
    def __init__(self, dmodel, epsilon: float=1e-6): #prevents division by zero
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(dmodel))
        self.bias = nn.Parameter(torch.zeros(dmodel))
        
    def forward(self, x):
        mean = x.mean(dim= -1, keepdim= True)
        std = x.std(dim= -1, keepdim= True)
        return self.alpha* (x-mean)/(std+self.epsilon)+ self.bias # just the standard layer norm eq
    
class FeedForward(nn.Module):
    def __init__(self, dmodel, dff, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear0 = nn.Linear(dmodel, dff)
        self.linear1 = nn.Linear(dff, dmodel) # dmodel -> dff -> dmodel
    
    def forward(self, x):
        x = F.gelu(self.linear0(x)) # original paper uses RELU but GELU is a smoother modern variant
        x = self.linear1(x)
        return self.dropout(x)
    
'''MULTI HEAD ATTENTION'''

class MHA(nn.Module):
    def __init__(self, dmodel, h, dropout):
        super().__init__()
        self.dmodel = dmodel
        assert dmodel %h == 0, "dmodel is not divisible by no. of heads"
        self.h = h
        self.dropout = nn.Dropout(dropout)
        self.d_k = dmodel // h
        self.w_q = nn.Linear(dmodel, dmodel)
        self.w_k = nn.Linear(dmodel, dmodel)
        self.w_v = nn.Linear(dmodel, dmodel)
        self.w_o = nn.Linear(dmodel, dmodel) # Q,K,V (Learnable projections), O (output projection)
    
    def forward(self, q, k, v, mask):
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        # split the multiplied matrices of dim(batch,seq_len,dmodel) into smaller heads -> dim(batch,seq_len,h,d_k)
        # transpose to dim(batch,h,seq_len,d_k) else cooked
        Q = Q.view(Q.shape[0], Q.shape[1], self.h, self.d_k).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.h, self.d_k).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.h, self.d_k).transpose(1, 2)
        d_k = Q.shape[-1]
        
        attention = ((Q @ K.transpose(-2, -1))/math.sqrt(d_k))
        
        # dim= (batch, h, seq_len, d_k)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = attention.softmax(dim= -1)
        attention = self.dropout(attention)
        x = attention @ V
        
        #making it (batch,seq_len,h,d_k) again and then clubbing to (batch,seq_len,dmodel)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.h *self.d_k)
        return self.w_o(x)
    
'''RESIDUAL CONNECTION'''

class Residual(nn.Module):
    def __init__(self, dmodel, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dmodel)
        
    def forward(self, x, sublayer):
        return x+ self.dropout(sublayer(self.norm(x))) # pre-norm or norm before sublayer and add (x → norm → sublayer → dropout → add skip)
        # other way around - post norm- add and then norm
        # return self.norm(x+self.dropout(sublayer(x)))
        
'''N-x ENCODER BLOCK'''

class EncoderBlocks(nn.Module):
    def __init__(self, mha: MHA, ff: FeedForward, dropout:float, dmodel):
        super().__init__()
        self.mha = mha
        self.ff = ff
        self.residual = nn.ModuleList([Residual(dmodel, dropout) for i in range(2)])
        
    def forward(self, x, inp_mask): # inp_mask= to mask the padding tokens that we added
        x = self.residual[0](x, lambda x: self.mha(x,x,x,inp_mask))
        x = self.residual[1](x, self.ff)
        # 2 residual parts: self attention and feed forward
        return x

class Encoder(nn.Module):
    def __init__(self, dmodel, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(dmodel)
        
    def forward(self, x, mask): # mask= to mask the padding tokens that we added
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # the call will be like this- layers=nn.ModuleList([EncoderBlocks for i in range(6)]) for 6 encoder blocks
    
'''N-x DECODER BLOCK'''

class DecoderBlocks(nn.Module):
    def __init__(self, self_mha: MHA, cross_mha: MHA, ff: FeedForward, dropout: float, dmodel):
        super().__init__()
        self.self_mha = self_mha
        self.cross_mha = cross_mha
        self.ff = ff
        self.residual = nn.ModuleList([Residual(dmodel, dropout) for i in range(3)])
    
    def forward(self, x, encoder_out, inp_mask, target_mask):
        x = self.residual[0](x, lambda x: self.self_mha(x,x,x,target_mask))
        x = self.residual[1](x, lambda x: self.cross_mha(x, encoder_out, encoder_out, inp_mask))
        x = self.residual[2](x, self.ff)
        # 3 residual parts: self attention, cross attetion and feed forward
        return x
    
class Decoder(nn.Module):
    def __init__(self, dmodel, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(dmodel)
    
    def forward(self, x, encoder_out, inp_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, inp_mask, target_mask)
        return self.norm(x)
    
'''LINEAR LAYER'''

class LinearLayer(nn.Module):
    def __init__(self, dmodel, vocab_size):
        super().__init__()
        self.linear = nn.Linear(dmodel, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim= -1) # converts into log probablities
    
'''TRANSFORMER WRAPPER'''

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,decoder: Decoder,inp_embed: InputEmbedding,target_embed: InputEmbedding,inp_pos: PositionalEncoding,target_pos: PositionalEncoding,linear: LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.inp_embed = inp_embed
        self.target_embed = target_embed
        self.inp_pos = inp_pos
        self.target_pos = target_pos
        self.linear = linear
        
    def encode(self, inp, inp_mask): # runs embedding -> positional encoding -> encoder stack
        inp = self.inp_embed(inp)
        inp = self.inp_pos(inp)
        return self.encoder(inp, inp_mask)
    
    def decode(self, target, encoder_out, inp_mask, target_mask): # runs embedding -> positional encoding -> decoder stack
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_out, inp_mask, target_mask)
    
    def mapping(self, x): # runs the final linear layer to map back into vocab with dim(batch_size,seq_len,dmodel) -> dim(batch_size,seq_len,vocab_size)
        return self.linear(x)
    
    def run(self, inp_vocab_size, target_vocab_size, inp_seq_len, target_seq_len, dmodel=512, N=6, h=8, dropout=0.1, dff=2048):
        #this function will be used to initialize all the values for different methods
        #input embeddings
        inp_embed = InputEmbedding(inp_vocab_size, dmodel)
        target_embed = InputEmbedding(dmodel, target_vocab_size)
        #positional encoding
        inp_pos = PositionalEncoding(inp_seq_len, dropout, dmodel)
        target_pos = PositionalEncoding(dmodel, target_seq_len, dropout)
        #encoder blocks
        encoder_blocks = []
        for i in range(N):
            encoder_attention = MHA(dmodel, h, dropout)
            encoder_ff = FeedForward(dmodel, dff, dropout)
            encoder_block = EncoderBlocks(encoder_attention, encoder_ff, dropout, dmodel)
            encoder_blocks.append(encoder_block)
        #decoder blocks
        decoder_blocks=[]
        for i in range(N):
            decoder_self_attention = MHA(dmodel,h,dropout)
            decoder_cross_attention = MHA(dmodel,h,dropout)
            decoder_ff = FeedForward(dmodel,dff,dropout)
            decoder_block = DecoderBlocks(decoder_self_attention, decoder_cross_attention, decoder_ff, dropout, dmodel)
            decoder_blocks.append(decoder_block)
        
        encoder = Encoder(dmodel, nn.ModuleList(encoder_blocks))
        decoder = Decoder(dmodel, nn.ModuleList(decoder_blocks))
        linear = LinearLayer(dmodel, target_vocab_size)
        
        # final wrap
        transformer = Transformer(encoder,decoder,inp_embed,target_embed,inp_pos,target_pos,linear)
        
        # params
        for i in transformer.parameters():
            if i.dim()>1:
                nn.init.xavier_uniform_(i) #initialize random weights
        return transformer