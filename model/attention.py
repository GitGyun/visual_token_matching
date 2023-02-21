import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_v, dim_o, num_heads=4, act_fn=nn.GELU,
                 dr=0.1, pre_ln=True, ln=True, residual=True, dim_k=None):
        super().__init__()
        
        if dim_k is None:
            dim_k = dim_q
        
        # heads and temperature
        self.num_heads = num_heads
        self.dim_split_q = dim_q // num_heads
        self.dim_split_v = dim_o // num_heads
        self.temperature = math.sqrt(dim_o)
        self.residual = residual
        
        # projection layers
        self.fc_q = nn.Linear(dim_q, dim_q, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_q, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_o, bias=False)
        self.fc_o = nn.Linear(dim_o, dim_o, bias=False)
        
        # nonlinear activation and dropout
        self.activation = act_fn()
        self.attn_dropout = nn.Dropout(dr)
        
        # layernorm layers
        if pre_ln:
            if dim_q == dim_k:
                self.pre_ln_q = self.pre_ln_k = nn.LayerNorm(dim_q)
            else:
                self.pre_ln_q = nn.LayerNorm(dim_q)
                self.pre_ln_k = nn.LayerNorm(dim_k)
        else:
            self.pre_ln_q = self.pre_ln_k = nn.Identity()
        self.ln = nn.LayerNorm(dim_o) if ln else nn.Identity()
        
    def compute_attention_scores(self, Q, K, mask=None, **kwargs):
        # pre-layer normalization
        Q = self.pre_ln_q(Q)
        K = self.pre_ln_k(K)
        
        # lienar projection
        Q = self.fc_q(Q)
        K = self.fc_k(K)

        # split into multiple heads
        Q_ = torch.cat(Q.split(self.dim_split_q, 2), 0)
        K_ = torch.cat(K.split(self.dim_split_q, 2), 0)

        # scaled dot-product attention with mask and dropout
        A = Q_.bmm(K_.transpose(1, 2)) / self.temperature
        A = A.clip(-1e4, 1e4)
        if mask is not None:
            A.masked_fill(mask, -1e38)
        A = A.softmax(dim=2)
        if mask is not None:
            A.masked_fill(mask, 0)
        A = self.attn_dropout(A)
        
        return A
    
    def project_values(self, V):
        # linear projection
        O = self.fc_v(V)
        
        # residual connection with non-linearity
        if self.residual:
            O = O + self.activation(self.fc_o(O))
        else:
            O = self.fc_o(O)
            
        return O

    def forward(self, Q, K, V, mask=None, get_attn_map=False, disconnect_self_image=False, H=None, W=None, **kwargs):
        # pre-layer normalization
        Q = self.pre_ln_q(Q)
        K = self.pre_ln_k(K)
        
        # lienar projection
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        # split into multiple heads
        Q_ = torch.cat(Q.split(self.dim_split_q, 2), 0)
        K_ = torch.cat(K.split(self.dim_split_q, 2), 0)
        V_ = torch.cat(V.split(self.dim_split_v, 2), 0)
        
        # scaled dot-product attention with mask and dropout
        L = Q_.bmm(K_.transpose(1, 2)) / self.temperature
        L = L.clip(-1e4, 1e4)
        
        # mask
        if mask is not None:
            mask = mask.transpose(1, 2).expand_as(L)
        elif disconnect_self_image:
            assert Q_.size(1) == K_.size(1)
            assert H is not None and W is not None
            N = Q_.size(1) // (H*W)
            mask = torch.block_diag(*[torch.ones(H*W, H*W, device=Q.device) for _ in range(N)]).bool()
        
        if mask is not None:
            L.masked_fill(mask, -1e38)
            
        A = L.softmax(dim=2)
        if mask is not None:
            A.masked_fill(mask, 0)
        A = self.attn_dropout(A)
        
        # apply attention to values
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        
        # layer normalization
        O = self.ln(O)
        
        # residual connection with non-linearity
        if self.residual:
            O = O + self.activation(self.fc_o(O))
        else:
            O = self.fc_o(O)
            
        if get_attn_map:
            return O, A
        else:
            return O