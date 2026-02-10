import math
import torch
import torch.nn as nn
import numpy as np
import time

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        return self.pe[:, :seq_len, :]

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out

class BasicTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, action_tokens, context_tokens, context_mask):

        x1 = self.norm1(action_tokens)
        attn_out, _ = self.attn1(x1, x1, x1)
        x1 = action_tokens + attn_out

        x2 = self.norm2(x1)
        attn_out2, _ = self.attn2(x2, context_tokens, context_tokens, key_padding_mask=context_mask)
        x2 = x1 + attn_out2

        x3 = self.norm3(x2)
        ff_out = self.ff(x3)
        x = x2 + ff_out

        return x

class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = nn.Linear(z_channels, model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels)
            for _ in range(num_res_blocks)
        ])

        self.final_layer = FinalLayer(model_channels, out_channels)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
class ResBlock(nn.Module):
    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ActionExpert(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.horizon = cfg.horizon
        self.action_dim = cfg.action_dim

        embed_dim = cfg.embed_dim
        hidden_dim = 4*cfg.embed_dim
        action_dim = cfg.action_dim
        state_dim = cfg.state_dim
        num_heads = cfg.num_heads
        num_layers = cfg.num_layers
        dropout = cfg.dropout
        self.num_inference_timesteps = 5

        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim)

        self.time_mlp = nn.Sequential(
                    SinusoidalPosEmb(embed_dim),
                    nn.Linear(embed_dim, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                )

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                   hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)
        self.state_encoder = MLP(input_dim=state_dim, hidden_dim=hidden_dim, output_dim=embed_dim)
        self.action_encoder = MLP(input_dim=action_dim, hidden_dim=hidden_dim, output_dim=embed_dim)

        self.mlp_head = SimpleMLPAdaLN(
            in_channels=action_dim,
            model_channels=1024,
            out_channels=action_dim,
            z_channels=embed_dim,
            num_res_blocks=5,
        )

        self.masked_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.diffusion_batch_mul = 10

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.horizon)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.from_numpy(np.array(orders)).to(device=self.device, dtype=torch.long)
        return orders

    def random_masking(self, bsz, seq_len, orders):
        # generate token mask
        k_rand = torch.randint(1, self.horizon, (1,), device=self.device).item()
        num_masked_tokens = self.horizon if (torch.rand((), device=self.device) < 0.3) else k_rand
        mask = torch.zeros(bsz, seq_len, device=self.device)
        idx = orders[:, :num_masked_tokens]
        mask = torch.scatter(mask, dim=-1, index=idx,
                             src=torch.ones_like(idx, dtype=mask.dtype, device=self.device))
        return mask

    def forward(self, fused_tokens, fused_mask, state, actions_gt):
        B, H, _ = actions_gt.size()
        ## context tokens
        context_tokens = fused_tokens
        state_emb = self.state_encoder(state)
        context_tokens = torch.cat([context_tokens, state_emb], dim=1) 
        ## context mask
        context_mask = (fused_mask != 0)
        context_mask = torch.cat([context_mask, torch.ones(B, 1, device=self.device, dtype=torch.bool)], dim=1)
        context_mask = ~context_mask
        ## orders
        orders = self.sample_orders(bsz=B)
        ## masked action tokens
        masked_token = self.masked_token.repeat(B, H, 1)
        action_emb = self.action_encoder(actions_gt)
        mask = self.random_masking(bsz=B, seq_len=H, orders=orders)
        m = mask.unsqueeze(-1)
        actions_new = action_emb * (1 - m) + masked_token * m
        ## transformer
        action_query = actions_new + self.time_pos_enc(H).repeat(B, 1, 1)
        x = action_query
        for block in self.transformer_blocks:
            x = block(x, context_tokens, context_mask)
        x = self.norm_out(x)
        x = x.view(B*H, -1)
        # actions_gt_seq: (B*H, D)
        actions_gt_seq = actions_gt.view(B * H, -1)

        M = getattr(self, "diffusion_batch_mul", 1)  # 例如 4

        if M > 1:
            x_rep = x.repeat_interleave(M, dim=0)                  # (B*H*M, C)
            actions_gt_rep = actions_gt_seq.repeat_interleave(M, 0) # (B*H*M, D)
        else:
            x_rep = x
            actions_gt_rep = actions_gt_seq

        t = torch.rand((B*H*M,), device=self.device).to(self.dtype)
        time_emb = self.time_mlp(t)

        # noise + interpolate
        noise_seq = torch.randn_like(actions_gt_rep)
        t_broadcast = t.to(dtype=actions_gt_rep.dtype).view(-1, 1)
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_rep

        pred_velocity = self.mlp_head(action_intermediate_seq, time_emb, x_rep)  # (B*H*M, D)

        # reshape 回来，方便你继续用原来的 loss 逻辑
        pred_velocity = pred_velocity.view(B, H, M, -1)
        noise_seq = noise_seq.view(B, H, M, -1)
        m = m.unsqueeze(2).expand(B, H, M, 1)  # mask 也扩展到 M 份

        return pred_velocity, noise_seq, m

    def get_actions(self, fused_tokens, fused_mask, state, K=16):
        B = fused_tokens.size(0)
        H = self.horizon
        D = self.action_dim
        ## context tokens
        context_tokens = fused_tokens
        state_emb = self.state_encoder(state)
        context_tokens = torch.cat([context_tokens, state_emb], dim=1)
        ## context mask
        context_mask = (fused_mask != 0)
        context_mask = torch.cat([context_mask, torch.ones(B, 1, device=self.device, dtype=torch.bool)], dim=1)
        context_mask = ~context_mask
        ## order
        orders = self.sample_orders(bsz=B)
        ## initialization of action acd mask
        actions = torch.zeros(B, H, D, device=self.device)
        mask = torch.ones(B, H, device=self.device)
        ## iterative generation
        num_iter = (H + K - 1) // K
        for it in range(num_iter):
            ## current indices to fill
            start = it * K
            end = min((it + 1) * K, H)
            idx_to_fill = orders[:, start:end] 
            k_now = idx_to_fill.size(1)
            ## masked action tokens
            masked_token = self.masked_token.repeat(B, H, 1)
            action_emb = self.action_encoder(actions)
            m = mask.unsqueeze(-1)
            action_new = action_emb * (1 - m) + masked_token * m
            ## transformer
            action_query = action_new + self.time_pos_enc(H).repeat(B, 1, 1)
            x = action_query
            for block in self.transformer_blocks:
                x = block(x, context_tokens, context_mask)
            x = self.norm_out(x)
            x_sel = x.gather(1, idx_to_fill.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            x_sel = x_sel.reshape(B * k_now, -1)
            ## flow matching inference
            action_seq = torch.randn(B * k_now, D, device=self.device)
            N = self.num_inference_timesteps
            dt = 1.0 / N
            for i in range(N):
                t = i / N
                t = torch.full((B*k_now,), t, device=self.device, dtype=self.dtype)
                time_emb = self.time_mlp(t)
                pred = self.mlp_head(action_seq, time_emb, x_sel)  
                action_seq = action_seq + dt * pred
            ## update actions and mask
            action_seq = action_seq.view(B, k_now, D)
            actions.scatter_(dim=1, index=idx_to_fill.unsqueeze(-1).expand(-1, -1, D), src=action_seq)
            mask.scatter_(dim=1, index=idx_to_fill, src=torch.zeros_like(idx_to_fill, dtype=mask.dtype))

        return actions

    @property
    def device(self):
      
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        
        return next(self.parameters()).dtype

