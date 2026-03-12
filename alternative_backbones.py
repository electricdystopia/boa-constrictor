"""
alternative_backbones.py
========================
Drop-in replacements for the BytewiseMamba backbone used in BOA Constrictor.

USAGE
-----
In model.py (or wherever BoaConstrictor is assembled), swap the backbone:

    # Original:
    from model import BoaConstrictor          # uses BytewiseMamba internally

    # With this file, replace the backbone by passing backbone="lstm" or "transformer":
    from alternative_backbones import build_boa_model
    model = build_boa_model(backbone="lstm", d_model=256, num_layers=4)
    model = build_boa_model(backbone="transformer", d_model=256, num_layers=4)

The output interface is identical to BoaConstrictor:
    input  : (batch, seq_len)         -- integer byte values 0-255
    output : (batch, seq_len, 256)    -- logits over next-byte distribution

DESIGN NOTES
------------
- Both backbones share the same byte embedding + output projection.
- LSTM:        O(seq_len) per step, constant memory, CPU-friendly, stateful.
- Transformer: O(seq_len^2) attention, highly parallel on GPU, no recurrence.
- Both are trained with cross-entropy loss exactly as the original Mamba model.

HOW TO RUN A QUICK BENCHMARK (CPU, no training, just throughput)
-----------------------------------------------------------------
    python alternative_backbones.py --benchmark --backbone lstm
    python alternative_backbones.py --benchmark --backbone transformer
"""

import math
import time
import argparse
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1.  Shared building blocks
# ---------------------------------------------------------------------------

class ByteEmbedding(nn.Module):
    """Map raw byte values (0-255) to dense vectors of size d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.num_embeddings = 256   
        self.embed = nn.Embedding(256, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) integers
        return self.embed(x)  # -> (batch, seq_len, d_model)


class ByteProjection(nn.Module):
    """Project hidden states back to logits over the 256-byte alphabet."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 256)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq_len, d_model)
        return self.proj(h)   # -> (batch, seq_len, 256)


# ---------------------------------------------------------------------------
# 2.  LSTM backbone
# ---------------------------------------------------------------------------

class LSTMBackbone(nn.Module):
    """
    Stacked bidirectional-optional LSTM backbone.

    For causal (autoregressive) compression we MUST use unidirectional LSTM
    (bidirectional=False) so each position only sees past bytes.

    Args:
        d_model    : hidden size == LSTM hidden_size
        num_layers : number of stacked LSTM layers
        dropout    : dropout between LSTM layers (0 to disable)
    """

    def __init__(self, d_model: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,       # input/output shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,    # MUST be causal
        )

    def forward(self, x: torch.Tensor, state=None):
        """
        Args:
            x     : (batch, seq_len, d_model)   embedded byte sequence
            state : optional (h_n, c_n) from previous chunk for streaming
        Returns:
            out   : (batch, seq_len, d_model)   hidden states
            state : (h_n, c_n) to pass to next chunk
        """
        out, state = self.lstm(x, state)
        return out, state


# ---------------------------------------------------------------------------
# 3.  Transformer backbone (GPT-style, causal)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask.
    Position i can only attend to positions 0..i (no future leakage).
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 32768):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Causal mask: upper-triangular = -inf
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                    # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)        # each (B, T, C)

        # Reshape to (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale   # (B, heads, T, T)

        # Apply causal mask
        causal = self.mask[:, :, :T, :T]
        attn = attn.masked_fill(causal == 0, float("-inf"))
        attn = attn.softmax(dim=-1)

        out = attn @ v                        # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block (GPT-2 style)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 dropout: float = 0.1, max_seq_len: int = 32768):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerBackbone(nn.Module):
    """
    Stacked causal Transformer (GPT-style) backbone.

    Positional encoding: learned absolute positions up to max_seq_len.
    This matches BOA's fixed seq_len chunks naturally.

    NOTE: O(seq_len^2) memory for attention — seq_len=32768 needs lots of VRAM.
          Reduce seq_len or use sliding-window if GPU memory is limited.

    Args:
        d_model     : embedding + hidden dimension
        num_layers  : number of Transformer blocks
        n_heads     : number of attention heads (default: d_model // 64)
        max_seq_len : maximum sequence length (must be >= config seq_len)
    """

    def __init__(self, d_model: int, num_layers: int, n_heads: int = None,
                 dropout: float = 0.1, max_seq_len: int = 32768):
        super().__init__()

        if n_heads is None:
            # Default: head_dim ≈ 64, a common sweet spot
            n_heads = max(1, d_model // 64)

        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout,
                             max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, d_model)   embedded byte sequence
        Returns:
            out : (batch, seq_len, d_model)  contextualised representations
        """
        B, T, C = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = self.drop(x + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_final(x)


# ---------------------------------------------------------------------------
# 4.  Unified model wrapper (same interface as BoaConstrictor)
# ---------------------------------------------------------------------------

class AlternativeBoaModel(nn.Module):
    """
    Full byte-level neural compressor model with a swappable backbone.

    Replaces BoaConstrictor from model.py while keeping the same:
      - input  : (batch, seq_len) int tensor, byte values 0-255
      - output : (batch, seq_len, 256) logit tensor

    Args:
        backbone   : "lstm" | "transformer"
        d_model    : hidden/embedding dimension
        num_layers : number of backbone layers
        **kwargs   : forwarded to backbone constructors
    """

    def __init__(self, backbone: str = "lstm", d_model: int = 256,
                 num_layers: int = 4, **kwargs):
        super().__init__()
        self.backbone_name = backbone
        self.embedding = ByteEmbedding(d_model)

        if backbone == "lstm":
            self.backbone = LSTMBackbone(d_model, num_layers,
                                         dropout=kwargs.get("dropout", 0.1))
            self._is_recurrent = True

        elif backbone == "transformer":
            self.backbone = TransformerBackbone(
                d_model, num_layers,
                n_heads=kwargs.get("n_heads", None),
                dropout=kwargs.get("dropout", 0.1),
                max_seq_len=kwargs.get("max_seq_len", 32768),
            )
            self._is_recurrent = False
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Use 'lstm' or 'transformer'.")

        self.proj = ByteProjection(d_model)

    def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
        d = device or next(self.parameters()).device
        if self._is_recurrent:
            # LSTM: return (h_0, c_0) zero state
            n_layers = self.backbone.lstm.num_layers
            d_model  = self.backbone.lstm.hidden_size
            h0 = torch.zeros(n_layers, batch_size, d_model, device=d)
            return (h0, h0.clone())
        else:
            # Transformer: return a token buffer
            return {'tokens': torch.zeros(batch_size, 0, dtype=torch.long, device=d)}

    @torch.inference_mode()
    def step(self, byte_t: torch.Tensor, cache):
        if self._is_recurrent:
            emb = self.embedding(byte_t).unsqueeze(1)   # [B, 1, D]
            out, cache = self.backbone.lstm(emb, cache)  # out: [B, 1, D]
            logits = self.proj(out.squeeze(1))           # [B, 256]
            return logits
        else:
            new_tok = byte_t.unsqueeze(1)
            cache['tokens'] = torch.cat([cache['tokens'], new_tok], dim=1)
            logits_all, _ = self.forward(cache['tokens'])  # [B, L, 256]
            return logits_all[:, -1, :]                    # [B, 256]

    def forward(self, x: torch.Tensor, state=None):
        """
        Args:
            x     : (batch, seq_len) integer byte values
            state : optional RNN state (only used by LSTM backbone)
        Returns:
            logits : (batch, seq_len, 256)
            state  : updated RNN state (None for Transformer)
        """
        emb = self.embedding(x)                        # (B, T, d_model)

        if self._is_recurrent:
            hidden, state = self.backbone(emb, state)
        else:
            hidden = self.backbone(emb)
            state = None

        logits = self.proj(hidden)                 # (B, T, 256)
        return logits, state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 5.  Factory function (mirrors BOA config dict interface)
# ---------------------------------------------------------------------------

def build_boa_model(backbone: str = "lstm", d_model: int = 256,
                    num_layers: int = 4, **kwargs) -> AlternativeBoaModel:
    """
    Build and return an AlternativeBoaModel.

    Example usage (mirroring boa YAML config):
        model = build_boa_model(
            backbone="lstm",
            d_model=256,
            num_layers=4
        )

    To integrate with BOA's main.py, replace the line:
        model = BoaConstrictor(config["model"]["d_model"],
                               config["model"]["num_layers"])
    with:
        model = build_boa_model(
            backbone=config["model"].get("backbone", "lstm"),
            d_model=config["model"]["d_model"],
            num_layers=config["model"]["num_layers"],
        )
    """
    return AlternativeBoaModel(backbone=backbone, d_model=d_model,
                               num_layers=num_layers, **kwargs)


# ---------------------------------------------------------------------------
# 6.  Simple benchmark (run standalone to test throughput without BOA)
# ---------------------------------------------------------------------------

def run_benchmark(backbone: str, d_model: int = 256, num_layers: int = 4,
                  seq_len: int = 1024, batch_size: int = 4,
                  n_iters: int = 20, device: str = "cpu"):
    """
    Measure forward-pass throughput (bytes/sec) for a given backbone.
    No training, no range coder — pure model inference speed.
    """
    model = build_boa_model(backbone=backbone, d_model=d_model,
                            num_layers=num_layers).to(device)
    model.eval()

    x = torch.randint(0, 256, (batch_size, seq_len), device=device)
    total_bytes = batch_size * seq_len

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            model(x)

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            logits, _ = model(x)
    elapsed = time.perf_counter() - start

    throughput = (total_bytes * n_iters) / elapsed
    params = model.count_parameters()

    print(f"\n{'='*50}")
    print(f"  Backbone     : {backbone.upper()}")
    print(f"  d_model      : {d_model}, num_layers: {num_layers}")
    print(f"  seq_len      : {seq_len}, batch_size: {batch_size}")
    print(f"  Device       : {device}")
    print(f"  Parameters   : {params:,}")
    print(f"  Throughput   : {throughput:,.0f} bytes/sec")
    print(f"  Avg latency  : {(elapsed/n_iters)*1000:.2f} ms / batch")
    print(f"{'='*50}\n")

    return {"backbone": backbone, "throughput_bps": throughput,
            "params": params, "latency_ms": (elapsed / n_iters) * 1000}

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark alternative BOA backbones"
    )
    parser.add_argument("--benchmark", action="store_true",
                        help="Run throughput benchmark")
    parser.add_argument("--backbone", type=str, default="lstm",
                        choices=["lstm", "transformer"],
                        help="Backbone to benchmark")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--compare", action="store_true",
                        help="Benchmark both backbones and compare")
    parser.add_argument("--print-integration", action="store_true",
                        help="Print integration instructions")
    args = parser.parse_args()

    if args.compare:
        results = []
        for b in ["lstm", "transformer"]:
            r = run_benchmark(b, d_model=args.d_model,
                              num_layers=args.num_layers,
                              seq_len=args.seq_len,
                              batch_size=args.batch_size,
                              device=args.device)
            results.append(r)

        print("\nCOMPARISON SUMMARY")
        print(f"{'Backbone':<15} {'Params':>12} {'Throughput (B/s)':>18} {'Latency (ms)':>14}")
        print("-" * 62)
        for r in results:
            print(f"{r['backbone']:<15} {r['params']:>12,} "
                  f"{r['throughput_bps']:>18,.0f} {r['latency_ms']:>14.2f}")

    elif args.benchmark:
        run_benchmark(args.backbone, d_model=args.d_model,
                      num_layers=args.num_layers,
                      seq_len=args.seq_len,
                      batch_size=args.batch_size,
                      device=args.device)

    else:
        # Smoke test both models
        print("Running smoke test...")
        for b in ["lstm", "transformer"]:
            m = build_boa_model(backbone=b, d_model=128, num_layers=2)
            x = torch.randint(0, 256, (2, 64))
            logits, state = m(x)
            assert logits.shape == (2, 64, 256), f"Wrong output shape: {logits.shape}"
            print(f"  {b}: OK — output shape {logits.shape}, "
                  f"params={m.count_parameters():,}")
        print("All smoke tests passed.")
