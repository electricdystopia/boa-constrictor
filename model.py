import torch
import torch.nn as nn
import numpy as np
import os

def BoaConstrictor(d_model=256, num_layers=4, vocab_size=256, device="cuda"):
    """ Construct a BoaBytePredictor with smaller model size for Boa experiments. """
    BACKBONE = os.environ.get('BOA_BACKBONE', 'mamba').lower()
    IS_CUDA = torch.cuda.is_available() and device == "cuda"

    if BACKBONE in ('lstm', 'transformer'):
        from alternative_backbones import build_boa_model
        print(f"[BOA] Using {BACKBONE} backbone from alternative_backbones.py")
        model = build_boa_model(
            backbone=BACKBONE,
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
        ).to(device)
        return model
    
    if IS_CUDA:
        device = "cuda"
        from mamba_ssm import Mamba
        from mamba_ssm.utils.generation import InferenceParams
    else:
        device = "cpu"
        from mambapy.mamba import MambaBlock as MambaCPU, MambaConfig

    def tag_mamba_layers_with_ids(model):
        """Give each Mamba layer a unique .layer_idx (0..N-1) for streaming cache."""
        i = 0
        for m in model.modules():
            if IS_CUDA:
                if isinstance(m, Mamba):
                    setattr(m, "layer_idx", i)
                    i += 1
            else:
                if isinstance(m, MambaCPU):
                    setattr(m, "layer_idx", i)
                    i += 1

    def bump_offset(inf, k: int = 1):
        # Most builds use seqlen_offset
        if hasattr(inf, "seqlen_offset"):
            inf.seqlen_offset += k
        elif hasattr(inf, "sequence_length_offset"):
            setattr(inf, "sequence_length_offset", getattr(inf, "sequence_length_offset") + k)
        else:
            # set a best-effort attribute for obscure builds
            setattr(inf, "seqlen_offset", getattr(inf, "seqlen_offset", 0) + k)


    # ---------- Model blocks that pass inference_params ----------
    class MambaBlock(nn.Module):
        def __init__(self, d_model: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            if IS_CUDA:
                self.mamba = Mamba(d_model=d_model)
            else:
                config = MambaConfig(d_model=d_model, n_layers=0, use_cuda=False)
                self.mamba = MambaCPU(config)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        def forward(self, x, inference_params=None):
            y = self.ln1(x)
            if IS_CUDA:
                y = self.mamba(y, inference_params=inference_params)  # <-- stream cache
            else:
                y = self.mamba(y)  # no stream cache in CPU version
            y = self.ln2(y)
            y = self.ff(y)
            return x + y
        
        if not IS_CUDA:
            def init_cache(self, batch_size: int, device):
                # cache for mambapy.MambaBlock.step: (h, inputs)
                d_inner = self.mamba.config.d_inner
                d_conv = self.mamba.config.d_conv
                inputs = torch.zeros(batch_size, d_inner, d_conv - 1, device=device)
                return (None, inputs)

            def step(self, x, cache):
                # x: [B, D] -> [B, D], cache passthrough
                y = self.ln1(x)
                y, cache = self.mamba.step(y, cache)  # mambapy step
                y = self.ln2(y)
                y = self.ff(y)
                return x + y, cache
        
    class BoaBytePredictor(nn.Module):
        """ Mamba model adapted to predict the next byte in a sequence. """
        def __init__(self, d_model=256, num_layers=4, vocab_size=256):
            super().__init__()
            # Embedding for vocab_size possible bytes
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(num_layers)])
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                # Output logits for each of the vocab_size possible next bytes
                nn.Linear(d_model, vocab_size)
            )

        def forward(self, x, inference_params=None):
            h = self.embedding(x)  # [B, L, D]
            for blk in self.blocks:
                h = blk(h, inference_params=inference_params)
            return self.head(h)  # [B, L, 256]
        
        if IS_CUDA:
            @torch.inference_mode()
            def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
                return InferenceParams(max_batch_size=batch_size, max_seqlen=max_len)

            @torch.inference_mode()
            def step(self, byte_t: torch.LongTensor, inf) -> torch.Tensor:
                # byte_t: [B]
                x = self.embedding(byte_t).unsqueeze(1)  # [B, 1, D]
                h = x
                for blk in self.blocks:
                    h = blk(h, inference_params=inf)      # O(1) per token (cached)
                logits_next = self.head(h).squeeze(1)     # [B, vocab_size]
                bump_offset(inf, 1)                       # advance stream
                return logits_next
        else:
            @torch.inference_mode()
            def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
                # returns list of per-block caches
                return [blk.init_cache(batch_size, device) for blk in self.blocks]

            @torch.inference_mode()
            def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
                # byte_t: [B] -> logits: [B, 256]
                h = self.embedding(byte_t)  # [B, D]
                for i, blk in enumerate(self.blocks):
                    h, caches[i] = blk.step(h, caches[i])  # O(1) per token with cache
                logits_next = self.head(h)  # [B, vocab_size]
                return logits_next
    model = BoaBytePredictor(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size)
    tag_mamba_layers_with_ids(model)
    return model

def _aligned_len(n_bytes: int, seq_len: int, batch_size: int) -> int:
    # number of usable bytes that fit whole (batch_size * seq_len) chunks
    block = seq_len * batch_size
    return (n_bytes // block) * block

def make_splits(data_bytes: bytes | np.ndarray, seq_len: int, batch_size: int,
                splits=(0.8, 0.1, 0.1)):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    buf = np.frombuffer(bytes(data_bytes), dtype=np.uint8)
    usable = _aligned_len(len(buf), seq_len, batch_size)
    buf = buf[:usable]

    n = len(buf)
    n_train = _aligned_len(int(n * splits[0]), seq_len, batch_size)
    n_val   = _aligned_len(int(n * splits[1]), seq_len, batch_size)
    n_test  = _aligned_len(n - n_train - n_val, seq_len, batch_size)

    i0, i1, i2 = 0, n_train, n_train + n_val
    train_bytes = buf[i0:i1].tobytes()
    val_bytes   = buf[i1:i2].tobytes()
    test_bytes  = buf[i2:i2+n_test].tobytes()

    return train_bytes, val_bytes, test_bytes

class ByteDataloader:
    """ Simple dataloader that yields batches of bytes. """
    def __init__(self, data_bytes, seq_len=1048576, batch_size=1, device="cuda"):
        self.data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pos = 0
        self.device = device
    def __len__(self):
        """ Returns the total number of batches in the dataset. """
        return len(self.data_bytes) // (self.seq_len * self.batch_size)
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos + self.seq_len * self.batch_size > len(self.data_bytes):
            self.pos = 0  # reset for simplicity
            raise StopIteration
        
        batch_indices = np.arange(self.pos, self.pos + self.seq_len * self.batch_size)
        batch_indices = batch_indices.reshape(self.batch_size, self.seq_len)
        self.pos += self.seq_len * self.batch_size
        
        batch = self.data_bytes[batch_indices]
        return torch.tensor(batch, dtype=torch.long).to(self.device)
    
