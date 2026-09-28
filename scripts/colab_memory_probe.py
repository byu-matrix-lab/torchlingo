# What a Colab GPU holds for A8's two candidate models: Task #118.
#
# Sent to Coulson on Discord on 2026-09-28, to paste into one Colab cell on a GPU runtime.
# Kept here because his output means nothing without the exact code that produced it.
#
# Each row: model x vocabulary (8,000 SentencePiece pieces, or a large 50,000-word word-level
# vocabulary, which is what the 8a notebook's NMTDataset builds) x sequence length (102 = A8's
# 100-word cap plus <sos>/<eos>; 180 = roughly the same sentences as subwords in A9, ~1.8x).
# Batch 64, AdamW, three full training steps, worst case: every sentence at the cap.
#
# The two models reproduce the a8-benchmark report's parameter counts exactly at vocab 8,000:
# 11,682,624 and 56,436,544. Dry-run tested on CPU at reduced size, including the
# out-of-memory branch; its first run on a real GPU was Coulson's.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchlingo>=0.2.1"], check=True)
import time, torch
from torchlingo.config import Config
from torchlingo.models import SimpleTransformer

assert torch.cuda.is_available(), "Runtime > Change runtime type > GPU"
free, total = torch.cuda.mem_get_info()
print(f"{torch.cuda.get_device_name(0)}: {total/2**30:.1f} GiB, {free/2**30:.1f} free")
MODELS = {
    "11.7M": dict(d_model=256, n_heads=8, num_encoder_layers=3, num_decoder_layers=3, d_ff=1024),
    "56.4M": dict(d_model=512, n_heads=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048),
}
for name, dims in MODELS.items():
    for vocab in (8_000, 50_000):
        for length in (102, 180):
            model = opt = out = None
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                model = SimpleTransformer(src_vocab_size=vocab, tgt_vocab_size=vocab, config=Config(**dims)).cuda()
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                x = torch.randint(4, vocab, (64, length), device="cuda")
                torch.cuda.synchronize(); t = time.time()
                for _ in range(3):
                    out = model(x, x[:, :-1])
                    torch.nn.functional.cross_entropy(out.reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
                    opt.step(); opt.zero_grad()
                torch.cuda.synchronize()
                r = f"peak {torch.cuda.max_memory_allocated()/2**30:5.2f} GiB, {(time.time()-t)/3*1000:.0f} ms/batch"
            except torch.cuda.OutOfMemoryError:
                r = "OUT OF MEMORY"
            print(f"{name}  vocab {vocab:>6,}  len {length}  {r}")
