import argparse, json
from pathlib import Path
import numpy as np
import torch
from gpt_model import GPTConfig, GPTLanguageModel

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="../data/kinyarwanda.bin")
    ap.add_argument("--out_dir", default="../model/checkpoint")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--eval_interval", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--vocab_size", type=int, default=50257)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    cfg_path = out_dir / "config.json"

    data = torch.tensor(np.fromfile(args.bin, dtype=np.uint16), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
    )
    model = GPTLanguageModel(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for it in range(args.iters):
        xb, yb = get_batch(train_data, cfg.block_size, args.batch, device)
        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if it % args.eval_interval == 0:
            print(f"Step {it}: loss {loss.item():.6f}")
            torch.save(model.state_dict(), ckpt_path)

    torch.save(model.state_dict(), ckpt_path)
    cfg_path.write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    print("✅ Training complete. Saved:", ckpt_path, "and", cfg_path)

if __name__ == "__main__":
    main()
