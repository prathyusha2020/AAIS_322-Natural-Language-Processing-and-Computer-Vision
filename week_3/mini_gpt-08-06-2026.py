import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from seq2seq_summarizer import make_dataset
except ImportError:
    raise SystemExit("put seq2seq_summarizer.py next to this file -- "
                     "it holds the dataset generator we reuse")

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

PAD, BOS, EOS, SEP = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "=>"]


# ----------------------------------------------------------------------------
# DATA -- article and summary flattened into ONE sequence
# ----------------------------------------------------------------------------
class Vocab:
    def __init__(self, texts):
        words = sorted({w for t in texts for w in t.split()})
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)


def build_sequences(pairs, vocab):
    """<bos> article => summary <eos>  -- a language-modelling problem now.

    There is no encoder and no decoder here. There is one stream of tokens, and
    the model's only job is to predict the next one. The '=>' token is doing the
    same work that '<|assistant|>' does in a chat template.
    """
    seqs = []
    for article, summary in pairs:
        seqs.append([BOS] + vocab.encode(article) + [SEP]
                    + vocab.encode(summary) + [EOS])
    return seqs


def batches(seqs, batch_size, shuffle=True, loss_on="summary"):
    """Yield (inputs, targets).

    inputs are everything but the last token, targets everything but the first:
    predicting position t from positions 0..t-1, at EVERY t, in one pass.

    loss_on="summary" blanks the targets before the '=>' marker. The model still
    READS the article -- it just is not graded on predicting it. This matters
    here (the article's slot words are random, so predicting them is impossible
    noise that drowns out the real signal) and it is also exactly what happens
    when a chat model is fine-tuned: the loss is taken on the assistant's turn,
    not on the user's.
    """
    data = list(seqs)
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        n = max(len(x) for x in chunk)
        out = torch.full((len(chunk), n), PAD, dtype=torch.long)
        for j, x in enumerate(chunk):
            out[j, :len(x)] = torch.tensor(x)
        x_in, y = out[:, :-1], out[:, 1:].clone()
        if loss_on == "summary":
            for j, seq in enumerate(chunk):
                y[j, :seq.index(SEP)] = PAD          # PAD is the ignore_index
        yield x_in, y


# ----------------------------------------------------------------------------
# SLIDES 8-11 -- MULTI-HEAD SELF-ATTENTION
# ----------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Score, normalise, blend -- with three things the seq2seq version lacked:
    separate key and value projections, several heads, and a causal mask.
    """

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "heads must divide the model dimension"
        self.n_head = n_head
        self.d_head = d_model // n_head          # 8 heads of 64, not 1 of 512

        # three separate learned matrices (slide 8). One nn.Linear producing
        # 3*d_model is the same thing, done in a single multiply.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)  # mixes the heads back together
        self.drop = nn.Dropout(dropout)
        self.last_attn = None                    # kept only for the visualiser

    def forward(self, x, pad_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        # (B, T, C) -> (B, n_head, T, d_head): every head gets its own subspace
        def heads(t):
            return t.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)

        # --- SCORE: every query against every key, in one matrix multiply -----
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # the sqrt matters: without it the dot products grow with dimension,
        # softmax saturates towards one-hot, and the gradient dies.

        # --- MASK: nothing may look to its right (slide 11) -------------------
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~causal, float("-inf"))
        if pad_mask is not None:                 # and nothing looks at padding
            att = att.masked_fill(~pad_mask[:, None, None, :], float("-inf"))

        # --- NORMALISE --------------------------------------------------------
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()
        att = self.drop(att)

        # --- BLEND: weighted average of the VALUE vectors ---------------------
        out = att @ v                            # (B, n_head, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(out))


# ----------------------------------------------------------------------------
# SLIDE 12 -- FEED-FORWARD, RESIDUALS, LAYER NORM
# ----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Attention mixes information between positions. This thinks about it, at
    each position independently. Most of the parameters live here."""

    def __init__(self, d_model, dropout=0.1, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block. Note the shape of both lines:

        x = x + sublayer(norm(x))

    The `x +` is the residual connection -- each sub-layer ADDS to the stream
    rather than replacing it, so the original signal (and the gradient) always
    has a clean path through the whole stack. Normalising before the sub-layer
    rather than after ("pre-LN") is what makes deep stacks train without a
    careful warm-up schedule.
    """

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ----------------------------------------------------------------------------
# SLIDE 5 -- THE WHOLE STACK
# ----------------------------------------------------------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=4,
                 block_size=64, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(block_size, d_model)   # slide 7
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying: the same table
                                                 # reads words in and writes them out
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "sequence longer than the context window"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # slides 6 + 7
        pad_mask = idx != PAD
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.head(self.ln_f(x))           # (B, T, vocab) -- a prediction
                                                 # at EVERY position, in one pass

    # ---- slides 18-20: the generation loop ---------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new=16, temperature=0.0, top_p=None):
        """temperature=0 is greedy. Otherwise sample, optionally from the
        smallest set of tokens whose probability sums to top_p."""
        self.eval()
        for _ in range(max_new):
            window = idx[:, -self.block_size:]   # the context window, literally
            logits = self.forward(window)[:, -1, :]   # only the last position
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_p is not None:
                    srt, order = probs.sort(descending=True, dim=-1)
                    cutoff = srt.cumsum(dim=-1) - srt > top_p
                    srt[cutoff] = 0.0
                    srt = srt / srt.sum(dim=-1, keepdim=True)
                    pick = torch.multinomial(srt, 1)
                    nxt = order.gather(-1, pick)
                else:
                    nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if (nxt == EOS).all():
                break
        return idx


# ----------------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------------
def train(model, seqs, epochs, batch_size, lr, loss_on="summary"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\ntraining MiniGPT -- {n_params:,} parameters")
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in batches(seqs, batch_size, loss_on=loss_on):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n += 1
        print(f"epoch {epoch:>2}  loss {total / n:.4f}")


# ----------------------------------------------------------------------------
# LOOKING INSIDE -- one head, as the slides draw it
# ----------------------------------------------------------------------------
def show_head(model, tokens, vocab, layer, head, top_k=3):
    """For each generated token, print where that head looked."""
    att = model.blocks[layer].attn.last_attn      # (B, heads, T, T)
    if att is None:
        return
    a = att[0, head]
    words = [vocab.itos[t] for t in tokens]
    start = words.index("=>") if "=>" in words else 0

    print(f"\nlayer {layer}, head {head} -- where each position looked.")
    print("  NB the row for position t is what produced the token AFTER it, so a")
    print("  row that stares at 'seattle' is the row that then emits 'seattle'.")
    for t in range(start + 1, min(len(words), a.size(0))):
        row = a[t, :t + 1]
        vals, idxs = row.topk(min(top_k, t + 1))
        looked = "   ".join(f"{words[i]} {v:.2f}"
                            for v, i in zip(vals.tolist(), idxs.tolist()))
        print(f"  {words[t]:>12}  <-  {looked}")


def sharpest_head(model, tokens):
    """Most heads are diffuse and hard to describe. A few are sharp and do an
    obvious job. This finds the sharpest one so you have something to show,
    rather than hunting through layers * heads by hand."""
    words_len = len(tokens)
    best, best_score = (0, 0), -1.0
    for li, block in enumerate(model.blocks):
        att = block.attn.last_attn
        if att is None:
            continue
        for h in range(att.size(1)):
            a = att[0, h]
            rows = []
            for t in range(1, min(words_len, a.size(0))):
                row = a[t, :t + 1].clone()
                row[t] = 0.0            # ignore self-attention
                row[0] = 0.0            # and the <bos> sink
                if row.sum() > 0:
                    rows.append(row.max().item())
            if rows:
                score = sum(rows) / len(rows)
                if score > best_score:
                    best_score, best = score, (li, h)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-on", choices=["summary", "all"], default="summary")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--show-layer", type=int, default=None)
    ap.add_argument("--show-head", type=int, default=0)
    args = ap.parse_args()

    pairs = make_dataset(args.examples)
    split = int(0.9 * len(pairs))
    vocab = Vocab([a for a, _ in pairs] + [s for _, s in pairs])
    train_seqs = build_sequences(pairs[:split], vocab)
    test_pairs = pairs[split:]
    block_size = max(len(s) for s in train_seqs) + 4

    print(f"{len(train_seqs)} sequences, vocabulary {len(vocab)}, "
          f"context window {block_size} tokens")
    print(f"one training sequence:\n  {vocab.decode(train_seqs[0])}")

    model = MiniGPT(len(vocab), args.d_model, args.heads, args.layers,
                    block_size)
    train(model, train_seqs, args.epochs, args.batch_size, args.lr,
          args.loss_on)

    print("\n--- generating (prompt is the article, model continues) ---")
    for article, gold in test_pairs[:3]:
        prompt = torch.tensor([[BOS] + vocab.encode(article) + [SEP]])
        out = model.generate(prompt, max_new=10,
                             temperature=args.temperature, top_p=args.top_p)
        produced = out[0, prompt.size(1):].tolist()
        if EOS in produced:
            produced = produced[:produced.index(EOS)]
        print(f"  article   : {article}")
        print(f"  reference : {gold}")
        print(f"  generated : {vocab.decode(produced)}\n")

    layer = args.show_layer if args.show_layer is not None else args.layers - 1
    article, _ = test_pairs[0]
    prompt = torch.tensor([[BOS] + vocab.encode(article) + [SEP]])
    full = model.generate(prompt, max_new=10)
    model(full)                                   # one pass to refresh last_attn
    show_head(model, full[0].tolist(), vocab, layer, args.show_head)
    if args.show_layer is None:
        sl, sh_ = sharpest_head(model, full[0].tolist())
        if (sl, sh_) != (layer, args.show_head):
            print(f"\n(the sharpest head in this model is layer {sl}, head {sh_})")
            show_head(model, full[0].tolist(), vocab, sl, sh_)
    print("\ncompare this with the LSTM attention map from last class: same idea,\n"
          "but every position has a row, and there are heads * layers of them.")


if __name__ == "__main__":
    main()
