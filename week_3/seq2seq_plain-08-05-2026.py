"""
PART 1: Sequence-to-sequence summarisation WITHOUT attention.

This is the vanilla encoder-decoder from slides 4-7. The encoder reads the
whole article and squeezes everything it learned into ONE vector (the
"context vector"). The decoder must write the entire summary from that
single vector -- it can never look back at the article again.

Run it and study the sample summaries at the end. The model learns the
FORMAT perfectly ("<event> <verb> <number> <things> in <city>") but keeps
getting the FACTS wrong -- wrong city, wrong number. That is the
bottleneck: one fixed-size vector cannot hold every detail of the input.

    python seq2seq_plain.py

Part 2 (seq2seq_summarizer.py) fixes this with attention.
Only dependency is torch.
"""

import argparse
import random

import torch
import torch.nn as nn

SEED = 13
random.seed(SEED)
torch.manual_seed(SEED)

PAD, SOS, EOS, UNK = 0, 1, 2, 3


# ----------------------------------------------------------------------------
# 1. DATA -- template-generated article/headline pairs (same as part 2)
# ----------------------------------------------------------------------------
CITIES = ["nashville", "memphis", "denver", "seattle", "boston", "austin",
          "phoenix", "portland", "atlanta", "chicago"]
DAYS = ["monday", "tuesday", "friday", "sunday", "saturday"]
NUMBERS = ["two", "three", "five", "seven", "twelve", "twenty"]
THINGS = ["schools", "flights", "roads", "bridges", "shops", "clinics"]
EVENTS = [("flooding", "closed"), ("snow", "cancelled"), ("wind", "damaged"),
          ("heat", "shut"), ("storms", "delayed")]

ARTICLE_TEMPLATES = [
    "officials said that {n} {thing} in {city} were {verb} on {day} "
    "after severe {event} moved through the region overnight",
    "residents of {city} woke on {day} to severe {event} which left "
    "{n} {thing} {verb} across the city according to local officials",
    "a spokesman confirmed on {day} that severe {event} had {verb} "
    "{n} {thing} in {city} and warned that more was expected",
]
SUMMARY_TEMPLATE = "{event} {verb} {n} {thing} in {city}"


def make_dataset(n_examples=3000):
    """Return a list of (article, summary) string pairs."""
    pairs = []
    for _ in range(n_examples):
        event, verb = random.choice(EVENTS)
        slots = dict(n=random.choice(NUMBERS), thing=random.choice(THINGS),
                     city=random.choice(CITIES), day=random.choice(DAYS),
                     event=event, verb=verb)
        article = random.choice(ARTICLE_TEMPLATES).format(**slots)
        summary = SUMMARY_TEMPLATE.format(**slots)
        pairs.append((article, summary))
    return pairs


class Vocab:
    """Words to integers. Slide: 'words into numbers'."""

    def __init__(self, texts, min_count=1):
        counts = {}
        for t in texts:
            for w in t.split():
                counts[w] = counts.get(w, 0) + 1
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.itos += sorted(w for w, c in counts.items() if c >= min_count)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, add_eos=True):
        ids = [self.stoi.get(w, UNK) for w in text.split()]
        return ids + [EOS] if add_eos else ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            out.append(self.itos[i])
        return " ".join(out)


def make_batches(pairs, vocab, batch_size, shuffle=True):
    """Pad to the longest sequence in each batch and yield tensors."""
    data = [(vocab.encode(a), [SOS] + vocab.encode(s)) for a, s in pairs]
    if shuffle:                 # shuffle for training; NEVER when evaluating,
        random.shuffle(data)    # or predictions stop lining up with articles
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        src_len = max(len(x) for x, _ in chunk)
        tgt_len = max(len(y) for _, y in chunk)
        src = torch.full((len(chunk), src_len), PAD, dtype=torch.long)
        tgt = torch.full((len(chunk), tgt_len), PAD, dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            src[j, :len(x)] = torch.tensor(x)
            tgt[j, :len(y)] = torch.tensor(y)
        yield src, tgt


# ----------------------------------------------------------------------------
# 2. ENCODER  (slide 4, left half)
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    """Reads the article one word at a time. Its state after the LAST word
    is the context vector -- the ONLY thing the decoder will ever see."""

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        emb = self.embedding(src)                       # (B, S, E)
        _, hidden = self.rnn(emb)                       # (2, B, H)
        hidden = torch.tanh(self.bridge(
            torch.cat([hidden[0], hidden[1]], dim=1)))  # (B, H)
        return hidden.unsqueeze(0)                      # (1, B, H)


# ----------------------------------------------------------------------------
# 3. DECODER  (slides 5-7: the bottleneck)
# ----------------------------------------------------------------------------
class Decoder(nn.Module):
    """Starts from the context vector and writes one word per step.
    Everything it knows about the article is that single starting vector.
    Nothing it produces can ever look back at the source again."""

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, vocab_size)

    def step(self, y_prev, hidden):
        emb = self.embedding(y_prev).unsqueeze(1)       # (B, 1, E)
        output, hidden = self.rnn(emb, hidden)
        return self.out(output.squeeze(1)), hidden


# ----------------------------------------------------------------------------
# 4. THE FULL MODEL
# ----------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing=0.5):
        """Teacher forcing: during training we usually feed the decoder the
        CORRECT previous word rather than its own guess, so one early mistake
        does not poison the whole sequence."""
        hidden = self.encoder(src)
        logits = []
        y_prev = tgt[:, 0]                              # <sos>
        for t in range(1, tgt.size(1)):
            step_logits, hidden = self.decoder.step(y_prev, hidden)
            logits.append(step_logits)
            use_gold = random.random() < teacher_forcing
            y_prev = tgt[:, t] if use_gold else step_logits.argmax(1)
        return torch.stack(logits, dim=1)               # (B, T-1, V)

    @torch.no_grad()
    def summarise(self, src, max_len=12):
        """Greedy decoding, one word at a time."""
        self.eval()
        hidden = self.encoder(src)
        y_prev = torch.full((src.size(0),), SOS, dtype=torch.long)
        tokens = []
        for _ in range(max_len):
            logits, hidden = self.decoder.step(y_prev, hidden)
            y_prev = logits.argmax(1)
            tokens.append(y_prev)
            if (y_prev == EOS).all():
                break
        return torch.stack(tokens, dim=1)               # (B, T)


# ----------------------------------------------------------------------------
# 5. TRAINING
# ----------------------------------------------------------------------------
def train(model, pairs, vocab, epochs, batch_size, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)     # never train on padding
    print(f"\n--- training seq2seq WITHOUT attention "
          f"({sum(p.numel() for p in model.parameters()):,} parameters) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for src, tgt in make_batches(pairs, vocab, batch_size):
            logits = model(src, tgt)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # RNNs explode
            opt.step()
            total += loss.item()
            n_batches += 1
        print(f"epoch {epoch:>2}  loss {total / n_batches:.4f}")


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--examples", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hid-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    pairs = make_dataset(args.examples)
    split = int(0.9 * len(pairs))
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    vocab = Vocab([a for a, _ in pairs] + [s for _, s in pairs])

    print(f"{len(train_pairs)} training pairs, vocabulary of {len(vocab)} words")
    print(f"\nexample article : {train_pairs[0][0]}")
    print(f"example summary : {train_pairs[0][1]}")

    model = Seq2Seq(
        Encoder(len(vocab), args.emb_dim, args.hid_dim),
        Decoder(len(vocab), args.emb_dim, args.hid_dim),
    )
    train(model, train_pairs, vocab, args.epochs, args.batch_size, args.lr)

    print("\nsample summaries -- watch the FACTS, not the format:")
    sample = test_pairs[:5]
    src, _ = next(make_batches(sample, vocab, len(sample), shuffle=False))
    tokens = model.summarise(src)
    for i, (article, gold) in enumerate(sample):
        pred = vocab.decode(tokens[i].tolist())
        marker = "OK " if pred == gold else "WRONG"
        print(f"  article   : {article}")
        print(f"  reference : {gold}")
        print(f"  predicted : {pred}   [{marker}]\n")

    print("The model nails the TEMPLATE but scrambles the DETAILS (city,")
    print("number, event). Everything it knows about the article had to fit")
    print("through one fixed-size vector, and the details did not survive.")
    print("Part 2 -- seq2seq_summarizer.py -- fixes this with attention.")


if __name__ == "__main__":
    main()
