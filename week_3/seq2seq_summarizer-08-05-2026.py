
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 13
random.seed(SEED)
torch.manual_seed(SEED)

PAD, SOS, EOS, UNK = 0, 1, 2, 3


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
# Real summarisation corpora (CNN/DailyMail) need hours on a GPU. For a lecture
# we want a task that is genuinely seq2seq -- reordering, length change, copying
# from the source -- but small enough that a tiny model succeeds while people
# watch. So we generate article/headline pairs from templates.
#
# To swap in real data, replace make_dataset() with anything that returns a list
# of (article_string, summary_string) pairs. Nothing else changes.

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
            if i > UNK or i == UNK:
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
    """Reads the article one word at a time and returns

        outputs -- its state AFTER every word   (B, S, H)   <- attention uses this
        hidden  -- its state after the LAST word (1, B, H)  <- the context vector
    """

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        emb = self.embedding(src)                       # (B, S, E)
        outputs, hidden = self.rnn(emb)                 # (B, S, 2H), (2, B, H)
        outputs = torch.tanh(self.bridge(outputs))      # (B, S, H)
        hidden = torch.tanh(self.bridge(
            torch.cat([hidden[0], hidden[1]], dim=1)))  # (B, H)
        return outputs, hidden.unsqueeze(0)             # (1, B, H)


# ----------------------------------------------------------------------------
# 3a. DECODER WITHOUT ATTENTION  (slides 5-7: the bottleneck)
# ----------------------------------------------------------------------------
class PlainDecoder(nn.Module):
    """Everything it knows about the article is the single vector it started
    with. Nothing it produces can ever look back at the source again."""

    uses_attention = False

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, vocab_size)

    def step(self, y_prev, hidden, enc_outputs, mask):
        emb = self.embedding(y_prev).unsqueeze(1)       # (B, 1, E)
        output, hidden = self.rnn(emb, hidden)
        return self.out(output.squeeze(1)), hidden, None


# ----------------------------------------------------------------------------
# 3b. ATTENTION + DECODER  (slides 8-10)
# ----------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    """The three steps from slide 10.

        score     e_i = v^T tanh(W_dec s + W_enc h_i)   'how relevant is word i?'
        normalise a   = softmax(e)                      'turn scores into shares'
        blend     c   = sum_i a_i h_i                   'weighted average'

    `a` is exactly the row of numbers plotted on slide 9.
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.W_dec = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W_enc = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask):
        # dec_hidden (B, H) -> (B, 1, H) so it broadcasts over source positions
        scores = self.v(torch.tanh(
            self.W_dec(dec_hidden).unsqueeze(1) + self.W_enc(enc_outputs)
        )).squeeze(2)                                   # (B, S)

        # padding must not receive attention: -inf becomes 0 after softmax
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)              # (B, S), rows sum to 1
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class AttentionDecoder(nn.Module):
    """Same GRU as above, but each step is fed a context vector computed fresh
    for the word it is about to write."""

    uses_attention = True

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.attention = BahdanauAttention(hid_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim * 2, vocab_size)

    def step(self, y_prev, hidden, enc_outputs, mask):
        emb = self.embedding(y_prev)                             # (B, E)
        context, weights = self.attention(hidden[0], enc_outputs, mask)
        rnn_in = torch.cat([emb, context], dim=1).unsqueeze(1)   # (B, 1, E+H)
        output, hidden = self.rnn(rnn_in, hidden)
        logits = self.out(torch.cat([output.squeeze(1), context], dim=1))
        return logits, hidden, weights


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
        enc_outputs, hidden = self.encoder(src)
        mask = src != PAD
        logits = []
        y_prev = tgt[:, 0]                              # <sos>
        for t in range(1, tgt.size(1)):
            step_logits, hidden, _ = self.decoder.step(
                y_prev, hidden, enc_outputs, mask)
            logits.append(step_logits)
            use_gold = random.random() < teacher_forcing
            y_prev = tgt[:, t] if use_gold else step_logits.argmax(1)
        return torch.stack(logits, dim=1)               # (B, T-1, V)

    @torch.no_grad()
    def summarise(self, src, max_len=12):
        """Greedy decoding, one word at a time -- the loop from slide 17."""
        self.eval()
        enc_outputs, hidden = self.encoder(src)
        mask = src != PAD
        y_prev = torch.full((src.size(0),), SOS, dtype=torch.long)
        tokens, attn_rows = [], []
        for _ in range(max_len):
            logits, hidden, weights = self.decoder.step(
                y_prev, hidden, enc_outputs, mask)
            y_prev = logits.argmax(1)
            tokens.append(y_prev)
            if weights is not None:
                attn_rows.append(weights)
            if (y_prev == EOS).all():
                break
        tokens = torch.stack(tokens, dim=1)             # (B, T)
        attn = torch.stack(attn_rows, dim=1) if attn_rows else None
        return tokens, attn


# ----------------------------------------------------------------------------
# 5. TRAINING
# ----------------------------------------------------------------------------
def train(model, pairs, vocab, epochs, batch_size, lr, label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)     # never train on padding
    print(f"\n--- training {label} "
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
    return total / n_batches


# ----------------------------------------------------------------------------
# 6. SHOWING THE ATTENTION MATRIX  (slide 9, in the terminal)
# ----------------------------------------------------------------------------
def print_attention(article, summary_tokens, weights, vocab, top_k=8):
    """weights: (T, S). Prints the heat map as text, keeping only the source
    words that actually received attention so it fits on screen."""
    src_words = article.split()
    weights = weights[:len(summary_tokens), :len(src_words)]
    keep = weights.max(dim=0).values.topk(min(top_k, len(src_words))).indices
    keep = sorted(keep.tolist())

    header = " " * 12 + "".join(f"{src_words[j][:9]:>10}" for j in keep)
    print(header)
    for i, word in enumerate(summary_tokens):
        row = "".join(f"{weights[i, j].item():>10.2f}" for j in keep)
        print(f"{word[:11]:>12}{row}")
    print("  (each row sums to 1 across the FULL article, not just these columns)")


# ----------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--examples", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hid-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--only", choices=["plain", "attn"], default=None)
    args = ap.parse_args()

    pairs = make_dataset(args.examples)
    split = int(0.9 * len(pairs))
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    vocab = Vocab([a for a, _ in pairs] + [s for _, s in pairs])

    print(f"{len(train_pairs)} training pairs, vocabulary of {len(vocab)} words")
    print(f"\nexample article : {train_pairs[0][0]}")
    print(f"example summary : {train_pairs[0][1]}")

    variants = []
    if args.only in (None, "plain"):
        variants.append(("seq2seq WITHOUT attention", PlainDecoder))
    if args.only in (None, "attn"):
        variants.append(("seq2seq WITH attention", AttentionDecoder))

    results = {}
    for label, decoder_cls in variants:
        model = Seq2Seq(
            Encoder(len(vocab), args.emb_dim, args.hid_dim),
            decoder_cls(len(vocab), args.emb_dim, args.hid_dim),
        )
        final_loss = train(model, train_pairs, vocab,
                           args.epochs, args.batch_size, args.lr, label)
        results[label] = (model, final_loss)

        print(f"\nsample summaries from {label}:")
        sample = test_pairs[:3]
        src, _ = next(make_batches(sample, vocab, len(sample), shuffle=False))
        tokens, attn = model.summarise(src)
        for i, (article, gold) in enumerate(sample):
            print(f"  article   : {article}")
            print(f"  reference : {gold}")
            print(f"  predicted : {vocab.decode(tokens[i].tolist())}\n")

        if attn is not None:
            article, _ = sample[0]
            words = vocab.decode(tokens[0].tolist()).split()
            print("attention matrix for the first example (slide 9):\n")
            print_attention(article, words, attn[0], vocab)

    if len(results) == 2:
        print("\n" + "=" * 64)
        for label, (_, loss) in results.items():
            print(f"{label:<32} final loss {loss:.4f}")
        print("Same encoder, same data, same number of epochs. The gap is the\n"
              "cost of squeezing the article into one vector.")


if __name__ == "__main__":
    main()
