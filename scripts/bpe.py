import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


def clean_text_for_bpe(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove redundant whitespace
    text = re.sub(r"[^\w\s]", "", text)  # Remove or replace punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.strip()
    return text


def get_pair_freq(vocab: dict) -> defaultdict:
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        tokens = word.split()
        for i in range(len(tokens) - 1):
            pairs[tokens[i], tokens[i + 1]] += freq
    return pairs


def merge_byte_pairs(vocab: dict[str, int], pair: tuple[str, str]):
    old_byte = " ".join(pair)
    new_byte = "".join(pair)
    out_vocab = {}
    for word, freq in vocab.items():
        word = word.replace(old_byte, new_byte)
        out_vocab[word] = freq
    return out_vocab


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filepath", type=Path, default="files/hamlet.txt")
    parser.add_argument("--max_num_merges", type=int, default=1000)
    args = parser.parse_args()

    corpus = clean_text_for_bpe(args.filepath.read_text())

    vocab = defaultdict(int)
    for word in corpus.split():
        word = " ".join(word + "_")
        vocab[word] += 1

    for i in range(args.max_num_merges):
        pairs = get_pair_freq(vocab)
        if not pairs:
            print(f"No more merges are possible, Stopping at {i} iterations!")
            break
        else:
            most_freq_pair, _ = max(pairs.items(), key=lambda x: x[1])
            vocab = merge_byte_pairs(vocab, most_freq_pair)
            print(most_freq_pair)

    # vocab = sorted(vocab, key=vocab.get, reverse=True)

# python -i scripts/bpe.py --max_num_merges 5000
# tail of Output
## ('car', 'nal_')
## ('acci', 'dental_')
## ('casu', 'al_')
## ('up', 'shot_')
## ('in', 'ventors_')
## ('in', 'vite_')
## ('al', 'so_')
## ('er', 'rors_')
## ('cap', 'tains_')
## ('roy', 'ally_')
## ('lou', 'dly_')
## ('fi', 'eld_')
## ('shoo', 't_')
## ('mar', 'ching_')
## No more merges are possible, Stopping at 4403 iterations!
