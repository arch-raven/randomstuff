import random
from pathlib import Path

THRESH = 100

def flip_coin(): 
    return random.random() > 0.5

def flip_coin_ntimes(n : int) -> bool:
    """Return True if n consecutive heads are observed else False"""  
    for _ in range(n):
        if not flip_coin():
            return False
    return True

def cvm(tokens : list[str]):
    round = 0
    vocab = set()
    for tok in tokens:
        if len(vocab) >= THRESH:
            vocab = set(v for v in vocab if flip_coin())
            round += 1        

        vocab.discard(tok)
        if flip_coin_ntimes(round):
            vocab.add(tok)
    
    print(f"{round=} {len(vocab)=}")
    print(f"Result: {len(vocab) * (2 ** round)}")

def preprocess_text(text):
    text = text.lower().split()
    tokens = [word.strip(".,?!") for word in text]
    utokens = set(tokens)
    print(f"# unique words (true) : {len(utokens)}")
    print(f"# tokens : {len(tokens)}")
    return tokens


if __name__ == "__main__":
    text = Path("hamlet.txt").read_text()
    tokens = preprocess_text(text)
    cvm(tokens)

