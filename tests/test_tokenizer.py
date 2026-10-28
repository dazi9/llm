import pytest

from tokenizer import Tokenizer, train_bpe


BASE_TEXT = (
    "Once upon a time there was a little girl named Lily. She liked to run. "
    "Lily ran and ran! Then she saw a dog. The dog was big and brown. "
    "One day, Lily found a red ball under the tree. She was very happy. "
    "The cat dog bird fish lion tiger bear wolf ran fast slow jump walk sleep wake eat drink play rest sit stand up down left right over under near far. "
    "Anna Ben Carl Dana Eva Finn Gabe Hope Ivan Jade Kate Liam Mia Noah Olga Paul Quinn Rosa Sam Tara Umar Vera Will Xena Yale Zara "
)


@pytest.fixture(scope="module")
def tiny_tok():
    text = (BASE_TEXT + "<|endoftext|>" + BASE_TEXT) * 20
    vocab, merges = train_bpe(text, 350, ["<|endoftext|>"])
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])


def test_vocab_size(tiny_tok):
    assert tiny_tok.vocab_size == 350
    assert len(tiny_tok.merges) == 350 - 256 - 1


def test_roundtrip(tiny_tok):
    s = "Once upon a time, there was a little girl named Lily! She ran 42 miles."
    assert tiny_tok.decode(tiny_tok.encode(s)) == s


def test_special_token(tiny_tok):
    s = "Hello world<|endoftext|>next story"
    ids = tiny_tok.encode(s)
    assert tiny_tok._special_to_id["<|endoftext|>"] in ids
    assert tiny_tok.decode(ids) == s


def test_all_ids_in_range(tiny_tok):
    s = "The quick brown fox jumps over the lazy dog. 12345 !!! "
    assert all(0 <= i < tiny_tok.vocab_size for i in tiny_tok.encode(s))


def test_compression(tiny_tok):
    # in-distribution text should compress well (well above 1 char/token);
    # (out-of-distribution text correctly compresses poorly — BPE can only
    # merge pairs it saw during training)
    s = (BASE_TEXT * 30)
    ids = tiny_tok.encode(s)
    # 93 merges + many one-off name words dilutes the ratio; 1.8 has margin
    assert len(s) / len(ids) > 1.8


def test_save_load(tmp_path, tiny_tok):
    p = tmp_path / "tok.json"
    tiny_tok.save(str(p))
    tok2 = Tokenizer.load(str(p))
    assert tok2.vocab_size == tiny_tok.vocab_size
    s = "Lily and the big brown dog ran to the park."
    assert tok2.decode(tok2.encode(s)) == s
    assert tok2.encode(s) == tiny_tok.encode(s)
