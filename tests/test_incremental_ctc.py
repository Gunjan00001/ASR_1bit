import numpy as np

from onebit_asr.serving.incremental_ctc import IncrementalCTCDecoder


class FakeTokenizer:
    def get_vocab(self):
        return {"<pad>": 0, "A": 1, "B": 2, "|": 4}


def onehot(ids, vocab=5):
    out = np.full((len(ids), vocab), -10.0, dtype=np.float32)
    out[np.arange(len(ids)), ids] = 10.0
    return out


def test_repeated_token_collapses_across_chunk_boundary():
    decoder = IncrementalCTCDecoder(FakeTokenizer())
    decoder.feed_logits(onehot([1, 1]))
    state = decoder.feed_logits(onehot([1, 1]))
    assert state["partial_text"] == "A"
    assert state["tokens_emitted"] == 1
    decoder.finalize()
    assert decoder.snapshot()["text"] == "A"


def test_blank_resets_repetition_across_chunks():
    decoder = IncrementalCTCDecoder(FakeTokenizer())
    decoder.feed_logits(onehot([1]))
    decoder.feed_logits(onehot([0]))  # blank is preserved across boundary
    state = decoder.feed_logits(onehot([1]))
    assert state["partial_text"] == "AA"
    decoder.finalize()
    assert decoder.snapshot()["text"] == "AA"


def test_word_delimiter_commits_immutable_word_and_flushes_tail():
    decoder = IncrementalCTCDecoder(FakeTokenizer())
    state = decoder.feed_logits(onehot([1, 4, 2]))
    assert state["final_text"] == "A"
    assert state["partial_text"] == "B"
    state = decoder.finalize()
    assert state["text"] == "A B"
    assert state["committed_words"] == 2


def test_reset_clears_previous_token_and_text():
    decoder = IncrementalCTCDecoder(FakeTokenizer())
    decoder.feed_logits(onehot([1, 4]))
    decoder.reset()
    state = decoder.feed_logits(onehot([2]))
    assert state["final_text"] == ""
    assert state["partial_text"] == "B"
