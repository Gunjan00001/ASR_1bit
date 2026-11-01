import itertools
import math

import torch

from onebit_asr.decoding.decode import ctc_collapse, greedy_decode, prefix_beam_search


def test_ctc_collapse_removes_repeats_and_blanks():
    assert ctc_collapse([0, 1, 1, 0, 2, 0, 2, 3], blank_id=0) == [1, 2, 2, 3]


def test_blank_penalty_changes_greedy_decision():
    logits = torch.tensor([[3.0, 1.0], [3.0, 1.0]])
    assert greedy_decode(logits, blank_id=0) == []
    assert greedy_decode(logits, blank_id=0, blank_penalty=3.0) == [1]


def test_beam_search_matches_exhaustive_path_marginals():
    torch.manual_seed(0)
    frames, vocab = 4, 3
    logits = torch.randn(frames, vocab)
    logp = torch.log_softmax(logits, dim=-1)
    scores = {}
    for path in itertools.product(range(vocab), repeat=frames):
        output = tuple(ctc_collapse(list(path), blank_id=0))
        score = float(logp[torch.arange(frames), torch.tensor(path)].sum())
        scores[output] = float(
            torch.logaddexp(
                torch.tensor(scores.get(output, -math.inf)), torch.tensor(score)
            )
        )
    expected = max(scores.items(), key=lambda item: item[1])
    score, tokens = prefix_beam_search(logits, beam_width=100, nbest=1)[0]
    assert tuple(tokens) == expected[0]
    assert abs(score - expected[1]) < 1e-5


def test_beam_search_keeps_repeated_label_only_across_blank():
    logits = torch.tensor(
        [
            [0.0, 5.0],
            [0.0, 5.0],
            [5.0, 0.0],
            [0.0, 5.0],
        ]
    )
    score, tokens = prefix_beam_search(logits, beam_width=20, nbest=1)[0]
    assert tokens == [1, 1]
    assert math.isfinite(score)
    assert greedy_decode(logits) == [1, 1]


def test_beam_search_nbest_is_sorted():
    torch.manual_seed(1)
    logits = torch.randn(5, 4)
    best = prefix_beam_search(logits, beam_width=20, nbest=3)
    assert len(best) == 3
    assert best[0][0] >= best[1][0] >= best[2][0]
    assert len({tuple(tokens) for _, tokens in best}) == 3
