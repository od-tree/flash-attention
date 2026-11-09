"""Decode-step inputs taken from the Qwen3.5 head-TP eval plan.

One case is one operator call, not the 200-request trace. Sequence lengths are
the plan's short input 1024, the last short decode step 1279, and the long
inputs 8192 and 32768. Head shape is Qwen3.5 Full Attention: 16 Q heads, 2 KV
heads, head dim 256, d = 2. A ``_local`` case keeps the same lengths and marks
every row local.
"""

from dataclasses import dataclass


NUM_Q = 16
NUM_KV = 2
HEAD_DIM = 256
DEGREE = 2
SHORT_LEN = 1024
LAST_SHORT_LEN = 1024 + 256 - 1
LONG_LENS = (8192, 32768)


@dataclass(frozen=True)
class DecodeCase:
    """One decode step fed to the operator."""

    name: str
    num_tokens: int
    long_count: int
    short_len: int
    long_len: int | None
    coop: bool
    num_q: int = NUM_Q
    num_kv: int = NUM_KV
    head_dim: int = HEAD_DIM
    degree: int = DEGREE

    def seq_lens(self) -> list[int]:
        if self.long_count == 0 or self.long_len is None:
            return [self.short_len] * self.num_tokens
        seq = [self.short_len] * self.num_tokens
        for token in range(self.long_count):
            seq[token] = self.long_len
        return seq

    def owners(self) -> list[int]:
        """Owner rank of a cooperative row, or -1 when the row stays local."""
        owners = []
        for token in range(self.num_tokens):
            if self.coop and token < self.long_count:
                owners.append(0)
            else:
                owners.append(-1)
        return owners


def _pair(name, num_tokens, long_count, short_len, long_len):
    coop = DecodeCase(name, num_tokens, long_count, short_len, long_len, True)
    local = DecodeCase(
        name + "_local", num_tokens, long_count, short_len, long_len, False
    )
    return coop, local


def eval_cases() -> list[DecodeCase]:
    """Capacity checks, pure loads, and the 1% / 5% / 20% mixes."""
    cases = [
        DecodeCase("capacity_1024", 1, 0, 1024, None, False),
        DecodeCase("capacity_8192", 1, 0, 8192, None, False),
        DecodeCase("capacity_32768", 1, 0, 32768, None, False),
        DecodeCase("pure_short_1024", 20, 0, SHORT_LEN, None, False),
        DecodeCase("pure_short_1279", 20, 0, LAST_SHORT_LEN, None, False),
    ]
    for long_len in LONG_LENS:
        cases.extend(_pair(f"pure_long_{long_len}", 20, 20, SHORT_LEN, long_len))
        cases.extend(_pair(f"mix_1pct_{long_len}", 100, 1, SHORT_LEN, long_len))
        cases.extend(_pair(f"mix_5pct_{long_len}", 20, 1, SHORT_LEN, long_len))
        cases.extend(_pair(f"mix_20pct_{long_len}", 20, 4, SHORT_LEN, long_len))
    return cases


def operator_input_shapes(case: DecodeCase) -> dict[str, tuple[int, ...]]:
    """Tensor shapes for one rank at this step.

    ``k_cache`` is sized to the longest row. The exchange kernels copy only the
    current token; the cache length is the attention mainloop's K length.
    Payload buffers are the per-rank receive layout ``[src, token, heads, dim]``.
    """
    seq = case.seq_lens()
    tokens = case.num_tokens
    dp = case.degree
    q_per = case.num_q // case.degree
    kv_per = case.num_kv // case.degree
    max_seq = max(seq)
    heads = case.head_dim
    return {
        "q": (tokens, case.num_q, heads),
        "k": (tokens, case.num_kv, heads),
        "v": (tokens, case.num_kv, heads),
        "k_cache": (tokens, max_seq, case.num_kv, heads),
        "v_cache": (tokens, max_seq, case.num_kv, heads),
        "seq_len": (tokens,),
        "row_kind": (tokens,),
        "slot_rank": (tokens, dp),
        "q_range": (tokens, dp, 2),
        "kv_range": (tokens, dp, 2),
        "payload_q": (dp, tokens, q_per, heads),
        "payload_k": (dp, tokens, kv_per, heads),
        "payload_v": (dp, tokens, kv_per, heads),
        "payload_out": (dp, tokens, q_per, heads),
        "arrived": (dp, tokens),
    }
