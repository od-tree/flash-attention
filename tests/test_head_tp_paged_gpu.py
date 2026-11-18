"""Paged head-TP decode on two peer GPUs.

Short rows use a normal page: every KV head, ``block_size`` tokens. Long rows
pack this rank's KV heads into the same physical page, ``d`` logical blocks
each. A short row is never padded out to the longest request.

The five pure-long shapes are the ones in ``operator_refactor.md``. The mixed
shapes are a short batch plus a few 128k or 256k requests. Those do not fit
if every row is padded to the max length; paged storage does.

``d = 4`` communication is not run. Two GPUs can only time ``d = 2``.

Peer buffers default to ``EnablePeerAccess`` + foreign ``data_ptr``. On
sm_120 / RTX 5090 (or ``P2P_BACKEND=memcpy``) the kernel writes local
payloads and the host moves them with ``cudaMemcpyPeer``.
"""

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch

# Allow ``import cuda_ipc_p2p`` when this file is launched by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from head_tp_decode import (
    BLOCK, DIM, DTYPE, KV, Q_PER, PARTNER, HeadTPDecode as _Case,
    _enable_peers, _move, _ops,
)

WARMUP = 3
ITERS = 8

def _page_tests():
    for name in ("test_head_tp_page_d2", "test_head_tp_page_d4"):
        path = Path(__file__).resolve().parent / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in sorted(dir(module)):
            if attr.startswith("test_"):
                getattr(module, attr)()
                print(f"page {name}.{attr} ok", flush=True)


def _bf16_ulps(got, ref):
    gi = got.view(torch.int16).to(torch.int32) & 0xFFFF
    ri = ref.view(torch.int16).to(torch.int32) & 0xFFFF

    def ordered(bits):
        return torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits)

    return (ordered(gi) - ordered(ri)).abs()


# Split-KV vs a single walk can differ by about one bf16 step once a sequence
# spans more than one KV tile. Ordered-code ulps inflate that near zero; the
# absolute gap stays around 2e-3. Allow one bf16 unit at |x|~1 with slack.
_MAX_ABS_SPLIT = 4e-3


def _max_abs(got, ref):
    return float((got.float() - ref.float()).abs().max())


def _check(case):
    case.base()
    case.s0.synchronize()
    ref = case.base_out
    case.pack()
    case.cooperative()
    n = case.n_long
    long_abs = _max_abs(case.owner_out[:n], ref[:n])
    if not long_abs <= _MAX_ABS_SPLIT:
        raise AssertionError(f"spliced long rows differ by {long_abs:.3e} abs")
    if case.n > n and not torch.equal(case.owner_out[n:], ref[n:]):
        local_abs = _max_abs(case.owner_out[n:], ref[n:])
        raise AssertionError(f"a local row changed ({local_abs:.3e} abs)")
    partner = case.partner_out[:n]
    if not torch.equal(partner[:, :Q_PER], torch.zeros_like(partner[:, :Q_PER])):
        raise AssertionError("partner wrote the owner's Q heads")
    remote_abs = _max_abs(partner[:, Q_PER:], _move(ref[:n, Q_PER:], PARTNER))
    if not remote_abs <= _MAX_ABS_SPLIT:
        raise AssertionError(f"partner Q heads differ by {remote_abs:.3e} abs")


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _time(launch, streams, before=None):
    for _ in range(WARMUP):
        if before is not None:
            before()
        launch(None, None)
    for stream in streams:
        stream.synchronize()
    columns = [[] for _ in streams]
    walls = []
    for _ in range(ITERS):
        if before is not None:
            before()
        for stream in streams:
            stream.synchronize()
        starts = []
        ends = []
        for stream in streams:
            with torch.cuda.device(stream.device):
                starts.append(torch.cuda.Event(True))
                ends.append(torch.cuda.Event(True))
        t0 = time.perf_counter()
        launch(starts, ends)
        for stream in streams:
            stream.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        for index, stream in enumerate(streams):
            del stream
            columns[index].append(starts[index].elapsed_time(ends[index]))
    return [_median(column) for column in columns], _median(walls)


def _splits_heuristic(batch_nheads_mblocks, num_sms, num_n_blocks, max_splits=128):
    """Same rule as flash_api.cpp num_splits_heuristic."""
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    def ceildiv(a, b):
        return (a + b - 1) // b
    def eligible(num_splits):
        return num_splits == 1 or ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1)
    efficiency = []
    max_eff = 0.0
    for num_splits in range(1, max_splits + 1):
        if not eligible(num_splits):
            efficiency.append(0.0)
            continue
        waves = (batch_nheads_mblocks * num_splits) / num_sms
        eff = waves / math.ceil(waves)
        max_eff = max(max_eff, eff)
        efficiency.append(eff)
    for num_splits in range(1, max_splits + 1):
        if not eligible(num_splits):
            continue
        if efficiency[num_splits - 1] >= 0.85 * max_eff:
            return num_splits
    return 1


def _predict_splits(batch, seqlen, num_sms):
    num_n_blocks = (seqlen + 64 - 1) // 64
    # After the decode swap, seqlen_q is the group size 8 and one m-block remains.
    return _splits_heuristic(batch * KV, num_sms * 2, num_n_blocks)


def _sweep_case(ops, group, name, seqs, n_long, base_splits, num_sms):
    print(f"run {group} {name}", flush=True)
    try:
        case = _Case(ops, seqs, n_long, fill="empty")
        case.base_splits = base_splits
        base_ev, base_wall = _time(lambda starts, ends: case.base(starts, ends), [case.s0])
        full_bytes = case.paged_bytes
        case.pack_random()
        torch.cuda.synchronize()
        std_ev, std_wall = _time(
            lambda starts, ends: case._std_timed(starts, ends),
            [case.s0, case.s1],
            before=case.clear_flags,
        )
        splits = 1 if base_splits == 1 else _predict_splits(case.n, case.max_k, num_sms)
        base_cta = splits * case.n * KV
        std_cta = n_long + (case.n - n_long) * KV
        base_gb = full_bytes / (base_ev[0] * 1e-3) / 1e9
        std_gb = case.owner_bytes / (std_ev[0] * 1e-3) / 1e9
        print(
            f"{group:<3} {name:<22} {base_ev[0]:8.3f} {std_ev[0]:8.3f} {std_ev[1]:8.3f} "
            f"{std_wall / base_wall:7.3f} {base_gb:8.1f} {std_gb:8.1f} {base_cta:6d} {std_cta:6d} "
            f"{base_gb / base_cta:7.2f} {std_gb / std_cta:7.2f} {splits:4d} {full_bytes / 2**30:7.2f}",
            flush=True,
        )
        del case
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{group:<3} {name:<22} out of memory", flush=True)
    except Exception as exc:
        print(f"{group:<3} {name:<22} failed: {type(exc).__name__}: {exc}", flush=True)
        raise


def _arm(case, mode):
    """Zero flags, then optionally publish them before the launch.

    ``send`` makes the partner's wait succeed immediately. ``both`` also makes
    the owner's splice succeed immediately. The timed region starts afterwards.
    """
    case.clear_flags()
    if mode == "std":
        return
    with torch.cuda.stream(case.s1):
        case.arrived[1].fill_(1)
    if mode == "both":
        with torch.cuda.stream(case.s0):
            case.arrived[0].fill_(1)
    case.s0.synchronize()
    case.s1.synchronize()


def _comm_case(ops, n, seq):
    name = f"n{n}_s{seq}"
    print(f"run {name}", flush=True)
    try:
        case = _Case(ops, [seq] * n, n, fill="empty")
        base_ev, base_wall = _time(lambda starts, ends: case.base(starts, ends), [case.s0])
        full_bytes = case.paged_bytes
        case.pack_random()
        torch.cuda.synchronize()
        timed = {}
        for mode in ("std", "send", "both"):
            timed[mode] = _time(
                lambda starts, ends: case._std_timed(starts, ends),
                [case.s0, case.s1],
                before=lambda mode=mode: _arm(case, mode),
            )
        owner_bytes = case.owner_bytes
        def gb(ev):
            return owner_bytes / (ev[0] * 1e-3) / 1e9
        base_gb = full_bytes / (base_ev[0] * 1e-3) / 1e9
        std_ev, std_wall = timed["std"]
        send_ev, send_wall = timed["send"]
        both_ev, both_wall = timed["both"]
        print(
            f"{name:<14} {base_ev[0]:8.3f} {std_ev[0]:8.3f} {std_ev[1]:8.3f} {std_wall / base_wall:7.3f} "
            f"{send_ev[0]:8.3f} {send_ev[1]:8.3f} {send_wall / base_wall:7.3f} "
            f"{both_ev[0]:8.3f} {both_ev[1]:8.3f} {both_wall / base_wall:7.3f} "
            f"{base_gb:8.1f} {gb(both_ev):8.1f} {std_ev[0] - both_ev[0]:8.3f}",
            flush=True,
        )
        del case
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{name:<14} out of memory", flush=True)
    except Exception as exc:
        print(f"{name:<14} failed: {type(exc).__name__}: {exc}", flush=True)
        raise


def _comm():
    _enable_peers()
    ops = _ops()
    print(
        "comm  send=partner flag already set, both=send and return flags already set. "
        "gap is std0-both0. bothGB is owner bytes over the no-wait kernel.",
        flush=True,
    )
    print(
        f"{'case':<14} {'base0':>8} {'std0':>8} {'std1':>8} {'w_std':>7} "
        f"{'send0':>8} {'send1':>8} {'w_send':>7} {'both0':>8} {'both1':>8} {'w_both':>7} "
        f"{'baseGB':>8} {'bothGB':>8} {'gap':>8}",
        flush=True,
    )
    for n, seq in (
        (4, 262144),
        (8, 262144),
        (16, 262144),
        (32, 32768),
        (48, 32768),
        (64, 32768),
        (80, 32768),
        (108, 32768),
        (128, 32768),
        (256, 32768),
    ):
        _comm_case(ops, n, seq)
    print("comm done", flush=True)


def _sweep():
    _enable_peers()
    ops = _ops()
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(
        f"sweep sms {num_sms}  GB/s uses 1e9. base GB/s is both KV heads. "
        "std GB/s is the owner pages and includes the flag wait.",
        flush=True,
    )
    print(
        f"{'grp':<3} {'case':<22} {'base0':>8} {'std0':>8} {'std1':>8} "
        f"{'w_std':>7} {'baseGB':>8} {'stdGB':>8} {'cta0':>6} {'ctaS':>6} "
        f"{'GBcta0':>7} {'GBctaS':>7} {'spl':>4} {'GiB':>7}",
        flush=True,
    )
    ab = [(n, 32768) for n in (1, 2, 4, 8, 16, 32, 48, 64, 80, 108, 128, 160, 256)]
    for n, seq in ab:
        _sweep_case(ops, "ab", f"n{n}_s{seq}", [seq] * n, n, 1, num_sms)
    length = [(8, seq) for seq in (8192, 32768, 131072, 262144)]
    length += [(128, seq) for seq in (8192, 32768, 131072)]
    for n, seq in length:
        _sweep_case(ops, "c", f"n{n}_s{seq}", [seq] * n, n, 1, num_sms)
    for n_long in (4, 16, 32):
        _sweep_case(ops, "e", f"long{n_long}_s262144", [262144] * n_long, n_long, 1, num_sms)
        mixed = [262144] * n_long + [512] * 512
        _sweep_case(ops, "e", f"long{n_long}_s262144_sh", mixed, n_long, 1, num_sms)
    for n, seq in ((4, 262144), (16, 262144), (32, 262144), (8, 32768), (128, 32768), (256, 32768)):
        _sweep_case(ops, "f", f"n{n}_s{seq}", [seq] * n, n, 0, num_sms)
    print("sweep done", flush=True)


def main():
    if os.environ.get("HEAD_TP_COMM"):
        _comm()
        return
    if os.environ.get("HEAD_TP_SWEEP"):
        _sweep()
        return
    _page_tests()
    _enable_peers()
    _check(_Case(_ops(), [64, 96, 128, 17], 3))
    # Performance comparisons use the workspace workload registry and clock.
    root = Path(os.environ.get("NANOCP_ROOT", Path(__file__).resolve().parents[5] / "vllm_nanocp"))
    driver = root / "nanocp/bench_decode_cp/bench_perf.py"
    if not driver.is_file():
        raise SystemExit("Set NANOCP_ROOT to the vllm_nanocp checkout for comparative benchmarks")
    args = [sys.executable, str(driver), "--method", "head", *sys.argv[1:]]
    if os.environ.get("HEAD_TP_ONLY"):
        args.extend(["--case", os.environ["HEAD_TP_ONLY"]])
    os.environ.setdefault("FLASH_ATTN_ROOT", str(Path(__file__).resolve().parents[1]))
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
