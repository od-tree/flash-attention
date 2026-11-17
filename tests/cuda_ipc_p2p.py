"""CUDA IPC mapping for peer kernel stores.

Default backend is raw ``cudaDeviceEnablePeerAccess`` + foreign ``data_ptr``
(the historical path). On RTX 5090 / sm_120 that path can report peer OK while
kernel stores to the foreign VA still fault. The IPC backend exports each
buffer with ``cudaIpcGetMemHandle`` and opens it on the writer device with
``cudaIpcMemLazyEnablePeerAccess``, then builds pointer tables from those
opened addresses.

Select with ``P2P_BACKEND``:

- ``peer`` — original EnablePeerAccess path, kernel stores to foreign ``data_ptr``
- ``memcpy`` — ``cudaMemcpyPeer`` for the bytes. Kernel stores stay on the local buffer
- ``ipc`` — CUDA IPC pointer tables (not usable on this 5090 guest)
- ``auto`` (default) — ``memcpy`` when any visible GPU is sm_120+ or named 5090
"""

from __future__ import annotations

import ctypes
import os
from typing import Iterable

import torch

_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
_CUDA_SUCCESS = 0
_CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
_CUDA_ERROR_MAP_BUFFER_OBJECT_FAILED = 205


class _IpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


_LIB = None


def _cudart():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = None
    for soname in ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.dylib"):
        try:
            lib = ctypes.CDLL(soname)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError("libcudart was not found")
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    lib.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
    lib.cudaGetLastError.restype = ctypes.c_int
    lib.cudaDeviceCanAccessPeer.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
    ]
    lib.cudaDeviceCanAccessPeer.restype = ctypes.c_int
    lib.cudaIpcGetMemHandle.argtypes = [
        ctypes.POINTER(_IpcMemHandle), ctypes.c_void_p,
    ]
    lib.cudaIpcGetMemHandle.restype = ctypes.c_int
    lib.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), _IpcMemHandle, ctypes.c_uint,
    ]
    lib.cudaIpcOpenMemHandle.restype = ctypes.c_int
    lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    lib.cudaIpcCloseMemHandle.restype = ctypes.c_int
    _LIB = lib
    return lib


def _check(err: int, what: str) -> None:
    if err == _CUDA_SUCCESS:
        return
    lib = _cudart()
    msg = lib.cudaGetErrorString(err)
    text = msg.decode() if isinstance(msg, (bytes, bytearray)) else str(msg)
    raise RuntimeError(f"{what} returned {err}: {text}")


def p2p_backend() -> str:
    raw = os.environ.get("P2P_BACKEND", "auto").strip().lower()
    if raw in ("peer", "ipc", "memcpy"):
        return raw
    if raw not in ("", "auto"):
        raise ValueError(f"P2P_BACKEND must be peer, memcpy, ipc, or auto; got {raw!r}")
    if not torch.cuda.is_available():
        return "peer"
    for index in range(torch.cuda.device_count()):
        major, _minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        if major >= 12 or "5090" in name:
            return "memcpy"
    return "peer"


def use_ipc() -> bool:
    return p2p_backend() == "ipc"


def use_memcpy() -> bool:
    return p2p_backend() == "memcpy"


def _cuda_ids(devices: Iterable) -> list[int]:
    ids = []
    for device in devices:
        if isinstance(device, torch.device):
            resolved = device
        elif isinstance(device, int):
            resolved = torch.device("cuda", device)
        else:
            resolved = torch.device(device)
        if resolved.type == "cuda":
            ids.append(int(resolved.index))
    return ids


def enable_peers(devices, *, backend: str | None = None) -> str:
    """Enable cross-device access. Returns the backend that was selected.

    ``peer`` requires EnablePeerAccess success (0 or 704).
    ``memcpy`` and ``ipc`` also accept 205. ``memcpy`` moves bytes with
    ``cudaMemcpyPeer`` and does not store through a peer virtual address.
    """
    mode = (backend or p2p_backend()).lower()
    if mode not in ("peer", "ipc", "memcpy"):
        raise ValueError(f"backend must be peer, memcpy, or ipc, got {mode!r}")
    cuda_ids = _cuda_ids(devices)
    if len(set(cuda_ids)) != len(cuda_ids):
        raise ValueError(f"duplicate CUDA devices in {cuda_ids}")
    if len(cuda_ids) < 2:
        return mode

    lib = _cudart()
    for src in cuda_ids:
        for dst in cuda_ids:
            if src == dst:
                continue
            can = ctypes.c_int(0)
            _check(lib.cudaDeviceCanAccessPeer(ctypes.byref(can), src, dst),
                   f"cudaDeviceCanAccessPeer({src},{dst})")
            if can.value == 0 and mode == "peer":
                raise SystemExit(f"cuda peer access {src}->{dst} is not available")

    for src in cuda_ids:
        _check(lib.cudaSetDevice(src), f"cudaSetDevice({src})")
        for dst in cuda_ids:
            if src == dst:
                continue
            err = lib.cudaDeviceEnablePeerAccess(dst, 0)
            if mode == "peer":
                if err not in (_CUDA_SUCCESS, _CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED):
                    raise SystemExit(f"peer access {src}->{dst} returned {err}")
            else:
                if err not in (
                    _CUDA_SUCCESS,
                    _CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
                    _CUDA_ERROR_MAP_BUFFER_OBJECT_FAILED,
                ):
                    raise SystemExit(f"peer access {src}->{dst} returned {err}")
    # 205 is sticky. The next PyTorch call would report it even after a later
    # cudaMemcpyPeer succeeds, so drop it once the mode has accepted it.
    lib.cudaGetLastError()
    torch.cuda.set_device(cuda_ids[0])
    return mode


# Device index -> stream. Set only while a CUDA graph is being captured.
# Copies then become stream-ordered cudaMemcpyAsync nodes. cudaMemcpyPeerAsync
# itself returns "not permitted when stream is capturing" on this driver.
_GRAPH_STREAMS: dict[int, torch.cuda.Stream] = {}


def graph_streams_active() -> bool:
    return bool(_GRAPH_STREAMS)


class use_graph_streams:
    """Route peer copies onto these streams for the duration of a capture."""

    def __init__(self, streams):
        self.streams = {int(stream.device.index): stream for stream in streams}
        self._prev: dict[int, torch.cuda.Stream] = {}

    def __enter__(self):
        global _GRAPH_STREAMS
        self._prev = _GRAPH_STREAMS
        _GRAPH_STREAMS = self.streams
        return self

    def __exit__(self, exc_type, exc, tb):
        global _GRAPH_STREAMS
        _GRAPH_STREAMS = self._prev
        return False


def fork_graph_streams(root: torch.cuda.Stream) -> None:
    """Let the other devices join this capture without waiting for ``root``'s later work."""
    if len(_GRAPH_STREAMS) < 2:
        return
    event = root.record_event()
    for stream in _GRAPH_STREAMS.values():
        if stream is not root:
            stream.wait_event(event)


def capture_graph(streams, work):
    """Warm ``work`` once, then capture it.

    Timing events stay outside the graph. Events recorded inside a capture
    cannot be queried with ``elapsed_time`` on this driver (invalid argument).
    Replaying on the root stream still includes the other GPU, because the
    capture joins that stream back before the graph ends.
    """
    work()
    torch.cuda.synchronize()
    root = streams[0]
    for stream in streams[1:]:
        torch.cuda.set_device(stream.device)
        torch.cuda.set_stream(stream)
    torch.cuda.set_device(root.device)
    root.wait_stream(torch.cuda.current_stream(root.device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(root):
        with torch.cuda.graph(graph, stream=root):
            with use_graph_streams(streams):
                fork_graph_streams(root)
                work()
                join_graph_streams(root)
    torch.cuda.current_stream(root.device).wait_stream(root)
    torch.cuda.set_device(root.device)
    return graph


def replay_median(graph, root, rounds: int) -> float:
    wall0 = torch.cuda.Event(True)
    wall1 = torch.cuda.Event(True)
    samples = []
    for _ in range(rounds):
        with torch.cuda.stream(root):
            wall0.record(root)
            graph.replay()
            wall1.record(root)
        wall1.synchronize()
        samples.append(wall0.elapsed_time(wall1))
    samples.sort()
    return samples[len(samples) // 2]


def join_graph_streams(root: torch.cuda.Stream) -> None:
    """Make ``root`` wait until every other captured stream has finished."""
    for stream in _GRAPH_STREAMS.values():
        if stream is not root:
            root.wait_event(stream.record_event())


def _memcpy_async(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Device-to-device copy on the source stream. Peer pointers are allowed.

    The destination stream waits for that copy, so a later kernel there sees
    the bytes. This form is capturable; ``cudaMemcpyPeerAsync`` is not.
    """
    src_stream = _GRAPH_STREAMS[int(src.device.index)]
    dst_stream = _GRAPH_STREAMS[int(dst.device.index)]
    if src.device == dst.device:
        with torch.cuda.stream(src_stream):
            dst.copy_(src)
        return
    lib = _cudart()
    lib.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
    nbytes = src.numel() * src.element_size()
    _check(
        lib.cudaMemcpyAsync(
            ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()),
            ctypes.c_size_t(nbytes), ctypes.c_int(3),  # cudaMemcpyDeviceToDevice
            ctypes.c_void_p(src_stream.cuda_stream),
        ),
        "cudaMemcpyAsync",
    )
    if dst_stream is not src_stream:
        dst_stream.wait_event(src_stream.record_event())


def memcpy_peer(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy ``src`` onto ``dst``. Cross-device uses ``cudaMemcpyPeer``.

    Both tensors must be contiguous and the same size. Same-device copies
    use ``copy_`` so a peer mapping is not required.
    """
    if dst.dtype != src.dtype or dst.numel() != src.numel():
        raise ValueError(
            f"memcpy_peer shape/dtype {tuple(src.shape)} {src.dtype} -> {tuple(dst.shape)} {dst.dtype}"
        )
    if not src.is_contiguous():
        src = src.contiguous()
    if not dst.is_contiguous():
        raise ValueError("memcpy_peer destination must be contiguous")
    nbytes = src.numel() * src.element_size()
    if nbytes == 0:
        return
    if _GRAPH_STREAMS:
        _memcpy_async(dst, src)
        return
    if src.device == dst.device:
        dst.copy_(src)
        return
    if src.device.type != "cuda" or dst.device.type != "cuda":
        dst.copy_(src)
        return
    lib = _cudart()
    lib.cudaMemcpyPeer.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
    ]
    lib.cudaMemcpyPeer.restype = ctypes.c_int
    _check(
        lib.cudaMemcpyPeer(
            ctypes.c_void_p(dst.data_ptr()), int(dst.device.index),
            ctypes.c_void_p(src.data_ptr()), int(src.device.index),
            ctypes.c_size_t(nbytes),
        ),
        "cudaMemcpyPeer",
    )


def _as_device(device) -> torch.device:
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def copy_tensor(src: torch.Tensor, device, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Move ``src`` to ``device``. Cross-GPU on the memcpy backend uses ``cudaMemcpyPeer``.

    PyTorch ``.to()`` between these two 5090s returns
    ``cudaErrorMapBufferObjectFailed``. Same-device and CPU copies stay on ``.to()``.
    """
    if dtype is not None and src.dtype != dtype:
        src = src.to(dtype=dtype)
    target = _as_device(device)
    if src.device == target:
        return src
    if src.device.type == "cpu" and target.type == "cuda" and torch.cuda.is_current_stream_capturing():
        # A pageable H2D copy cannot be captured. Pinning is host-side.
        src = src.pin_memory()
        return src.to(target, non_blocking=True)
    if not (src.is_cuda and target.type == "cuda" and use_memcpy()):
        return src.to(target)
    packed = src.contiguous()
    dst = torch.empty(packed.shape, dtype=packed.dtype, device=target)
    memcpy_peer(dst, packed)
    return dst


def assign_tensor(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Write ``src`` into ``dst``. Cross-GPU on the memcpy backend uses ``cudaMemcpyPeer``."""
    if dst.shape != src.shape or dst.dtype != src.dtype:
        raise ValueError(
            f"assign_tensor shape/dtype {tuple(src.shape)} {src.dtype} -> {tuple(dst.shape)} {dst.dtype}"
        )
    if dst.device == src.device or not (dst.is_cuda and src.is_cuda and use_memcpy()):
        dst.copy_(src)
        return
    memcpy_peer(dst, src)


class IpcPeerMesh:
    """Export CUDA allocations and reopen them on each writer device."""

    def __init__(self, device_ids: list[int]):
        if len(device_ids) < 2:
            raise ValueError("IpcPeerMesh needs at least two CUDA devices")
        self.device_ids = [int(i) for i in device_ids]
        self._handles: dict[int, tuple[_IpcMemHandle, int, int]] = {}
        # storage_ptr -> (handle, owner_device, storage_nbytes_hint)
        self._opened: dict[tuple[int, int], int] = {}
        # (storage_ptr, writer_device) -> opened storage base on writer
        self._opened_bases: list[int] = []
        self._keep: list[torch.Tensor] = []
        self._lib = _cudart()

    def export(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cuda":
            return
        if not tensor.is_contiguous():
            raise ValueError("IPC export requires a contiguous tensor")
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        if storage_ptr == 0 or storage_ptr in self._handles:
            self._keep.append(tensor)
            return
        owner = int(tensor.device.index)
        handle = _IpcMemHandle()
        prev = torch.cuda.current_device()
        try:
            _check(self._lib.cudaSetDevice(owner), f"cudaSetDevice({owner})")
            _check(
                self._lib.cudaIpcGetMemHandle(ctypes.byref(handle), storage_ptr),
                f"cudaIpcGetMemHandle on device {owner}",
            )
        finally:
            self._lib.cudaSetDevice(prev)
        self._handles[storage_ptr] = (handle, owner, int(storage.nbytes()))
        self._keep.append(tensor)

    def writer_ptr(self, tensor: torch.Tensor, writer_device: int) -> int:
        """Address of ``tensor`` usable from kernels launched on ``writer_device``."""
        if tensor.device.type != "cuda":
            raise ValueError("writer_ptr expects a CUDA tensor")
        owner = int(tensor.device.index)
        data_ptr = int(tensor.data_ptr())
        if owner == int(writer_device):
            return data_ptr
        storage_ptr = int(tensor.untyped_storage().data_ptr())
        offset = data_ptr - storage_ptr
        if storage_ptr not in self._handles:
            self.export(tensor)
        opened_base = self._open_storage(storage_ptr, int(writer_device))
        return opened_base + offset

    def _open_storage(self, storage_ptr: int, writer_device: int) -> int:
        key = (storage_ptr, writer_device)
        hit = self._opened.get(key)
        if hit is not None:
            return hit
        handle, owner, _nbytes = self._handles[storage_ptr]
        if owner == writer_device:
            self._opened[key] = storage_ptr
            return storage_ptr
        opened = ctypes.c_void_p()
        prev = torch.cuda.current_device()
        try:
            _check(self._lib.cudaSetDevice(writer_device), f"cudaSetDevice({writer_device})")
            _check(
                self._lib.cudaIpcOpenMemHandle(
                    ctypes.byref(opened),
                    handle,
                    _CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
                ),
                f"cudaIpcOpenMemHandle writer={writer_device} owner={owner}",
            )
        finally:
            self._lib.cudaSetDevice(prev)
        base = int(opened.value)
        self._opened[key] = base
        self._opened_bases.append(base)
        return base

    def pointer_table(
        self,
        tensors: list[torch.Tensor],
        writer_device: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ptrs = [self.writer_ptr(tensor, writer_device) for tensor in tensors]
        table = torch.tensor(ptrs, dtype=torch.int64)
        if device is not None and device.type == "cuda":
            host = table.pin_memory()
            out = torch.empty(host.shape, dtype=torch.int64, device=device)
            out.copy_(host, non_blocking=True)
            return out
        if device is not None:
            return table.to(device=device)
        return table

    def close(self) -> None:
        lib = self._lib
        prev = torch.cuda.current_device()
        try:
            for (storage_ptr, writer_device), base in list(self._opened.items()):
                handle_owner = self._handles.get(storage_ptr)
                if handle_owner is None:
                    continue
                _owner = handle_owner[1]
                if writer_device == _owner or base == storage_ptr:
                    continue
                lib.cudaSetDevice(writer_device)
                lib.cudaIpcCloseMemHandle(ctypes.c_void_p(base))
        finally:
            lib.cudaSetDevice(prev)
        self._opened.clear()
        self._opened_bases.clear()
        self._handles.clear()
        self._keep.clear()
