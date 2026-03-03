# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Optional
import os
import psutil
import socket
import torch
import torch.distributed as dist
import pynvml
import psutil
from codetiming import Timer

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.logger import DecoratorLoggerBase

_pynvmlInited = False

def get_local_ip():
    """Get the local IP address of the current machine."""
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"

def get_physical_device_id():
    """Get the physical GPU device ID by matching device UUID.
    
    In distributed environments with CUDA_VISIBLE_DEVICES, torch.cuda.current_device()
    always returns 0 (logical device). This function maps the current torch device's
    UUID to its physical GPU index.
    
    Returns:
        int: Physical GPU device index
        
    Raises:
        RuntimeError: If GPU device matching fails
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Get the UUID of current torch device
    current_device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    current_uuid = current_device_props.uuid
    # Convert torch UUID object to string
    current_uuid_str = str(current_uuid)
    
    # Initialize pynvml and find matching physical device
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    for physical_id in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_id)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        # Match UUID - normalize by removing "GPU-" or "MIG-" prefixes and comparing
        uuid_normalized = uuid.replace("GPU-", "").replace("MIG-", "").lower()
        current_uuid_normalized = current_uuid_str.replace("GPU-", "").replace("MIG-", "").lower()
        if uuid_normalized == current_uuid_normalized:
            pynvml.nvmlShutdown()
            return physical_id
    
    pynvml.nvmlShutdown()
    
    # If no match found, raise error
    raise RuntimeError(f"Failed to match device UUID {current_uuid_str} to any physical GPU. "
                       f"Checked {device_count} devices.")


def get_cpu_memory_info() -> dict:
    """Return node-level system RAM, this process's RSS, and top 10 processes by RSS, all in GB."""
    try:
        vm = psutil.virtual_memory()
        current = psutil.Process()
        proc_rss = current.memory_info().rss
        process_list = []
        for p in psutil.process_iter(["pid", "name"]):
            try:
                rss = p.memory_info().rss
                name = (p.info.get("name") or p.name() or "?").strip()
                if not name:
                    name = "?"
                process_list.append((p.info["pid"], name[:64], rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        process_list.sort(key=lambda x: x[2], reverse=True)
        top10 = [
            {"pid": pid, "name": name, "rss_gb": round(rss / (1024**3), 2)}
            for pid, name, rss in process_list[:10]
        ]
        if not top10:
            top10 = [
                {
                    "pid": current.pid,
                    "name": (current.name() or "current")[:64],
                    "rss_gb": round(proc_rss / (1024**3), 2),
                }
            ]
        return {
            "node_total_gb": round(vm.total / (1024**3), 2),
            "node_used_gb": round(vm.used / (1024**3), 2),
            "node_avail_gb": round(vm.available / (1024**3), 2),
            "node_used_pct": round(vm.percent, 1),
            "proc_rss_gb": round(proc_rss / (1024**3), 2),
            "top10_procs": top10,
        }
    except Exception:
        return {}


def format_cpu_memory_str(cpu_info: Optional[dict] = None) -> str:
    """Format CPU memory info (from get_cpu_memory_info()) as a log string. Always includes top10_procs."""
    if not cpu_info:
        return "N/A"
    top10 = cpu_info.get("top10_procs", [])
    top10_str = ", ".join(
        f"pid={p['pid']}({p['name']}):{p['rss_gb']}GB" for p in top10
    ) if top10 else "N/A"
    return (
        f"cpu_node_used/avail/total (GB): "
        f"{cpu_info.get('node_used_gb', '?')}/{cpu_info.get('node_avail_gb', '?')}/{cpu_info.get('node_total_gb', '?')} "
        f"({cpu_info.get('node_used_pct', '?')}%), proc_rss: {cpu_info.get('proc_rss_gb', '?')} GB, "
        f"top10_procs: [{top10_str}]"
    )


def get_gpu_memory_by_processes(device_id: int = None):
    # Check if we're on a GPU device
    if not torch.cuda.is_available():
        return {'device_id': 'cpu', 'error': 'GPU memory info not available on CPU device'}

    if device_id is None:
        # Use physical device ID instead of logical device ID
        device_id = get_physical_device_id()
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    pynvml.nvmlShutdown()
    
    def get_process_name(pid):
        try:
            return psutil.Process(pid).cmdline()[0]
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            return "Unknown"
    
    return {
        'device_id': device_id,
        'total_gb': mem_info.total / (1024 ** 3),
        'used_gb': mem_info.used / (1024 ** 3),
        'free_gb': mem_info.free / (1024 ** 3),
        'processes': [
            {'pid': p.pid, 'process_name': get_process_name(p.pid), 'memory_gb': p.usedGpuMemory / (1024 ** 3)}
            for p in processes
        ]
    }

def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> tuple[str]:
    """Get current memory usage.

    Note that CPU device memory info is always 0.

    Args:
        unit (str, optional): The unit of memory measurement. Defaults to "GB".
        precision (int, optional): The number of decimal places to round memory values. Defaults to 2.

    Returns:
        tuple[str]: A tuple containing memory allocated, memory reserved, memory used, and memory total
        in the specified unit.
    """
    assert unit in ["GB", "MB", "KB"]
    device = get_torch_device()
    # torch.cpu.memory_allocated() does not exist
    if device == torch.cpu:
        return "0.00", "0.00", "0.00", "0.00"

    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.WARNING, rank: int = 0):
    """Log GPU memory usage information.

    Args:
        head (str): A descriptive header for the memory usage log message.
        logger (logging.Logger, optional): Logger instance to use for logging. If None, prints to stdout.
        level: Logging level to use. Defaults to logging.DEBUG.
        rank (int): The rank of the process to log memory for. Defaults to 0.
    """
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        cpu = get_cpu_memory_info()
        cpu_str = format_cpu_memory_str(cpu)
        local_ip = get_local_ip()
        # Get physical device ID to query correct GPU
        physical_device_id = get_physical_device_id()
        message = (
            f"[ip={local_ip}] {head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}, "
            f"memory breakdown: {get_gpu_memory_by_processes(physical_device_id)}"
            f", cpu memory: {cpu_str}" if cpu_str else ""
        )

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)


class GPUMemoryLogger(DecoratorLoggerBase):
    """A decorator class to log GPU memory usage.

    Example:
        >>> from verl.utils.profiler.performance import GPUMemoryLogger
        >>> @GPUMemoryLogger(role="actor")
        >>> def update_actor(self, batch):
        ...     # real actor update logics
        ...     return
    """

    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, log_only_rank_0: bool = True):
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        super().__init__(role, logger, level, rank, log_only_rank_0)

    def __call__(self, decorated_function: callable):
        def f(*args, **kwargs):
            return self.log(decorated_function, *args, **kwargs)

        return f

    def log(self, func, *args, **kwargs):
        name = func.__name__
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        cpu = get_cpu_memory_info()
        cpu_str = format_cpu_memory_str(cpu)
        local_ip = get_local_ip()
        # Get physical device ID to query correct GPU
        physical_device_id = get_physical_device_id()
        message = (
            f"[ip={local_ip}] Before {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
            f"memory breakdown: {get_gpu_memory_by_processes(physical_device_id)}"
            f", cpu memory: {cpu_str}" if cpu_str else ""
        )
        self.logging_function(message)

        output = func(*args, **kwargs)

        cpu = get_cpu_memory_info()
        cpu_str = format_cpu_memory_str(cpu)
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"[ip={local_ip}] After {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
            f"memory breakdown: {get_gpu_memory_by_processes(physical_device_id)}"
            f", cpu memory: {cpu_str}" if cpu_str else ""
        )

        self.logging_function(message)
        return output


def log_print(ctn: Any):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    file_name = frame.f_code.co_filename.split("/")[-1]
    print(f"[{current_time}-{file_name}:{line_number}:{function_name}]: {ctn}")


def _timer(name: str, timing_raw: dict[str, float]):
    """Inner function that handles the core timing logic.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def simple_timer(name: str, timing_raw: dict[str, float]):
    """Context manager for basic timing without NVTX markers.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with platform markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds platform markers for profiling.
    This function is a default implementation when hardware profiler is not available.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the marker. Defaults to None.
        domain (Optional[str]): Domain for the marker. Defaults to None.
        category (Optional[str]): Category for the marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


def reduce_timing(
    timing_raw: dict[str, float], reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.AVG
) -> dict[str, float]:
    """Reduce timing information across all processes.

    This function uses distributed communication to gather and sum the timing
    information from all processes in a distributed environment.

    Args:
        timing_raw (Dict[str, float]): Dictionary containing timing information.

    Returns:
        Dict[str, float]: Reduced timing information.
    """
    if not dist.is_initialized():
        return timing_raw

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])
    timing_list = torch.tensor(timing_list, dtype=torch.float32, device=get_device_id())
    torch.distributed.all_reduce(timing_list, op=reduce_op)
    timing_list = [tensor.item() for tensor in timing_list.to("cpu")]
    timing_generate = {key_list[i]: timing_list[i] for i in range(len(key_list))}
    return timing_generate


def topk_reduce_ratio_min_max(timing: float, k: int = 10) -> tuple[float, float, float]:
    """Calculate topk items take-up ratio, and min/max timing across all ranks."""
    if not dist.is_initialized():
        return -1.0, -1.0, -1.0

    world_size = dist.get_world_size()
    timing_tensor = torch.tensor(timing, dtype=torch.float32, device=get_device_id())
    tensor_list = [torch.zeros(1, dtype=torch.float32, device=get_device_id()) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, timing_tensor)
    tensor_stack = torch.stack(tensor_list)
    timing_min = tensor_stack.min().cpu().item()
    timing_max = tensor_stack.max().cpu().item()
    top_k_percentile = torch.quantile(tensor_stack, 1 - k / 100)
    tail_ratio = torch.mean((tensor_stack > top_k_percentile).float()).cpu().item()
    return tail_ratio, timing_min, timing_max


def gather_timing(timing_raw: dict[str, float]) -> dict[str, list[float]]:
    if not dist.is_initialized():
        return {k: [v] for k, v in timing_raw.items()}

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])

    world_size = torch.distributed.get_world_size()

    object_gather_list = [None] * world_size

    torch.distributed.all_gather_object(object_gather_list, timing_list)

    timing_generate = {
        key_list[i]: [timing_list[i] for timing_list in object_gather_list] for i in range(len(key_list))
    }

    return timing_generate
