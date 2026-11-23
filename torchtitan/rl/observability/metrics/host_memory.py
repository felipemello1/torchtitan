# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process and host memory, read from Linux /proc files."""


def read_proc_kib(path: str, field: str) -> int:
    """Return the kB value of one field of a Linux /proc file.

    Example: the line "VmRSS:  1745628 kB" in /proc/self/status -> 1745628.
    """
    with open(path) as proc_file:
        for line in proc_file:
            if line.startswith(f"{field}:"):
                return int(line.split()[1])
    raise RuntimeError(f"{path} has no {field} line")


def host_memory_available_gib() -> float:
    """The memory this host can still give to new allocations without swapping, in GiB
    (/proc/meminfo MemAvailable)."""
    return read_proc_kib("/proc/meminfo", "MemAvailable") / 2**20
