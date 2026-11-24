# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host memory, read from Linux /proc/meminfo."""


def host_memory_available_gib() -> float:
    """The memory this host can still give to new allocations without swapping, in GiB
    (/proc/meminfo MemAvailable).

    Example: the line "MemAvailable:  163577860 kB" -> 156.0.
    """
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    raise RuntimeError("/proc/meminfo has no MemAvailable line")
