# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-Bench environment for Verifiers: a bash-tool agent and Harbor tasks.

Framework-agnostic: these modules import only Verifiers, OpenAI, httpx, pydantic,
and the standard library, so any trainer can run the same agent. TitanRL's
recipes live in ``torchtitan_recipes.rl.verifiers_terminal_bench``.
"""
