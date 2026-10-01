# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bash-tool agent that runs next to the trainer and sends only commands to the sandbox."""

import asyncio
import json
import sys
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import Field
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

# Verifiers resolves plugin ids by top-level module name; register this module under one.
HARNESS_ID = __name__.replace(".", "_").lower()
sys.modules.setdefault(HARNESS_ID, sys.modules[__name__])

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a bash command in the persistent task sandbox. Filesystem "
            "state is preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


class AgentOutsideHarnessConfig(HarnessConfig):
    """Bash-tool agent loop that runs in the env-server process."""

    command_timeout_sec: float = Field(300.0, gt=0)
    """Per-command limit; past it the model gets a timeout error and continues."""

    max_tool_output_chars: int = Field(64 * 1024, gt=0)
    """Longer tool results keep only their tail."""


class AgentOutsideHarness(Harness[AgentOutsideHarnessConfig]):
    """Run the chat loop in this process and send each `bash` call to the sandbox.

    The sandbox never calls the model, so it can be a remote VM. Verifiers ends the loop at `max_turns`.
    """

    APPENDS_SYSTEM_PROMPT = True
    NEEDS_CONTAINER = False

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        if mcp_urls:
            raise ValueError("the agent-outside harness supports only its bash tool")
        system_prompt, prompt = self.resolve_prompt(data)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages.extend(message_to_wire(message) for message in prompt)

        # A 16k-token turn on a slow generator outlives openai's 600 s read
        # timeout, and its retry would resend the turn. The rollout deadline
        # bounds the call instead, as in Verifiers' own `null` harness.
        async with AsyncOpenAI(
            base_url=endpoint,
            api_key=secret,
            timeout=httpx.Timeout(None, connect=5.0),
        ) as client:
            while True:
                completion = await client.chat.completions.create(
                    model=ctx.model, messages=messages, tools=[BASH_TOOL]
                )
                message = completion.choices[0].message
                # Keep the assistant turn so each tool result can cite its call id.
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    break
                for tool_call in message.tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": await self._run_tool(
                                runtime,
                                name=tool_call.function.name,
                                arguments_json=tool_call.function.arguments,
                            ),
                        }
                    )

        # No program ran; the model's turns are already on the intercepted trace.
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def _run_tool(
        self, runtime: Runtime, *, name: str, arguments_json: str
    ) -> str:
        """Run one tool call in the sandbox and return the text the model sees."""
        if name != "bash":
            return f"error: unknown tool {name!r}; use bash"
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as error:
            return f"error: invalid JSON arguments: {error}"
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if not isinstance(command, str):
            return "error: expected a string 'command' field"

        try:
            result = await asyncio.wait_for(
                runtime.run(["bash", "-lc", command], {}),
                timeout=self.config.command_timeout_sec,
            )
        except TimeoutError:
            return (
                f"error: command timed out after {self.config.command_timeout_sec:g}s"
            )
        # Exit code last, so truncating from the front keeps it.
        content = (
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"exit_code: {result.exit_code}"
        )
        limit = self.config.max_tool_output_chars
        if len(content) > limit:
            return "[output truncated]\n" + content[-limit:]
        return content


# prime-rl loads the harness through a module that re-exports __all__.
__all__ = ["AgentOutsideHarness"]
