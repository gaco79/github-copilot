"""Base entity for GitHub Copilot."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Iterable
from typing import TYPE_CHECKING, Any

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.shared_params import FunctionDefinition
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

if TYPE_CHECKING:
    from . import GitHubCopilotConfigEntry

MAX_TOOL_ITERATIONS = 10


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format an HA LLM tool for the OpenAI chat completions API."""
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name=tool.name,
            description=tool.description or "",
            parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        ),
    )


def _convert_content_to_messages(
    content: Iterable[conversation.Content],
) -> list[ChatCompletionMessageParam]:
    """Convert HA chat log content to OpenAI chat completion messages."""
    messages: list[ChatCompletionMessageParam] = []

    for item in content:
        if isinstance(item, conversation.SystemContent):
            if item.content:
                messages.append(
                    ChatCompletionSystemMessageParam(
                        role="system", content=item.content
                    )
                )
        elif isinstance(item, conversation.UserContent):
            if item.content:
                messages.append(
                    ChatCompletionUserMessageParam(role="user", content=item.content)
                )
        elif isinstance(item, conversation.AssistantContent):
            msg = ChatCompletionAssistantMessageParam(role="assistant")
            if item.content:
                msg["content"] = item.content
            if item.tool_calls:
                msg["tool_calls"] = [
                    ChatCompletionMessageToolCallParam(
                        id=tc.id,
                        type="function",
                        function={
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.tool_args),
                        },
                    )
                    for tc in item.tool_calls
                ]
            messages.append(msg)
        elif isinstance(item, conversation.ToolResultContent):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=item.tool_call_id,
                    content=json.dumps(item.tool_result),
                )
            )

    return messages


async def _transform_stream(
    stream: openai.AsyncStream,
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform an OpenAI chat completion stream into HA delta format."""
    current_tool_calls: dict[int, dict[str, Any]] = {}
    started = False

    async for chunk in stream:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        finish_reason = choice.finish_reason

        # Signal the start of an assistant message
        if not started and (delta.role or delta.content is not None or delta.tool_calls):
            yield {"role": "assistant"}
            started = True

        if delta.content:
            yield {"content": delta.content}

        if delta.tool_calls:
            tc_delta: ChoiceDeltaToolCall
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in current_tool_calls:
                    current_tool_calls[idx] = {
                        "id": tc_delta.id or "",
                        "name": (tc_delta.function.name or "")
                        if tc_delta.function
                        else "",
                        "arguments": (tc_delta.function.arguments or "")
                        if tc_delta.function
                        else "",
                    }
                else:
                    if tc_delta.id:
                        current_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            current_tool_calls[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            current_tool_calls[idx]["arguments"] += (
                                tc_delta.function.arguments
                            )

        if finish_reason == "tool_calls" and current_tool_calls:
            if not started:
                yield {"role": "assistant"}
                started = True
            for idx in sorted(current_tool_calls):
                tc = current_tool_calls[idx]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=tc["id"],
                            tool_name=tc["name"],
                            tool_args=args,
                        )
                    ]
                }
            current_tool_calls = {}
            started = False


class GitHubCopilotBaseLLMEntity(Entity):
    """Base entity for GitHub Copilot LLM interactions."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(
        self, entry: GitHubCopilotConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="GitHub",
            model=subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure: vol.Schema | None = None,
    ) -> None:
        """Generate a response for the chat log using the GitHub Models API."""
        options = self.subentry.data
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        client: openai.AsyncOpenAI = self.entry.runtime_data

        for _iteration in range(MAX_TOOL_ITERATIONS):
            messages = _convert_content_to_messages(chat_log.content)

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                #"max_completion_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                #"temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                #"top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "stream": True,
            }

            if structure is not None:
                kwargs["response_format"] = {"type": "json_object"}

            tools: list[ChatCompletionToolParam] = []
            if chat_log.llm_api:
                tools = [
                    _format_tool(tool, chat_log.llm_api.custom_serializer)
                    for tool in chat_log.llm_api.tools
                ]
            if tools:
                kwargs["tools"] = tools

            try:
                stream = await client.chat.completions.create(**kwargs)
            except openai.AuthenticationError as err:
                self.entry.async_start_reauth(self.hass)
                raise HomeAssistantError("Authentication error") from err
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by GitHub Models: %s", err)
                raise HomeAssistantError("Rate limited or insufficient quota") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to GitHub Models: %s", err)
                raise HomeAssistantError("Error talking to GitHub Models") from err

            [
                _
                async for _ in chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _transform_stream(stream),
                )
            ]

            if not chat_log.unresponded_tool_results:
                break
                break
        else:
            LOGGER.warning(
                "Maximum tool iterations (%s) reached for entity %s with unresolved tool results",
                MAX_TOOL_ITERATIONS,
                self.entity_id,
            )
            raise HomeAssistantError(
                "Maximum tool iterations reached with unresolved tool calls"
            )