"""AI Task entity for GitHub Copilot."""

from __future__ import annotations

from json import JSONDecodeError
from typing import TYPE_CHECKING

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .entity import GitHubCopilotBaseLLMEntity

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

    from . import GitHubCopilotConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GitHubCopilotConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [GitHubCopilotAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GitHubCopilotAITaskEntity(
    ai_task.AITaskEntity,
    GitHubCopilotBaseLLMEntity,
):
    """GitHub Copilot AI Task entity."""

    def __init__(
        self, entry: GitHubCopilotConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        self._attr_supported_features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
            | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(chat_log, task.structure)

        if not chat_log.content:
            raise HomeAssistantError("Chat log has no content")

        last_content = chat_log.content[-1]

        if not isinstance(last_content, conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = last_content.content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except (JSONDecodeError, ValueError) as err:
            raise HomeAssistantError(
                "Error parsing structured response from GitHub Copilot"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
