"""The GitHub Copilot integration."""

from __future__ import annotations

import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ACCESS_TOKEN, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.httpx_client import get_async_client

from .const import DOMAIN, GITHUB_MODELS_BASE_URL, LOGGER

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION)

type GitHubCopilotConfigEntry = ConfigEntry[openai.AsyncOpenAI]


async def async_setup_entry(
    hass: HomeAssistant, entry: GitHubCopilotConfigEntry
) -> bool:
    """Set up GitHub Copilot from a config entry."""
    client = openai.AsyncOpenAI(
        base_url=GITHUB_MODELS_BASE_URL,
        api_key=entry.data[CONF_ACCESS_TOKEN],
        http_client=get_async_client(hass),
    )

    try:
        await hass.async_add_executor_job(
            client.with_options(timeout=10.0).models.list
        )
    except openai.AuthenticationError as err:
        raise ConfigEntryAuthFailed(err) from err
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: GitHubCopilotConfigEntry
) -> bool:
    """Unload GitHub Copilot."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(
    hass: HomeAssistant, entry: GitHubCopilotConfigEntry
) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)