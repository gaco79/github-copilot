"""Constants for the GitHub Copilot integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "github_copilot"
LOGGER: logging.Logger = logging.getLogger(__package__)

# GitHub OAuth App Client ID (device flow) - same app used by the HA GitHub integration
GITHUB_OAUTH_CLIENT_ID = "1440cafcc86e3ea5d6a2"

# GitHub Models API
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
GITHUB_CATALOG_URL = "https://models.github.ai/catalog/models"

# Config keys
CONF_CHAT_MODEL = "chat_model"
CONF_PROMPT = "prompt"

# Default names
DEFAULT_NAME = "GitHub Copilot"
DEFAULT_CONVERSATION_NAME = "GitHub Copilot"
DEFAULT_AI_TASK_NAME = "GitHub Copilot AI Task"

# Recommended defaults
RECOMMENDED_CHAT_MODEL = "openai/gpt-4o"

RECOMMENDED_CONVERSATION_OPTIONS: dict[str, Any] = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

RECOMMENDED_AI_TASK_OPTIONS: dict[str, Any] = {}
