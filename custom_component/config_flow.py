"""Config flow for GitHub Copilot integration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from aiogithubapi import (
    GitHubDeviceAPI,
    GitHubException,
    GitHubLoginDeviceModel,
    GitHubLoginOauthModel,
)
from aiogithubapi.const import OAUTH_USER_LOGIN
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import (
    SERVER_SOFTWARE,
    async_get_clientsession,
)
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    GITHUB_CATALOG_URL,
    GITHUB_OAUTH_CLIENT_ID,
    LOGGER,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)


async def _fetch_models(hass: HomeAssistant, access_token: str) -> list[str]:
    """Fetch available models from the GitHub Models catalog."""
    session = async_get_clientsession(hass)
    try:
        async with session.get(
            GITHUB_CATALOG_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=10,
        ) as response:
            if response.status == 200:
                data = await response.json()
                return [model["id"] for model in data if "id" in model]
    except Exception:  # noqa: BLE001
        LOGGER.debug("Could not fetch models from catalog, using default")
    return [RECOMMENDED_CHAT_MODEL]


class GitHubCopilotConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for GitHub Copilot."""

    VERSION = 1

    login_task: asyncio.Task | None = None

    def __init__(self) -> None:
        """Initialize."""
        self._device: GitHubDeviceAPI | None = None
        self._login: GitHubLoginOauthModel | None = None
        self._login_device: GitHubLoginDeviceModel | None = None

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")

        return await self.async_step_device(user_input)

    async def async_step_device(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle device OAuth steps."""

        async def _wait_for_login() -> None:
            if TYPE_CHECKING:
                assert self._device is not None
                assert self._login_device is not None

            response = await self._device.activation(
                device_code=self._login_device.device_code
            )
            self._login = response.data

        if not self._device:
            self._device = GitHubDeviceAPI(
                client_id=GITHUB_OAUTH_CLIENT_ID,
                session=async_get_clientsession(self.hass),
                client_name=SERVER_SOFTWARE,
            )

            try:
                response = await self._device.register()
                self._login_device = response.data
            except GitHubException as exception:
                LOGGER.exception(exception)
                return self.async_abort(reason="could_not_register")

        if self.login_task is None:
            self.login_task = self.hass.async_create_task(_wait_for_login())

        if self.login_task.done():
            if self.login_task.exception():
                return self.async_show_progress_done(next_step_id="could_not_register")
            return self.async_show_progress_done(next_step_id="finish")

        if TYPE_CHECKING:
            assert self._login_device is not None

        return self.async_show_progress(
            step_id="device",
            progress_action="wait_for_device",
            description_placeholders={
                "url": OAUTH_USER_LOGIN,
                "code": self._login_device.user_code,
            },
            progress_task=self.login_task,
        )

    async def async_step_finish(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Create the config entry after successful login."""
        if TYPE_CHECKING:
            assert self._login is not None

        return self.async_create_entry(
            title=DEFAULT_CONVERSATION_NAME,
            data={CONF_ACCESS_TOKEN: self._login.access_token},
            subentries=[
                {
                    "subentry_type": "conversation",
                    "data": RECOMMENDED_CONVERSATION_OPTIONS,
                    "title": DEFAULT_CONVERSATION_NAME,
                    "unique_id": None,
                },
                {
                    "subentry_type": "ai_task_data",
                    "data": RECOMMENDED_AI_TASK_OPTIONS,
                    "title": DEFAULT_AI_TASK_NAME,
                    "unique_id": None,
                },
            ],
        )

    async def async_step_could_not_register(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle issues during device registration."""
        return self.async_abort(reason="could_not_register")

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": GitHubCopilotSubentryFlowHandler,
            "ai_task_data": GitHubCopilotSubentryFlowHandler,
        }


class GitHubCopilotSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing GitHub Copilot subentries (services)."""

    options: dict[str, Any]
    _models: list[str]

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        if self._subentry_type == "ai_task_data":
            self.options = RECOMMENDED_AI_TASK_OPTIONS.copy()
        else:
            self.options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage initial options."""
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        options = self.options

        # Fetch available models
        entry = self._get_entry()
        self._models = await _fetch_models(
            self.hass, entry.data[CONF_ACCESS_TOKEN]
        )
        # Ensure the current model is in the list
        current_model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        if current_model not in self._models:
            self._models.insert(0, current_model)

        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        if suggested_llm_apis := options.get(CONF_LLM_HASS_API):
            if isinstance(suggested_llm_apis, str):
                suggested_llm_apis = [suggested_llm_apis]
            valid_apis = {api.id for api in llm.async_get_apis(self.hass)}
            options[CONF_LLM_HASS_API] = [
                api for api in suggested_llm_apis if api in valid_apis
            ]

        step_schema: dict[Any, Any] = {}

        if self._is_new:
            if self._subentry_type == "ai_task_data":
                default_name = DEFAULT_AI_TASK_NAME
            else:
                default_name = DEFAULT_CONVERSATION_NAME
            step_schema[vol.Required(CONF_NAME, default=default_name)] = str

        step_schema[
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)},
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=self._models,
                mode=SelectSelectorMode.DROPDOWN,
                custom_value=True,
            )
        )

        if self._subentry_type == "conversation":
            step_schema.update(
                {
                    vol.Optional(
                        CONF_PROMPT,
                        description={
                            "suggested_value": options.get(
                                CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                            )
                        },
                    ): TemplateSelector(),
                    vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                        SelectSelectorConfig(options=hass_apis, multiple=True)
                    ),
                }
            )

        step_schema[
            vol.Required(CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False))
        ] = bool

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)

            if user_input[CONF_RECOMMENDED]:
                if self._is_new:
                    data = user_input.copy()
                    title = data.pop(CONF_NAME, DEFAULT_CONVERSATION_NAME)
                    # Ensure recommended advanced options are included when using recommended settings
                    data.setdefault(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
                    data.setdefault(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
                    data.setdefault(CONF_TOP_P, RECOMMENDED_TOP_P)
                    return self.async_create_entry(
                        title=title,
                        data=data,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=user_input,
                )

            options.update(user_input)
            if CONF_LLM_HASS_API in options and CONF_LLM_HASS_API not in user_input:
                options.pop(CONF_LLM_HASS_API)
            return await self.async_step_advanced()

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), options
            ),
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage advanced model options."""
        options = self.options

        step_schema: dict[Any, Any] = {
            vol.Optional(
                CONF_MAX_TOKENS,
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        }

        if user_input is not None:
            options.update(user_input)
            if self._is_new:
                return self.async_create_entry(
                    title=options.pop(CONF_NAME, DEFAULT_CONVERSATION_NAME),
                    data=options,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=options,
            )

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), options
            ),
        )
