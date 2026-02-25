"""Config flow for GitHub Copilot integration."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
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
    SOURCE_REAUTH,
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
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    GITHUB_CATALOG_URL,
    GITHUB_OAUTH_CLIENT_ID,
    LOGGER,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
)


async def _fetch_models(hass: HomeAssistant, access_token: str) -> list[SelectOptionDict]:
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
                options: list[SelectOptionDict] = []
                for model in data:
                    if "id" not in model:
                        continue
                    model_id = model["id"]
                    summary = model.get("summary", "")
                    label = f"{model_id} - {summary}" if summary else model_id
                    options.append(SelectOptionDict(value=model_id, label=label))
                if options:
                    return options
    except Exception:  # noqa: BLE001
        LOGGER.debug("Could not fetch models from catalog, using default")
    return [SelectOptionDict(value=RECOMMENDED_CHAT_MODEL, label=RECOMMENDED_CHAT_MODEL)]


class GitHubCopilotConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for GitHub Copilot."""

    VERSION = 1

    login_task: asyncio.Task | None = None

    def __init__(self) -> None:
        """Initialize."""
        self._device: GitHubDeviceAPI | None = None
        self._login: GitHubLoginOauthModel | None = None
        self._login_device: GitHubLoginDeviceModel | None = None
        self._access_token: str | None = None

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")

        return await self.async_step_device(user_input)

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Perform reauth upon an authentication error."""
        self._device = None
        self._login = None
        self._login_device = None
        self.login_task = None
        return await self.async_step_device()

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
        """Store token and proceed to model selection."""
        if TYPE_CHECKING:
            assert self._login is not None

        if self.source == SOURCE_REAUTH:
            return self.async_update_reload_and_abort(
                self._get_reauth_entry(),
                data_updates={CONF_ACCESS_TOKEN: self._login.access_token},
            )

        self._access_token = self._login.access_token
        return await self.async_step_model()

    async def async_step_model(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Let the user select a model from the GitHub Models catalog."""
        if TYPE_CHECKING:
            assert self._access_token is not None

        if user_input is not None:
            selected_model = user_input[CONF_CHAT_MODEL]
            conv_options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
            conv_options[CONF_CHAT_MODEL] = selected_model
            ai_task_options = RECOMMENDED_AI_TASK_OPTIONS.copy()
            ai_task_options[CONF_CHAT_MODEL] = selected_model
            return self.async_create_entry(
                title=DEFAULT_CONVERSATION_NAME,
                data={CONF_ACCESS_TOKEN: self._access_token},
                subentries=[
                    {
                        "subentry_type": "conversation",
                        "data": conv_options,
                        "title": DEFAULT_CONVERSATION_NAME,
                        "unique_id": None,
                    },
                    {
                        "subentry_type": "ai_task_data",
                        "data": ai_task_options,
                        "title": DEFAULT_AI_TASK_NAME,
                        "unique_id": None,
                    },
                ],
            )

        model_options = await _fetch_models(self.hass, self._access_token)

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=model_options,
                            mode=SelectSelectorMode.DROPDOWN,
                            custom_value=True,
                        )
                    ),
                }
            ),
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
    _models: list[SelectOptionDict]

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
        model_values = [opt["value"] for opt in self._models]
        if current_model not in model_values:
            self._models.insert(0, SelectOptionDict(value=current_model, label=current_model))

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

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)

            if self._is_new:
                data = user_input.copy()
                title = data.pop(CONF_NAME, DEFAULT_CONVERSATION_NAME)
                return self.async_create_entry(
                    title=title,
                    data=data,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=user_input,
            )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), options
            ),
        )
