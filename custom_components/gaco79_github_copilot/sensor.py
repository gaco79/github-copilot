"""Sensor platform for GitHub Copilot rate limit monitoring."""

from __future__ import annotations

import datetime
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import GitHubCopilotConfigEntry
from .const import DOMAIN, SIGNAL_RATE_LIMIT_UPDATED


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GitHubCopilotConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up GitHub Copilot rate limit sensor."""
    async_add_entities([GitHubCopilotRateLimitSensor(config_entry)])


class GitHubCopilotRateLimitSensor(SensorEntity):
    """Diagnostic sensor showing remaining GitHub Models API rate limit requests."""

    _attr_has_entity_name = True
    _attr_translation_key = "rate_limit_remaining"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "requests"

    def __init__(self, entry: GitHubCopilotConfigEntry) -> None:
        """Initialize the sensor."""
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_rate_limit_remaining"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="GitHub Copilot",
            manufacturer="GitHub",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        """Register dispatcher callback when added to hass."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_RATE_LIMIT_UPDATED}_{self.entry.entry_id}",
                self._handle_rate_limit_update,
            )
        )

    @callback
    def _handle_rate_limit_update(self) -> None:
        """Handle a rate limit update dispatched from the entity."""
        self.async_write_ha_state()

    @property
    def native_value(self) -> int | None:
        """Return the number of remaining API requests."""
        remaining = self.entry.runtime_data.rate_limit.get("remaining")
        if remaining is not None:
            try:
                return int(remaining)
            except (ValueError, TypeError):
                pass
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra state attributes with full rate limit context."""
        rate_limit = self.entry.runtime_data.rate_limit
        attrs: dict[str, Any] = {}

        if limit := rate_limit.get("limit"):
            try:
                attrs["limit"] = int(limit)
            except (ValueError, TypeError):
                pass

        if used := rate_limit.get("used"):
            try:
                attrs["used"] = int(used)
            except (ValueError, TypeError):
                pass

        if resource := rate_limit.get("resource"):
            attrs["resource"] = resource

        if reset := rate_limit.get("reset"):
            try:
                attrs["reset"] = datetime.datetime.fromtimestamp(
                    int(reset), tz=datetime.UTC
                ).isoformat()
            except (ValueError, TypeError):
                pass

        return attrs
