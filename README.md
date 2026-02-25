# GitHub Copilot Home Assistant Integration

This custom component provides a Home Assistent integration to interact with GitHub Copilot.

A choice of AI models is available. Make sure to consult [GitHub documentation](https://docs.github.com/en/copilot/concepts/billing/copilot-requests#model-multipliers) for pricing.

Conversation agent and AI Task entities can be created. Different entities can use different models.

## Development Environment

 * Clone this repo
 * Make sure you have docker installed
 * Place code for this integration in the custom_components/github_copilot/ folder
 * Use `docker compose` to spin up an instance of Home Assistant. It should now be possible to add the integration there.