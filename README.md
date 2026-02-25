![GitHub Release](https://img.shields.io/github/v/release/gaco79/github-copilot?style=for-the-badge)
![Downloads](https://img.shields.io/github/downloads/gaco79/github-copilot/total?style=for-the-badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/gaco79/github-copilot?style=for-the-badge)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/gaco79/github-copilot/cd.yml?style=for-the-badge)
[![BuyMeACoffee](https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee&style=for-the-badge)](https://www.buymeacoffee.com/gaco79)

# GitHub Copilot Home Assistant Integration

This custom component provides a Home Assistent integration to interact with GitHub Copilot.

A choice of AI models is available. Make sure to consult [GitHub documentation](https://docs.github.com/en/copilot/concepts/billing/copilot-requests#model-multipliers) for pricing.

Conversation agent and AI Task entities can be created. Different entities can use different models.

## Development Environment

 * Clone this repo
 * Make sure you have docker installed
 * Place code for this integration in the custom_components/github_copilot/ folder
 * Use `docker compose` to spin up an instance of Home Assistant. It should now be possible to add the integration there.