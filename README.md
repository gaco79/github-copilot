# GitHub Copilot Home Assistant Integration

This custom component provides a Home Assistent integration to interact with GitHub Copilot.

## Roadmap

[ ] Allow OAuth login to GitHub
  * Existing Github integration which already achieves this is provided in information/github/ folder
  
[ ] Allow user to choose which model should be used to fulfil requests for a given service
  * List of available models availabe via GitHub REST API. See information/github-REST-api/ folder for details
  * Provide model information from the API.
  * Link to [GitHub Model Multipliers](https://docs.github.com/en/copilot/concepts/billing/copilot-requests#model-multipliers) to explain costs of using different models.
  * Allow user to create multiple services, each using a different model

[ ] Each service should provide a `conversation agent` and `AI Task` device / entity.

## Environment

Use `docker compose` to spin up an instance of Home Assistant. It should be possible to add the integration there.