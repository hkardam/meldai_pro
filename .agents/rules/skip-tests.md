# Docker Test Execution Rule

## Objective
All automated test cases (`pytest`, unit tests, integration tests) MUST be executed inside the Docker app container.

## Guidelines
1. **Always Use Docker**: Execute tests using `docker compose exec app pytest <args>` (or `docker compose run --rm app pytest <args>`).
2. **No Host Execution**: Never run `pytest` directly on the local host machine, as the environment dependencies reside in Docker.
