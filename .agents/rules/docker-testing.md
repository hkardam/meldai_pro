# Always Run Test Cases Inside Docker

## Objective
All test cases (`pytest`, unit tests, integration tests) MUST be executed inside the Docker container environment, never directly on the host machine.

## Guidelines
1. **Container Execution**:
   - Use `docker compose exec app pytest <test_path>` if the app container is already running.
   - Use `docker compose run --rm app pytest <test_path>` or `make test` if creating a fresh temporary container.
2. **Never Run Host pytest**: Do NOT invoke `pytest` directly in the host terminal environment, as dependencies, ontologies, and database services are hosted inside Docker.
