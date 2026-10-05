# Skip Automated Test Cases Rule

## Objective
Do not write or execute automated test cases (`pytest`, unit tests, integration tests). The developer will test all changes manually.

## Guidelines
1. **No Test Authoring**: Do NOT create new test files or add new test cases to existing test suites unless explicitly requested by the user.
2. **No Test Execution**: Do NOT run `pytest`, `docker compose run --rm app pytest`, or `make test` as part of workflows.
3. **Manual Testing by Developer**: Assume verification and validation of code modifications will be performed manually by the developer.
