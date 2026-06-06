# Contributing to coreai-torch

The coreai-torch source code is open source under the
[BSD 3-Clause license](LICENSE). We welcome contributions within a defined
scope — please read this document before opening a pull request or issue.

## How you can help

- Reporting bugs and conversion failures with clear, reproducible steps
- Reporting unsupported ops or layer types (see
  [issue templates](../../issues/new/choose))
- Improving documentation
- Adding or improving tests for existing conversion paths
- Bug fixes for existing conversion behavior
- Minor enhancements to existing extension points

## Commitment to quality

We maintain high standards across the project. Contributions are evaluated on:

- Technical quality and correctness
- Adherence to existing conventions and architectural patterns
- Your demonstrated understanding of the implementation and its implications
- Clarity of communication
- Maintainability

### Productivity and AI tools

You may use productivity tools, including AI-assisted coding tools, to help you
work more efficiently. However, you remain fully accountable for all submitted
code, issues, pull requests, and comments. Maintainers expect you to thoroughly
review, understand, and validate everything you submit — and be able to explain
your contributions in detail. AI-generated content must meet the same rigorous
standards as human-written code.

**Important:** Low-quality pull requests and issues, including those that appear
to be generated without understanding, validation, or genuine human review, will
be treated as spam and moderated accordingly.

## Contribution scope

We keep the API surface intentionally limited to ensure reliability and
maintainability across PyTorch releases.

**In scope:**

- Bug fixes and conversion failures
- Support for missing ops or layer types via existing conversion mechanisms
  (e.g. adding an entry to the ATen-to-Core resolver, fixing a numerical
  mismatch in an existing lowering)
- Minor enhancements to existing extension points
- Documentation improvements
- Test additions

**Not in scope at this time:**

- Major new conversion features or architectural changes
- Changes to the core API surface

## Setting up your environment

A detailed setup guide is in [README.md](README.md#getting-started).

## Before you open a pull request

For anything beyond a straightforward bug fix, open a
[Contribution Proposal](../../issues/new?template=contribution_proposal.yml)
first. Describe what you want to contribute and wait for a response before
writing code.

## Submitting issues

Before opening an issue:

- Search [existing issues](../../issues) to avoid duplicates
- Use the appropriate [issue template](../../issues/new/choose)
- For conversion failures, include a minimal reproducible model or code snippet
- For security issues, follow the
  [Apple Open Source security disclosure process](https://github.com/apple/.github/blob/main/SECURITY.md)

## Submitting pull requests

Please review the [PR template](.github/PULL_REQUEST_TEMPLATE.md) before
submitting. Small, focused pull requests are much more likely to be reviewed
and merged than large ones.

### Testing

All contributions require tests.

- **Before submitting:** Ensure all existing tests pass locally. See
  [README.md](README.md) for instructions.
- **New functionality:** New features and bug fixes require corresponding
  automated tests.
- **Numerical accuracy:** For changes that affect model output, include a test
  that validates the result is correct.

### Style guide

This project uses [`ruff`](https://docs.astral.sh/ruff/) for linting and
formatting, configured in `pyproject.toml`. Run before submitting:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Type annotations are required for all new public APIs.

Commit messages should be clear and concise, describing what changed and why.

## Response time

This project is maintained best-effort. We aim to triage issues and pull
requests as bandwidth allows; please be patient if a response takes some time.

## Code of conduct

This project follows the
[Apple Open Source Code of Conduct](https://github.com/apple/.github/blob/main/CODE_OF_CONDUCT.md).
All community members are expected to adhere to these guidelines.
