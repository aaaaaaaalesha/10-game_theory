<h1 align="center"> Game Theory. 10 semester  </h1>

<p align="center">
  <a href="https://badgen.net/badge/python/3.10 | 3.11/blue">
      <img alt="Python 3.10 | 3.11" src="https://badgen.net/badge/python/3.10 | 3.11/blue" >
  </a>
  <a href="https://github.com/astral-sh/ruff">
      <img alt="Code style: Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json" >
  </a>
  <a href="https://github.com/aaaaaaaalesha/10-game_theory/actions/workflows/github-code-scanning/codeql">
      <img alt="CodeQL" src="https://github.com/aaaaaaaalesha/10-game_theory/actions/workflows/github-code-scanning/codeql/badge.svg" >
  </a>
  <a href="https://wakatime.com/projects/10-game_theory">
      <img alt="Wakatime" src="https://wakatime.com/badge/user/0e21c3c1-25e0-47ee-9c0f-77ef4b6b71e2/project/018c9823-7706-4998-af38-9a8734cf127a.svg" >
  </a>
</p>

## Quick Start

1. Installing **Poetry** to manage dependencies.

```bash
# Install Poetry.
pip3 install poetry==1.7.1
# Verify that Poetry is working.
poetry --version
```

> [!WARNING]
> If you receive an error: `"Command not found.", add Poetry to your user Path variable`.

```bash
# For *-nix systems.
export PATH="$HOME/.local/bin:$PATH"
```

```powershell
# For Windows.
[System.Environment]::SetEnvironmentVariable('path', $env:USERPROFILE + "\AppData\Roaming\Python\Scripts;" + [System.Environment]::GetEnvironmentVariable('path', "User"),"User")
```

2. All dependencies have already locked in `poetry.lock`, you just need to install it.

- If you need the complete environment with dev-tools.

```bash
poetry install
# Install pre-commit hook to .git.
pre-commit install
```

- If you need minimal production dependencies without dev-tools.

```bash
poetry install --without dev
```

3. And now you're ready to run jupyter notebook using env based on poetry venv dependencies.

```bash
poetry run jupyter notebook
# OR just activate your created virtual environment like this.
poetry shell
```