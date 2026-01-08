# uv Cheat Sheet

uv is an extremely fast Python package and project manager, written in Rust. It serves as a unified replacement for `pip`, `pip-tools`, `poetry`, `pyenv`, `virtualenv`, and more.

## Installation and Setup

**Install uv:**
```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip (not recommended for main usage, but possible)
pip install uv
```

**Upgrade uv:**
```bash
uv self update
```

**Shell Autocompletion:**
```bash
# Example for bash
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```

---

## Project Management
uv allows you to manage Python projects with `pyproject.toml` and a cross-platform `uv.lock`.

**Initialize a Project:**
```bash
# Create a new project
uv init my-project

# Initialize in current directory
uv init
```

**Manage Dependencies:**
```bash
# Add a dependency
uv add requests

# Add a specific version
uv add 'requests==2.31.0'

# Add a development dependency
uv add --dev pytest

# Remove a dependency
uv remove requests
```

**Sync Environment:**
```bash
# Sync virtual environment with lockfile (creates .venv if needed)
uv sync

# specific extras
uv sync --all-extras
```

**Run Commands in Project Scope:**
```bash
# Runs command in the project's environment (auto-syncs if needed)
uv run python main.py
uv run pytest
```

---

## Scripts
uv can run standalone scripts and manage their dependencies automatically without manual virtual environments.

**Run a Script:**
```bash
# Run a script (auto-creates ephemeral environment)
uv run script.py
```

**Run with Dependencies (Ad-hoc):**
```bash
# Requires 'rich' and 'requests' available for this run
uv run --with rich --with requests script.py
```

**Inline Script Metadata:**
Standardized way to declare dependencies inside the script (PEP 723).
```python
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint
# ...
```
Run it simply with:
```bash
uv run script.py
```

---

## Tool Management
Execute and install Python command-line tools (similar to `pipx`).

**Run a Tool Ephemerally (`uvx`):**
```bash
# Run 'ruff' without installing it globally
uvx ruff check .

# Run a secure shell with 'httpie'
uvx httpie https://google.com
```

**Install Tools Globally:**
```bash
# Install 'ruff' tool
uv tool install ruff

# List installed tools
uv tool list

# Upgrade installed tools
uv tool upgrade --all
```

---

## Python Version Management
uv can manage Python installations itself, replacing `pyenv`.

**Install Python:**
```bash
# Install latest Python
uv python install

# Install specific versions
uv python install 3.12 3.11
```

**List Available Versions:**
```bash
uv python list
```

**Pin Python Version for Project:**
```bash
# Creates/Updates .python-version file
uv python pin 3.11
```

---

## Pip Interface
uv provides a drop-in replacement for `pip` commands, useful for legacy workflows or CI.

```bash
# Install packages into current environment
uv pip install requests

# specific requirements file
uv pip install -r requirements.txt

# Compile requirements (pip-compile replacement)
uv pip compile requirements.in -o requirements.txt

# Sync environment (pip-sync replacement)
uv pip sync requirements.txt
```
---

## Configuration & Environment

**Common Environment Variables:**
- `UV_CACHE_DIR`: Directory for uv cache.
- `UV_PROJECT_ENVIRONMENT`: Path to the virtual environment (default: `.venv`).
- `UV_PYTHON_DOWNLOADS`: Control python downloads (`auto` [default], `manual`, `never`).
- `UV_NATIVE_TLS`: Set `true` to use system certificate store (important for corporate certs).
- `HTTP_PROXY` / `HTTPS_PROXY`: Standard proxy settings.
- `UV_INDEX_URL` / `UV_EXTRA_INDEX_URL`: Default and extra package indexes.

**pyproject.toml Configuration:**
Example for a data analysis project (not installed as a package).

```toml
[project]
name = "analysis-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pandas",
    "numpy",
    "jupyterlab",
    "polars",
    # GitHub dependency (name only here, details in tool.uv.sources)
    "internal-tools", 
]

[tool.uv]
package = false # Important: Prevents building this project itself as a package

[tool.uv.sources]
# robust reproducibility: pin git tag/commit for github deps
internal-tools = { git = "https://github.com/my-org/internal-tools", tag = "v1.0.2" }
torch = { index = "pytorch" }

# Custom Index (e.g. for CUDA specific pytorch)
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
```

---

## Single-File Scripts & Metadata
PEP 723 allows defining dependencies directly within a script file. This replaces the need for `uv run --no-project --with ...` in many persistent cases.

**Example `script.py`:**
```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests<3",
#     "rich",
# ]
# ///

import requests
from rich.pretty import pprint

print("Running with self-contained dependencies!")
```

**Run it:**
```bash
uv run script.py
```
uv will automatically detect the metadata block, create an ephemeral environment, install `requests` and `rich`, and execute the script.

---

## Useful Flags & Concepts

- **`--no-project`**: Run a command ignoring the current project context.
- **`--with`**: Add ephemeral dependencies to a run command.
- **`--refresh`**: Force refresh of cached data.
- **Cache**: uv caches aggressively. Use `uv cache clean` to clear.
