# Ruff Cheat Sheet (with uv)

Modern Python formatting and linting using `ruff` and `uv`.

## 1. Installation & Usage with uv

Since `uv` and `ruff` are both from Astral, they work together seamlessly.

### Add Ruff to your project
Add `ruff` as a development dependency:
```bash
uv add --dev ruff
```

### Basic Commands
Run ruff through `uv` to ensure you use the project's pinned version.

- **Lint (check errors):**
  ```bash
  uv run ruff check .
  ```
- **Lint (fix errors):**
  ```bash
  uv run ruff check --fix .
  ```
- **Format (check):**
  ```bash
  uv run ruff format --check .
  ```
- **Format (auto-format):**
  ```bash
  uv run ruff format .
  ```

### Using without installation (`uvx`)
You can run ruff instantly without installing it in the project:
```bash
uvx ruff check .
```

---

## 2. Configuration (`pyproject.toml`)

Centralize all configuration in `pyproject.toml`.

```toml
[project]
# ... standard project metadata ...

[tool.ruff]
# Target Python version (adjust to your project's needs)
target-version = "py310"
# Maximum line length (Ruff defaults to 88, similar to Black)
line-length = 88

[tool.ruff.lint]
# Enable rules:
# E: Pycodestyle errors
# F: Pyflakes
# I: isort (Import sorting)
# B: Flake8-bugbear (Potential bugs)
# UP: Pyupgrade (Upgrade syntax to newer Python versions)
# N: Pep8-naming
select = ["E", "F", "I", "B", "UP", "N"]

# Ignore specific rules if needed
ignore = []

# Allow unused variables when underscore-prefixed.
dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"

[tool.ruff.lint.isort]
# Organize imports settings
known-first-party = ["my_project"]

[tool.ruff.format]
# Like Black, use double quotes for strings.
quote-style = "double"
# Indent with spaces, rather than tabs.
indent-style = "space"
# Respect magic trailing commas.
skip-magic-trailing-comma = false
# Automatically detect the appropriate line ending.
line-ending = "auto"
```

---

## 3. VSCode Configuration

To use Ruff effectively in VSCode, install the **Ruff** extension (Extension ID: `astral-sh.ruff` or formerly `charliermarsh.ruff`).

Add the following to your `.vscode/settings.json` (workspace settings) or global `settings.json`:

```json
{
  "[python]": {
    "editor.defaultFormatter": "astral-sh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  // If you are using a custom venv or uv, Ruff usually finds it automatically.
  // But you can explicitly set the python path if needed (rarely required with uv python finding):
  // "ruff.interpreter": ["/path/to/.venv/bin/python"]
}
```
*Note: Ensure "Editor: Default Formatter" is set to Ruff for Python files.*

---

## 4. Advanced Formatting & Linting Usage

### Type Hints (mypy / flake8-type-checking equivalent)
Ruff doesn't do full type checking (use `mypy` or `pyright` for that), but it can enforce type hint strictness via `flake8-type-checking` rules (`TCH`).

**Configuration:**
```toml
[tool.ruff.lint]
# Add "TCH" to select
select = [..., "TCH"]
```

**What it does:**
It ensures imports used ONLY for type hinting are moved to a `TYPE_CHECKING` block to avoid circular imports and runtime costs.

**Example Code:**

*Before (Ruff flags this):*
```python
import pandas as pd  # Used only for type hint

def process_data(df: pd.DataFrame) -> None:
    pass
```

*After (`ruff check --fix`):*
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

def process_data(df: "pd.DataFrame") -> None:
    pass
```

### Docstrings (`pydocstyle` equivalent)
Enforce docstring standards (e.g., Google, NumPy, or PEP 257 style).

**Configuration:**
```toml
[tool.ruff.lint]
# Add "D" to select
select = [..., "D"]

[tool.ruff.lint.pydocstyle]
convention = "google"  # or "numpy", "pep257"
```

**Example Code:**

*Bad Docstring (Ruff flags missing args):*
```python
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b
```

*Good Docstring (Google Style):*
```python
def add(a: int, b: int) -> int:
    """Adds two numbers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of the two integers.
    """
    return a + b
```

### Import Sorting (isort equivalent)
Ruff has a built-in sort (Rule `I`). It sorts standard library, third-party, and local imports automatically.

**Configuration:**
Already included if you add `"I"` to `select`.

**Example:**
*Before:*
```python
import sys
from my_lib import foo
import os
import requests
```

*After:*
```python
import os
import sys

import requests

from my_lib import foo
```

### Ignoring Rules Inline
You can ignore specific rules for a line using `# noqa`.

```python
x = 1  # noqa: F841 (Variables assigned but never used)
```
