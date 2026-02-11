# btlight
The backtesting light (**btlight**) package is a lightweight library for equity portfolio optimization and backtesting produced for educational purposes.

## Quick start (uv)

### 1) Install uv
- **Linux/macOS**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
- **Windows (PowerShell)**
    ```powershell
    iwr https://astral.sh/uv/install.ps1 -useb | iex
    ```

### 2) Create a venv
```bash
uv venv
```

### 3) Sync dependencies
```bash
uv sync
```

### 5) Add a package
Below an example of how to add a package automatically to the package (directly into the `pyprocet.toml`)
```bash
uv add xgboost
```

