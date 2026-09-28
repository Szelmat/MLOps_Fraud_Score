# MLOps Fraud Score

An MLOps project for scoring card transactions for fraud. It covers data versioning with DVC, a FastAPI scoring service, tests with coverage enforced by a pre-commit hook, and a Python environment managed with [uv](https://docs.astral.sh/uv/). Experiment tracking with MLflow is planned.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git

## Setup

```bash
git clone git@github.com:Szelmat/MLOps_Fraud_Score.git
cd MLOps_Fraud_Score

# Create .venv and install the dependencies from pyproject.toml / uv.lock
uv sync

# Install the git pre-commit hook (once per clone)
uv run pre-commit install

# Download the data from the DVC remote (needs DagsHub credentials, see "DVC remote")
uv run dvc pull
```

You don't need to activate the virtual environment. Prefix commands with `uv run`:

```bash
uv run python <script>.py
uv run dvc <command>
uv run pytest
```

To add a dependency:

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # development-only dependency (tests, linters, ...)
```

Commit both `pyproject.toml` and `uv.lock` after any change to the dependencies.

## Running the API

The scoring service is a [FastAPI](https://fastapi.tiangolo.com/) app defined in `main.py`.

```bash
uv run fastapi dev main.py
```

The server listens on http://127.0.0.1:8000, and the interactive docs are at http://127.0.0.1:8000/docs.

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/` | `{"project": "MLOps Fraud Score"}` |

## Testing

Tests live in `tests/` and use pytest with FastAPI's `TestClient`.

```bash
uv run pytest                                      # run the tests
uv run pytest --cov --cov-report=term-missing      # with a coverage report
```

`pyproject.toml` adds the repository root to the import path, so tests can use `from main import app`. Coverage measures all project code except `tests/` and `.venv/`.

### Pre-commit hook

`.pre-commit-config.yaml` runs the full test suite with coverage before every commit that includes Python files. If a test fails, the commit is blocked. To run the hook manually on all files:

```bash
uv run pre-commit run --all-files
```

## Data

The raw data is versioned with [DVC](https://dvc.org/). Git stores only the small `.dvc` pointer files. DVC stores the data itself. DVC is a dev dependency, so `uv sync` installs it into `.venv`; you don't need a global install.

| File | Rows | Description |
|------|------|-------------|
| `data/raw/fraud_5000.csv` | 5,000 | Labelled card transactions (target: `Fraud_Label`) |

The dataset has 21 columns:

- **Identifiers:** `Transaction_ID`, `User_ID`
- **Transaction:** `Transaction_Amount`, `Transaction_Type`, `Timestamp`, `Merchant_Category`, `Location`, `Transaction_Distance`, `Is_Weekend`
- **Account and card:** `Account_Balance`, `Card_Type`, `Card_Age`, `Device_Type`, `Authentication_Method`
- **Behavioural history:** `Daily_Transaction_Count`, `Avg_Transaction_Amount_7d`, `Failed_Transaction_Count_7d`, `Previous_Fraudulent_Activity`, `IP_Address_Flag`
- **Existing score:** `Risk_Score`
- **Target:** `Fraud_Label` (0 = legitimate, 1 = fraud)

### DVC remote

The remote is the DagsHub repository's DVC storage, configured in `.dvc/config`:

```
https://dagshub.com/Szelmat/MLOps_Fraud_Score.dvc
```

The repository is private, so both `pull` and `push` need your DagsHub credentials. Without them, `dvc pull` fails with a misleading "missing files" error. Add them to the local config. The `--local` flag writes them to `.dvc/config.local`, which git ignores:

```bash
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <dagshub-username>
uv run dvc remote modify origin --local password <dagshub-token>
```

To fetch the data after cloning:

```bash
uv run dvc pull
```

To track a new or changed dataset:

```bash
uv run dvc add data/raw/<file>.csv
git add data/raw/<file>.csv.dvc data/raw/.gitignore
git commit -m "data: ..."
uv run dvc push
```

## Project structure

```
.
├── data/
│   └── raw/                  # DVC-tracked raw datasets
├── src/                      # Source code (empty for now)
├── tests/
│   └── test_api.py           # API tests
├── main.py                   # FastAPI app
├── .dvc/                     # DVC configuration and remote
├── .pre-commit-config.yaml   # Pre-commit hook: pytest with coverage
├── pyproject.toml            # Project metadata, dependencies, pytest and coverage config
└── uv.lock                   # Locked dependency versions
```

Git ignores the following local outputs: `.venv/`, `mlruns/`, `mlartifacts/`, `*.db`, `logs/`, `.env`, and coverage or pytest caches.

## Environment variables

Put local secrets and settings, such as MLflow tracking URIs or DVC remote credentials, in a `.env` file at the repository root. Git ignores this file, so never commit it.
