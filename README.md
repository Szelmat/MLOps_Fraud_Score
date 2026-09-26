# MLOps Fraud Score

An MLOps project for scoring card transactions for fraud. It covers data versioning with DVC, experiment tracking with MLflow, and a Python environment managed with [uv](https://docs.astral.sh/uv/).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git

## Setup

```bash
git clone <repo-url>
cd mlops_fraud_score

# Create .venv and install the dependencies from pyproject.toml / uv.lock
uv sync
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

## Data

The raw data is versioned with [DVC](https://dvc.org/). Git stores only the small `.dvc` pointer files. DVC stores the data itself.

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

To fetch the data after cloning (a DVC remote must be configured):

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
│   └── raw/            # DVC-tracked raw datasets
├── src/                # Source code
├── .dvc/               # DVC configuration
├── pyproject.toml      # Project metadata and dependencies (uv)
└── uv.lock             # Locked dependency versions
```

Git ignores the following local outputs: `.venv/`, `mlruns/`, `mlartifacts/`, `*.db`, `logs/`, `.env`, and coverage or pytest caches.

## Environment variables

Put local secrets and settings, such as MLflow tracking URIs or DVC remote credentials, in a `.env` file at the repository root. Git ignores this file, so never commit it.
