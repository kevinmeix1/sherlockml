# SherlockML operator runbook

This runbook is for a local developer, interview rehearsal, or hackathon demo.
It intentionally assumes a single laptop and synthetic data.

## Prerequisites

- Python 3.10 or newer
- `pip` and a virtual environment
- Docker Desktop only if using the optional container path

No cloud account, API key, database, model registry, or customer dataset is
required for the default workflow.

## First-time setup

```bash
cd /Users/kaiwenmei/Desktop/SherlockML
python3 -m venv .venv
source .venv/bin/activate
make install
make check
```

`make install` expects the repository’s `pyproject.toml` to define its runtime
and developer dependencies. It installs the project in editable mode so code
changes are picked up by the local servers.

## Start locally

Use two terminals with the same virtual environment activated.

**Terminal 1 — API**

```bash
make run-api
```

The FastAPI service binds to <http://127.0.0.1:8788>. Verify it with:

```bash
curl http://127.0.0.1:8788/health
```

**Terminal 2 — dashboard**

```bash
make run-ui
```

The Streamlit control room binds to <http://127.0.0.1:8502>.

To change local ports without editing source:

```bash
make run-api API_PORT=9000
make run-ui UI_PORT=9001
```

## Local API interface

The FastAPI documentation is available at <http://127.0.0.1:8788/docs> while
the server is running. The small public demo surface is:

| Method | Endpoint | Effect |
| --- | --- | --- |
| GET | /health | Return local liveness and supported scenarios. |
| GET | /api/incidents | List safe, deterministic incident names. |
| POST | /api/incidents/{incident_type}/investigate | Run a full local case. |
| POST | /api/reset | Restore the baseline synthetic pipeline contract. |

For example:

```bash
curl -X POST http://127.0.0.1:8788/api/incidents/data_drift/investigate
```

## Standard investigation

1. Open the dashboard.
2. Confirm the baseline health display is present.
3. Select one incident scenario.
4. Start an investigation.
5. Inspect the case file, evidence, War Room, treatment, experiment, and
   validation decision.
6. Locate the generated report under `artifacts/` if the runtime exposes it.

Each fresh default run should be deterministic for the configured seed. If you
need a clean demonstration state, use the dashboard reset control; it calls the
same reset path that restores the synthetic pipeline contract. You can also run:

```bash
python -c "from api.main import reset_demo; print(reset_demo())"
```

Generated outputs are intentionally separate from the source files.

## Verification commands

| Command | Purpose |
| --- | --- |
| `make test` | Run the deterministic test suite. |
| `make check` | Run lint, type checks, and tests. Use before a demo or commit. |
| `make format` | Apply the configured formatter. Review its diff before committing. |
| `curl http://127.0.0.1:8788/health` | Verify the API process is alive. |

## Local MLflow tracking

The default case artifacts are local JSON/Markdown files. Each investigation
also attempts to record its baseline/candidate metrics to the repository-local
MLflow SQLite database.

When `mlflow` is installed, the tracker uses
`artifacts/mlflow_tracking.db`; it falls back to its JSON artifact if MLflow is
unavailable. There is no configured hosted MLflow server, remote artifact store,
or team-shared tracking backend.

If you enable a local tracking URI, keep it inside the repository or another
explicitly chosen local path. Do not put secrets or customer data in the run
metadata.

## Optional Git evidence

SherlockML applies a controlled repair only to its synthetic local pipeline
contract and writes a patch artifact as part of the investigation. The artifact
is the default evidence; it does not require a Git repository. Reset the demo
state before a fresh rehearsal if you need to restore the original contract.

An actual **local** Git commit is opt-in via:

```bash
SHERLOCKML_AUTOCOMMIT=1 make run-api
```

Use this only inside a disposable or deliberately chosen local repository. It
does not push, open a pull request, contact GitHub, or deploy anything. Remote
Git/GitHub workflows remain design-only.

## Docker Compose path

Docker is optional. Build and run the FastAPI service only:

```bash
docker compose up --build
```

To include the Streamlit UI, opt in to the `dashboard` profile:

```bash
docker compose --profile dashboard up --build
```

The containers expose the same host ports: API `8788`, dashboard `8502`. They
mount `artifacts/` and `runtime/` locally so case evidence remains visible after
the containers stop. Stop the stack without deleting local folders:

```bash
docker compose down
```

## Troubleshooting

| Symptom | Likely cause | Safe recovery |
| --- | --- | --- |
| `make install` fails | Wrong Python version or stale virtual environment. | Confirm `python --version`, recreate `.venv`, then rerun `make install`. |
| API port is busy | Another local service owns 8788. | Run `make run-api API_PORT=9000` and update any dashboard API setting if applicable. |
| Dashboard cannot reach API | API is stopped or port configuration differs. | Start API first, test `/health`, then restart dashboard. |
| Docker dashboard waits forever | API health check is not passing. | Run `docker compose logs api`, then test the host API endpoint. |
| Run looks different from expected | Local generated state or a changed seed/configuration. | Restart services; inspect local artifacts and the active scenario before rerunning. |
| Candidate is not approved | Validation correctly failed a gate. | Treat it as a case result, inspect evidence, then use the failure/recovery drill. |
| MLflow tracking falls back | The local tracking package or database is unavailable. | Use JSON case artifacts; do not block the demo on tracking UI. |

## What not to do

- Do not expose the local API or dashboard to the public internet.
- Do not use real customer, financial, or employer data as a substitute for the
  synthetic fixtures.
- Do not set `SHERLOCKML_AUTOCOMMIT=1` in an unrelated working tree.
- Do not describe a local candidate approval as a production deployment.
- Do not claim remote Git, GitHub, CI/CD, or hosted MLflow integration unless
  you actually configure and exercise it separately.
