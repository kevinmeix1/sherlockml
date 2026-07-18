# SherlockML implementation matrix

This matrix distinguishes what the repository is intended to demonstrate on a
laptop from integrations that are optional or intentionally only designed. It
is a guardrail against overstating a hackathon prototype.

## Legend

| Status | Meaning |
| --- | --- |
| Local | Part of the deterministic, offline-first demo and expected to run after dependencies are installed. |
| Optional | A real local enhancement enabled when its dependency or setting is available; the core demo still works without it. |
| Design-only | Documented future architecture or boundary. It is not configured, exercised, or represented as a live integration. |

## Capability matrix

| Capability | Status | What is actually demonstrated | Boundary / evidence |
| --- | --- | --- | --- |
| Synthetic fraud-like data | Local | Deterministic generated transactions and labels. | No customer, employer, or live financial data. |
| Fraud-model baseline/candidate evaluation | Local | Controlled before/after evaluation for a named incident. | Results are synthetic demo observations, not production claims. |
| Incident simulator | Local | Data drift, feature-pipeline bug, and model-regression scenarios. | Scenarios are bounded and seed-controlled. |
| Drift and data-health analysis | Local | Statistical and schema-style evidence used in the case file. | Signals support diagnosis; they do not prove causation alone. |
| LangGraph supervisor and specialist roles | Local | Stateful Detective, Statistician, Infra, Moderator, Doctor, Engineer, and Experiment workflow. | Default logic is deterministic and needs no hosted LLM. |
| War Room transcript | Local | Evidence-linked claims, disagreement, and consensus saved with a run. | It is a local case artifact, not a Slack integration. |
| FastAPI service | Local | Local investigation interface and health endpoint on port 8788. | No authentication, multi-tenancy, or public exposure model. |
| Streamlit control room | Local | Local visual walkthrough on port 8502. | It is a demo UI, not a production operations console. |
| Local report/artifact storage | Local | JSON/Markdown evidence, patch, and recovery artifacts. | Generated artifacts remain on the operator’s machine. |
| Controlled engineering repair | Local | Synthetic local pipeline-contract repair plus reviewable diff/config artifact. | Approval means ready for review, never deployed. |
| Local Git commit | Optional | A real local commit can be requested with SHERLOCKML_AUTOCOMMIT=1. | Disabled by default; never pushes or opens a PR. |
| Git diff/history evidence | Optional | Local repository context may be included when available. | No assumption that the demo runs inside a Git checkout. |
| MLflow experiment tracking | Local | Each investigation records baseline/candidate metrics in a local SQLite-backed MLflow store. | JSON artifacts remain the explicit fallback; no tracking server is required. |
| Docker packaging | Optional | FastAPI runs in the supplied image; dashboard can be added through a Compose profile. | Local Compose only; no image registry or runtime hardening claim. |
| Hosted LLM/provider | Design-only | None required by the deterministic demo. | Adding one requires credentials, safety policy, cost controls, and agent evaluation. |
| Remote Git/GitHub workflow | Design-only | None. | No remote push, PR, repository provisioning, or CI status is claimed. |
| Hosted MLflow / artifact store | Design-only | None. | No server, bucket, credentials, or access control is configured. |
| Production data sources | Design-only | None. | Requires governance, privacy review, contracts, and retention policy. |
| Production deployment / rollback | Design-only | None. | Candidate approval is not a deployment decision. |
| Auth, RBAC, secrets, audit retention | Design-only | None. | Needed before any multi-user or production use. |
| Kubernetes, CI/CD, monitoring platform | Design-only | None. | Mentioned only as possible future hardening. |

## What a reviewer can verify locally

~~~
make install
make check
make run-api
make run-ui
~~~

Then trigger a named scenario through the dashboard and inspect the resulting
timeline, evidence, War Room, patch artifact, experiment comparison, validation
decision, and report. Those local observables are the project’s primary proof.

## How to describe the integration paths accurately

- MLflow: “The demo records runs to a local SQLite-backed MLflow store and
  preserves a JSON artifact as a portable fallback. We do not run a shared
  MLflow service.”
- Git: “The system applies a controlled synthetic local repair and preserves a
  reviewable diff. A local Git commit is opt-in via an environment variable;
  there is no remote push or GitHub action.”
- Docker: “Compose packages the same local demo as containers. It is a
  convenience for reproducibility, not evidence of production readiness.”

## Promotion checklist for a future real deployment

Before moving any capability from design-only to operational, add explicit
evidence for data governance, authentication and authorization, secret
management, model/feature versioning, human approval, isolated code execution,
rollback, observability, incident ownership, and agent evaluation. Update this
matrix only after those paths are configured and exercised.
