# SherlockML study guide

SherlockML is an interview-study project because it combines ML evaluation,
data-quality diagnostics, agent orchestration, observable decision-making,
local artifacts, and an operator workflow. The goal is not to memorize the
code; it is to explain the trade-offs behind it.

## The one-minute explanation

> “SherlockML is a deterministic multi-agent ML reliability demo. A synthetic
> fraud model suffers a named incident such as data drift or a feature-pipeline
> bug. A LangGraph supervisor sends the case to specialist agents: a Detective
> gathers evidence, a Statistician quantifies drift, Infra rules out service
> issues, and a Moderator records their debate. A Doctor turns the consensus
> into treatment, an Engineer writes a reviewable local patch, and an Experiment
> agent validates a candidate against explicit gates. The output is an auditable
> local case report—not an automatic deployment.”

## How to explore the repository

Read the system in the same order an incident unfolds:

1. simulator/ — learn the synthetic data-generating process and what each
   incident changes.
2. ml/ — inspect preprocessing, metrics, baseline/candidate evaluation, and
   drift tests.
3. agents/ — follow the typed case state, specialist evidence, and supervisor
   routing.
4. api/main.py — see how a local investigation begins and is serialized.
5. dashboard/app.py — connect the case state to the human workflow.
6. tests/ — learn intended happy-path, temporal, and failed-gate behavior.

## Concepts to explain clearly

### Why simulate incidents instead of use a live feed?

A synthetic simulator gives causal control. You know which distribution,
transform, or configuration changed, so you can test whether the diagnostic
system identifies the right cause. A live feed would be harder to reproduce,
harder to test, and inappropriate for a portfolio project.

### Why multiple agents?

| Role | Question it owns | Example evidence |
| --- | --- | --- |
| Detective | What changed around the incident? | metric deltas, metadata, patch evidence |
| Statistician | Did data behavior materially change? | PSI, KS statistic, missingness, class balance |
| Infra | Could serving explain the signal? | latency and error indicators |
| Moderator | Which claim is best supported? | evidence-linked consensus and dissent |
| Doctor | What is the treatment and risk? | severity, prescription, monitoring advice |
| Engineer | What bounded change addresses it? | local config/code patch artifact |
| Experiment | Did it recover under the evaluation contract? | baseline/candidate metrics and gate result |

The roles are not there for novelty. They separate evidence-gathering concerns,
reduce premature convergence, and leave a review trail.

### Why LangGraph rather than a single sequential function?

A linear function is enough for the happy path, but incidents need stateful
conditional routing. LangGraph represents a case state, fans out evidence
collection, preserves the War Room, and routes a failed candidate back to
analysis. The graph is chosen for explicit state and routing, not because a
graph magically improves model quality.

### Which drift signals matter?

- PSI compares binned feature distributions between reference and current
  samples. It is easy to operationalize, but needs sensible bins and sufficient
  sample size.
- A KS test checks whether continuous samples likely share a distribution. It
  is sensitive at large sample sizes, so pair it with effect size and context.
- Missingness and schema checks catch feature-pipeline incidents that a drift
  score alone may not explain.
- Class balance and label delay prevent invalid conclusions from stale or
  incomplete outcome feedback.

Treat these as evidence, never automatic proof of root cause.

### Why evaluate baseline and candidate in a controlled context?

If models are evaluated on different data or at different times, apparent
improvement can be a sample artifact. The experiment agent uses a controlled,
deterministic comparison so the case report can attribute a difference to the
proposed remediation more credibly.

### What is a validation gate?

A gate is an explicit rule for whether a candidate can be recommended for human
review. Common gates include a quality floor, no unacceptable regression,
reproducible execution, and evidence that the patch fits the diagnosis. A gate
turns “the chart looks better” into a reviewable decision.

### Why is the engineering action bounded?

Autonomous systems should have constrained authority. SherlockML creates a
local patch artifact and, only with SHERLOCKML_AUTOCOMMIT=1, may create a local
Git commit. It does not push remotely or deploy. That boundary makes the demo
more credible than pretending it can safely self-deploy ML changes.

## Interview questions and answer outlines

### How did you keep the demo deterministic?

Named, seed-controlled incidents; synthetic fixtures; shared baseline/candidate
evaluation context; and local artifacts. In production, I would retain sampled,
versioned data plus model, environment, and feature versions.

### How would you avoid false-positive drift alerts?

Combine statistical significance with effect size, minimum sample size, feature
criticality, sustained-window rules, and correlation with performance and
operational signals. A detector feeds a triage policy; it is not a root-cause
label.

### How would you handle delayed labels?

Separate leading indicators from confirmed quality metrics. Feature drift,
prediction distributions, and calibration proxies can trigger investigation
before labels arrive; precision/recall updates later. The report should state
label horizon and confidence explicitly.

### Why keep a War Room?

Data drift, preprocessing bugs, and serving degradation can produce similar
symptoms. Recording claims, evidence, and dissent prevents one fluent narrative
from collapsing uncertainty into an unearned answer.

### How would you make it production ready?

Add authenticated APIs, RBAC, secrets, durable state/event storage, a
feature/model registry, real monitoring sources, data contracts, approval
workflows, isolated patch execution, canary release, rollback, and an agent
evaluation suite. Keep deployment authority outside the diagnostic agents.

### What risks come with an Engineer agent?

It may optimize the wrong metric, change a shared branch, introduce leakage, or
overfit a narrow evaluation. SherlockML limits the demo with fixed scenarios,
explicit gates, local artifacts, and opt-in local commits. Production needs
sandboxing, policy checks, least privilege, review, canaries, and rollback.

### Why is F1 insufficient for fraud?

Fraud is imbalanced and its costs are asymmetric. Precision, recall, PR-AUC,
calibration, latency, alert volume, and human-review capacity can all matter.
The threshold and gates must reflect the cost of false positives and negatives.

### What would you inspect for a feature-pipeline bug?

Schema, dtypes, null rates, ranges, categorical vocabulary, feature ordering,
transform versions, and training/serving parity. A model can serve successfully
while consuming semantically wrong features.

### What does local MLflow add?

It records the baseline/candidate comparison in a familiar local experiment-
tracking interface, while the fallback JSON artifact keeps the demo independent
of a tracking server or external service.

### How would you evaluate the agents?

Build a labeled incident corpus with known root causes. Score evidence retrieval,
diagnosis accuracy, abstention, patch safety, gate decisions, latency, and cost.
Version policies/prompts and run offline regression tests before orchestration
changes.

## Common pitfalls

- Do not say the system “fixed production.” It produced and validated a local
  recommendation against synthetic data.
- Do not treat a drift statistic as proof of causation.
- Do not claim a local Git commit is a review, merge, or deployment.
- Do not bury failed candidates; they prove the gate is meaningful.
- Do not describe remote Git/GitHub, hosted MLflow, or cloud integration unless
  you configure and exercise it.

## High-value extensions

| Extension | Why it matters | Main caveat |
| --- | --- | --- |
| Versioned feature contracts | Catch training/serving mismatch early. | Schema evolution needs discipline. |
| Calibration monitoring | Detect probability-quality degradation. | Requires reliable labels/windows. |
| Shadow or canary evaluation | Test candidates before promotion. | Privacy, power, and rollback matter. |
| Human approval UI | Makes release authority explicit. | Requires permissions and audit retention. |
| Agent evaluation suite | Measures diagnosis and patch safety. | Needs a representative incident corpus. |

## Practice exercise

Run the standard data-drift case twice. First, tell it as a two-minute product
demo. Then explain it as a systems-design interview: state machine, data
contracts, validation gates, safety boundary, and production hardening. Being
able to switch audiences is the most useful skill this project can teach.
