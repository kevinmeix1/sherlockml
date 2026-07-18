# SherlockML demo script

This script is optimized for a five-to-seven-minute hackathon or interview
demo. It demonstrates a full recovery loop without pretending a synthetic
laptop application is a production incident platform.

## Before the audience arrives

1. Create and activate the virtual environment, then run `make install`.
2. In terminal one, run `make run-api`.
3. In terminal two, run `make run-ui`.
4. Open <http://127.0.0.1:8502> and keep the terminal logs visible but tidy.
5. Run `make check` once after the final change. Keep the result available if a
   judge asks how the demo was verified.

Optional API smoke check:

```bash
curl http://127.0.0.1:8788/health
```

## Opening: 20 seconds

> “ML systems often fail silently: predictions still return, but a shift in
> data, preprocessing, or training quality can make them unreliable. SherlockML
> is a deterministic multi-agent incident-response demo. Instead of asking a
> chatbot what happened, I trigger a model incident and show an AI engineering
> team gather evidence, debate it, test a remedy, and produce an approval report.”

## Act 1 — Show a healthy patient: 30 seconds

1. Point to the **Model Health** view.
2. Explain that the model and transactions are synthetic and deterministic.
3. Call out that the dashboard shows a healthy baseline before an incident.

Say:

> “This baseline is our control. The result is reproducible because the demo
> uses a fixed seed and synthetic fixtures, not a live fraud feed.”

## Act 2 — Create an incident: 30 seconds

1. Select **Data Drift**. It is usually the clearest first demo.
2. Briefly state that the incident changes the production-like transaction
   distribution while leaving the original model unchanged.
3. Click **Investigate Incident**.

Say:

> “The model hasn’t crashed; it has become less trustworthy. That is exactly
> the kind of reliability incident that can be easy to miss in production.”

## Act 3 — Let the detective team work: 90 seconds

As the Evidence Timeline populates, narrate each role:

| Dashboard area | What to point out | The engineering point |
| --- | --- | --- |
| Case file | Baseline versus incident metrics and ranked suspects. | We begin with evidence rather than a guessed fix. |
| Statistical evidence | Feature-shift and data-health signals. | Drift needs a measured signal, not visual intuition alone. |
| Infra finding | Simulated latency/error context. | Not every model degradation is caused by the model. |
| War Room | Competing claims and the moderator’s evidence-backed consensus. | The system records dissent and rationale instead of fabricating certainty. |
| Doctor | Severity and treatment plan. | Diagnosis is separated from the engineering implementation. |

Suggested narration:

> “The Detective sees the degraded outcome and recent change evidence. The
> Statistician measures whether the distribution movement is meaningful. Infra
> checks whether service behavior explains the symptom. The moderator only
> resolves the case after these claims are compared.”

## Act 4 — Show the engineering action: 75 seconds

1. Open the remediation / patch section.
2. Explain that the Engineer applies a controlled repair to the synthetic local
   pipeline contract and creates a reviewable change artifact.
3. Call out that nothing is pushed remotely or deployed automatically.
4. Show the candidate experiment next to the baseline.

Say:

> “This is intentionally bounded autonomy. SherlockML applies only a
> synthetic local contract repair, preserves the diff, and runs it against the
> deterministic evaluation set. Its decision is approval for human review—not
> permission to deploy.”

If the optional local Git path is enabled, show the generated local patch or
commit evidence. Do not claim a GitHub push or production release; both are
outside the demo.

## Act 5 — Validate recovery: 60 seconds

1. Show the before/after experiment comparison.
2. Explain that the candidate is checked against explicit floor and improvement
   gates, not just a single attractive metric.
3. Read the recovery decision and report summary.

Say:

> “The important outcome is not merely a higher score. SherlockML asks whether
> the candidate improves the relevant evaluation, clears the defined quality
> floor, and fits the diagnosed cause. If it does not, the case is not approved.”

## Close: 20 seconds

> “SherlockML turns ML reliability from a black box into a visible case file:
> incident, evidence, disagreement, treatment, experiment, validation, and
> report. The whole story runs deterministically on a laptop, so every decision
> can be inspected and replayed.”

## Suggested audience questions

- **“Is this using real fraud data?”** No. Every dataset and metric in the demo
  is synthetic and local.
- **“Does it deploy the change?”** No. It repairs only a synthetic local
  contract and produces a recommendation for human review.
- **“What happens if the fix fails?”** The validation gate rejects it or marks
  it for review, preserving the case artifact instead of reporting a recovery.
- **“Does it need an LLM API?”** No. The default workflow is deterministic and
  laptop-runnable. A hosted model integration would be a separate extension.

## Failure and recovery drill

Use this when a judge asks how the system behaves when the candidate is not
good enough.

1. Choose a scenario that yields a failed validation path, or use the dashboard
   control that keeps the faulty configuration in place if available.
2. Run the investigation and point to the failed gate / needs-review outcome.
3. Explain that the case goes back to evidence and treatment rather than
   declaring success based on one metric.
4. Reset the deterministic local state with the dashboard reset control. It
   restores the synthetic local pipeline contract before the next rehearsal.
5. Rerun the default Data Drift scenario and show the standard, reproducible
   recovery path.

This is a meaningful recovery drill because it exercises rejection and reset,
not just the happy path.

## A compact 60-second version

> “I trigger a deterministic fraud-model incident. SherlockML’s Detective,
> Statistician, and Infra agents gather different evidence; the War Room records
> their disagreement before the Doctor proposes treatment. The Engineer applies
> a controlled synthetic local repair, the Experiment agent compares it against the
> baseline, and explicit validation gates decide whether it is approved for
> human review. It is a complete ML reliability case file, not a chatbot and
> not an automatic deployment system.”
