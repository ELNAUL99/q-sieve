# Q-SIEVE demo

An offline analyst-workflow prototype based on the supplied **Q-SIEVE — Team Onboarding** document (October 1, 2026). All accounts, transaction windows, labels and benchmark data in this package are synthetic.

## Open the demo

Open `index.html` in a modern browser. No server, account, API token or internet connection is required. The single HTML file includes its model data.

The Python model computes quantum-kernel scores ahead of time. The browser performs queue filtering, policy routing and analyst actions locally; it does not connect to quantum hardware.

## Five-minute walkthrough

1. Inspect the ranked queue. Select a prioritized window and compare its money flow and behavioural features with its similar synthetic reference cases.
2. Select an uncertain case. Explain why Q-SIEVE retains it for human review.
3. Select a clear-eligible case. Explicitly enable demo-only auto-clear, then process eligible alerts and observe the remaining queue and audit entries.
4. Adjust the policy thresholds. Watch cases move between routing categories. This is a policy illustration, not a validated safety threshold.
5. Escalate a case and export the session audit as JSON. Visit evaluation to compare measured synthetic-model results.

Auto-clear starts disabled. At the fixed demonstration cutoff of 0.20, 10 of 103 eligible held-out windows are synthetically suspicious (9.7% error among eligible cases). This policy is not safe for production.

Use Reset to start again. Local audit history is resettable and is not a tamper-evident production audit system.

## Model and evidence

- The unit of analysis is one account and one flagged window.
- The examples cover inbound/outbound and fragmented/near-threshold activity. The threshold band is synthetic, not a legal reporting rule.
- An eight-feature representation feeds an eight-qubit, two-repetition ZZ statevector simulation.
- Pairwise squared state overlap forms a fidelity kernel. A classical support-vector classifier consumes that kernel.
- Classical baseline results use the same feature representation. Inspect `model-data.json` for the exact split, metrics, method, policy and limitations.
- Precedents show similarity to synthetic training examples. Similarity does not establish that a case is legitimate or laundering.
- Evidence cues describe observable features, not causal model attribution.

This prototype does not establish quantum advantage, real-world AML performance, safe automatic case closure, or a production-ready governance system. It does not ingest IBM AML or SAML-D, call live quantum hardware, reconstruct laundering networks, or use real customer data.

## Real-data build (`index-real.html`)

`make_real_demo.py` regenerates the model data from the **real** Q-SIEVE pipeline in
the parent repo — real IBM AML HI-Small windows, angle-kernel SVC scores, real
precedents and transactions, the tuned XGBoost/RBF/quantum benchmark, and the
three-QPU (IBM / IQM / VTT) hardware report. It writes `model-data.real.json`, which
`frontend-template.real.html` embeds to produce the standalone **`index-real.html`**.
Same UI, real numbers that match the pitch deck. The original synthetic `index.html`
is left untouched.

```
../.venv/Scripts/python.exe make_real_demo.py      # -> model-data.real.json (needs ../data/X.csv + raw)
# then re-embed into index-real.html (see the bundling snippet in make_real_demo.py's header)
```

## Measured synthetic results

The model uses 240 training windows, 120 calibration windows and 160 test windows. Each split contains 30% generator-labelled suspicious windows. Twelve examples in the dashboard are selected by score from a separate 120-window demonstration pool; they are not the test set and should not be used to infer prevalence or performance.

| Model | Average precision | Test recall at calibration-selected cutoff | Legitimate test alerts below cutoff |
| --- | ---: | ---: | ---: |
| ZZ fidelity + SVC | 0.869 | 83.3% | 77.7% |
| RBF SVC | 0.930 | 91.7% | 88.4% |
| Histogram gradient boosting | 0.943 | 93.8% | 81.2% |

The cutoff targets at least 90% recall on calibration data. Test recall differs, so this is not a comparison at identical realized recall. The 0.20 auto-clear demonstration cutoff is a separate policy. Results use one seed and fixed hyperparameters; full baseline tuning, repeated trials, uncertainty estimates and the planned positive-label ablation study remain future work. Histogram gradient boosting is not XGBoost.

## Reproduce

Use Python 3.11 or newer in a virtual environment. Install `requirements.txt`, then run `python model.py`. The model is seeded and writes its generated data and reproducibility details. After regenerating model output, run `python bundle_demo.py` in this directory to rebuild the HTML's embedded data. Run `python model.py --test` for the independent circuit and policy checks.

## Reference material

- Supplied Q-SIEVE onboarding document, pages 1–7: scope, proposed workflow, experimental plan.
- Havlíček et al., *Supervised learning with quantum enhanced feature spaces*, Nature 567 (2019), https://arxiv.org/abs/1804.11326 — method background, not evidence for this demo's efficacy.
- Cardaioli et al., *FD4QC*, https://arxiv.org/abs/2507.19402 — related comparison. Its abstract reports stronger classical tree performance and a QSVM false-positive/recall tradeoff; it does not establish a Q-SIEVE advantage.

The accompanying presentation includes speaker notes and an editable benchmark chart.

## Verification limits

Model checks and application workflow checks passed, including policy guards and audit persistence. The build environment blocked browser installation, so browser rendering and responsive layout were not visually verified.
