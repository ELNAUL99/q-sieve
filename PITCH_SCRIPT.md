# Q-SIEVE — 3-minute pitch script & Q&A prep

> Out of date. This draft opens with the bank queue and still says the tree wins 0.19 to 0.12. Speak from `PITCH_TALK.md`.

Deck: `QSIEVE_pitch.pptx` (8 slides, same script lives in each slide's speaker notes).
**Hard rule: 3:00 max, sound at 0:00.** Speak slow, pause between slides, use the mic.

## The script (≈430 words ≈ 2:50, leaves buffer)

**S1 · Hook (0:00–0:25)**
> Every year a mid-size bank flags around **388,000 accounts** as suspicious. **96%** are completely innocent people — and today a human analyst reviews every single one by hand. Q-SIEVE clears the obviously-innocent ones automatically, abstains when it's unsure, and — the part we're proud of — we proved the quantum engine runs on **three real quantum computers** this weekend.

**S2 · Problem (0:25–0:55)**
> The data is real — IBM's AML dataset, 5 million transactions. A simple velocity rule raises 388,000 account-week windows. Only **9%** are real laundering. The other 384,000 are innocent, and a human clears each one. It's expensive, it burns out analysts, and genuine cases hide in the flood.

**S3 · Solution (0:55–1:25)**
> Q-SIEVE is a **trust layer**. It only ever *removes* work: it auto-clears windows it's confidently sure are legitimate, and abstains on anything uncertain. It never accuses anyone, and it logs every decision — so it's audit- and regulator-safe. Three layers: features, a similarity kernel, and the trust layer that decides. The middle layer is where quantum comes in.

**S4 · Where quantum fits (1:25–1:50)**
> The hard question is *similarity*. Our quantum kernel encodes 12 behavioural features as rotation angles on qubits and measures how much two quantum states overlap — that fidelity **is** the similarity score. It's deliberately a swappable component: classical-first, quantum-enhanced. The chip does only the similarity; the decision stays classical and explainable.

**S5 · Honest benchmark (1:50–2:15)**
> We did not cherry-pick. We tuned the classical models properly and gave everyone identical data. Honest result: classical XGBoost wins on accuracy today at **0.19**; the quantum kernel scores **0.125** — real signal, and it beats the classical RBF kernel. We report it whichever way it falls, because that's the rigour a bank and a regulator need. The ceiling is the features, not the kernel.

**S6 · Hardware — the credibility slide (2:15–2:45)**
> What we're proudest of: we ran the exact frozen kernel on **three vendors' real quantum computers**. IBM and IQM — zero degradation, identical to simulation. VTT's Q50, the noisiest, cost just one PR-AUC point. Our shallow encoding survives real hardware noise. And three of those vendors are on this jury — the quantum part isn't theory, it runs today.

**S7 · Business + path (2:45–2:55)**
> Buyers are bank AML teams and RegTech vendors; value is measurable analyst time saved. It deploys classical-first now, with the quantum module already wired in for when hardware matures. Next step: a pilot on a bank's own labelled history.

**S8 · Demo + close (2:55–3:00)**
> *(show dashboard)* Q-SIEVE clears the innocent, flags the uncertain, and the quantum engine runs today — on three real quantum computers. Thank you.

---

## Q&A prep (3 min — be concise, don't repeat the pitch)

**"Classical wins — so why quantum at all?"**
> Honesty is the point. Today it's a feasibility study: we proved the kernel is hardware-ready and swappable, so the day a quantum advantage appears on this problem, Q-SIEVE already carries it — with zero re-architecture. We're future-proofing a product banks can use now.

**"Why is PR-AUC so low (~0.1–0.19)?"**
> Floor is 0.093 at this imbalance, so everyone's close — the signal is genuinely hard. The feature set is the ceiling; richer graph/network features lift everyone, classical and quantum alike. We measured that deliberately rather than inflating it.

**"Why only 6 features / 6 qubits on hardware when you have 12?"**
> Readout error compounds per qubit. We use the full 12 on the simulator benchmark and a shallow 6-qubit frozen batch on real hardware to keep noise low — that's why we see near-zero drift.

**"Is quantum actually *used*, or simulated?"**
> Both. Simulator for the full benchmark; and the real fidelity kernel executed on IBM, IQM, and VTT hardware — reconstructed decisions from real shot counts. That's in the results log.

**"What did you build this weekend vs before?"**
> Pre-event: the idea and the dataset. During: the full 3-layer pipeline, the fair benchmark, three real-hardware runs, the live dashboard — and we rebuilt the whole thing from scratch after a total local data loss on Friday.

**"PR-AUC vs accuracy — why?"**
> Accuracy is meaningless at 9% prevalence — predicting "all clean" scores 91%. PR-AUC measures ranking of the rare positive class, which is what triage actually needs.

**"How is this different from existing AML tools?"**
> Existing tools flag *more*. We're a trust layer that *removes* confident false positives and abstains otherwise — additive, low-risk, with a quantum similarity engine no classical vendor ships.

---

## Pre-submission checklist (deadline Sat 09:30)
- [ ] Add demo video or screenshots to slide 8 (placeholder marked)
- [ ] Put team member names on slide 1
- [ ] Rehearse to a timer twice — must land ≤ 3:00
- [ ] Export/keep as ONE `.pptx`, submit via the form
- [ ] (Safety) push repo to private GitHub
