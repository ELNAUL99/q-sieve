# Q-SIEVE — spoken talk

**Prize to aim at:** Best Technical Implementation Utilising Quantum.
**Do not open with the bank.** Open with the stuck quantum kernel.
**Do not use `PITCH_SCRIPT.md`.** That draft still says the tree wins 0.19 to 0.12 and starts from the queue. Bandwidth changed that headline.
**Clock:** ~3 minutes, 8 beats. Conversational — talk TO the room, use the pauses. The 〔brackets〕 are for you; don't say them. If you're over at rehearsal, cut beat ⑦ to one line and trim the hardware beat.

Numbers you may say out loud, and the ones you must not upgrade:

- Fair untuned benchmark: tree **0.19**, quantum **0.12**, classical RBF **0.097**, random floor **0.093**.
- After bandwidth, same splits, 5 seeds, 3,000 test windows: quantum **0.18 ± 0.04**, tree **0.17 ± 0.03**. A **tie**. Never "quantum wins."
- Do not quote **0.23**. That is a smaller 1,500-window run. The number to defend is 0.18.
- Untuned hardware, 6 qubits, 656 circuits: Garnet **0.111 → 0.111**, IBM Marrakesh **0.111 → 0.111**, Q50 **0.111 → 0.100** (drift 0.018, 358 s).
- Tuned hardware, c=0.1, simulator **0.231**: three full-shot runs (one per vendor) **scatter** 0.183 / 0.265 / **0.395** — the IBM 0.395 is *above* the noiseless sim. Do **not** average these or say "held" — the tuned config is **noise-dominated / unstable** on hardware. The reliable hardware result is the **robust c=1 config** above.
- Safe auto-clear at ≤1% error: **0%** for both. The queue is not cleared today.
- Speed: the tree is **faster** — ~0.1 s vs ~18 s for the kernel on 3,000 windows; a real chip takes minutes. Quantum is slower. Say it before they ask.

---

## The talk  (~2:40 — safely under the hard 3:00 cap. Speak at a normal pace; this leaves buffer. Beat ⑦ is the first thing to cut if you run long.)

**① The stuck place**
Let's be honest. Everyone here knows that today, on real data, a quantum kernel usually can't beat tuned classical. So why are we here? The real question isn't who wins today — it's how *close* quantum gets, *why*, and whether the fix survives real hardware. We went and found out.

**② The testbed + scope**
The data: IBM's anti-money-laundering set, five million payments, 388,000 flagged account-weeks, almost all innocent. We're not solving money-laundering, or clearing the whole queue. One narrow job: clear only the windows we're confidently sure are innocent, so analysts focus on the uncertain ones. The kernel answers one thing: how similar are two windows?

**③ The floor**
Before any score, one number. Of the 388,000 flagged windows, about 1% are actually suspicious — that's the real queue. The table we score keeps all four thousand suspicious windows and caps the innocent at forty thousand, so it's 9.3% suspicious. A model ranking at random scores exactly that: 0.093. That's the floor. Every number after this starts there.

**④ The fair loss**
The fair fight first. The tuned classical tree scores 0.19; our quantum kernel, 0.12 — both above the 0.093 floor, so both are real signal, but the tree wins. We tried every repair: four encodings, training the circuit, a hybrid. Nothing moved. Stuck.

**⑤ Bandwidth — the fix**
Then we found it. The features were rotations across the full range — so every pair of windows looked equally different, and the kernel went blind. That range is the bandwidth. We shrank it: the kernel jumped from 0.12 to 0.18, level with the tree's 0.17. A tie, within noise; not a win. We swept to find it, then showed a known rule — one over root n times d, from Shaydulin and Wild — lands in that peak. The next choice does not need another sweep. That's the innovation.

**⑥ Three real machines**
A simulator tie is cheap, so we ran it on three real quantum computers — IQM, IBM, VTT's Q50. The robust config held: Garnet and IBM reproduced the simulator exactly, Q50 off a single point, drift under 0.02. Then the honest part — the aggressive tuned version bounced from 0.18 to 0.40, one run even *above* the noiseless simulator, which is impossible. It amplifies noise. Tuning buys simulator score, costs hardware stability. The robust config is the one that runs.

**⑦ The demo**
And it's live — a working trust layer: ranked queue, auto-clear, an audit line per decision, and the three-machine table. Real data. 〔Point at the two links. Code: github.com/ELNAUL99/q-sieve. Live dashboard: elnaul99.github.io/q-sieve.〕

**⑧ What we won't claim**
Let me close with what we *won't* claim. It doesn't empty the queue today — at a safe bar, both models clear zero, and the tree is faster. We say that before you ask. The honest map *is* the contribution: why it was stuck, the derived fix, three real machines. It tells a bank *when* quantum is worth the bet. Diagnose, derive, validate. Thank you.

---

## If a judge cuts in — one sentence each

| They are scoring | You say |
|---|---|
| Wake a decision-maker | The useful news is where quantum does *not* belong yet: not the decision, and not the full queue. |
| Business case | The layer only removes work, and today the safe threshold removes none. That is a finding, not a product claim. |
| Progress during the event | Untuned loss, encoding sweep, hybrid, bandwidth, analytic scale, three real QPUs. |
| Different from classical tools | Classical tools rank and escalate. This one is built to abstain, and the similarity was executed on hardware. |
| More effective or efficient? | No. After the fix it is level on accuracy and slower on time. Say that before they do. |
| Demo | The dashboard, plus the three-machine table. |
| Was quantum used? | Yes, in practice. Garnet, IBM, Q50. Not a Bell pair. The AML kernel. |

---

## Q&A

**"So quantum still doesn't win?"**
Correct. 0.18 against 0.17, and the spread is larger than the gap. The claim is that an untuned kernel at 0.12 was an artifact of the rotation scale, and once that scale is set, quantum is statistically level with the tree. A tie, reported as a tie.

**"Why is this the technical prize rather than the business one?"**
Because the queue does not shrink at the safe threshold. The work that will survive questions is the diagnosis, the formula, and the hardware degradation report. We would rather win on that than on a clearance rate we do not have.

**"Is 1 over square-root of n-d your invention?"**
No. Bandwidth and concentration are Shaydulin and Wild. What we showed on this dataset is that the geometric rule lands on the measured peak, and kernel-target alignment does not, because of the class imbalance. We report the selector that failed.

**"Did the tuned kernel really hold on hardware?"**
No, and we say so. The same tuned batch gave 0.183, 0.265 and 0.395 across the three vendors, around a simulator of 0.231. One IBM run landed *above* the noiseless simulator, which is impossible for a real signal — it is noise. The tuned model is near-degenerate, every point a support vector, so tiny readout noise is amplified by large dual coefficients into big PR-AUC swings; the kernel drift stays low, 0.033, while the score flies. The reliable hardware result is the robust config: Garnet and IBM reproduce the simulator exactly, 0.111 to 0.111, Q50 down 0.011, drift 0.007 to 0.018.

**"Isn't the big 156-qubit IBM chip just noisier?"**
Our data says the opposite. We use six physical qubits on every device, so the total size does not add noise to our circuit — the other 150 sit idle. Measured drift: Garnet at 20 qubits 0.040, Q50 at 50 qubits 0.116, IBM at 156 qubits 0.033. The largest device was the cleanest. What matters is the calibration and readout fidelity of the six qubits we actually use, not the chip size. So the 0.395 swing is model degeneracy, not device noise.

**"Why only six qubits when the model has twelve features?"**
The kernel entry is the all-zero shot rate. At six qubits an identical pair already comes back around 0.74 instead of 1. That 0.26 is readout, and it barely changes with circuit depth. Twelve qubits would push that floor toward 0.5, so we would be ranking readout noise. The twelve-feature model stays on the simulator. Hardware keeps the six strongest features.

**"Why not score all 388,000 windows?"**
The quantum kernel compares pairs. Training at a few hundred windows is already the limit we held every model to, so the comparison stays fair. Scoring the full queue on a QPU, at the rate Garnet ran 656 circuits, is on the order of weeks. The 300,000 figure is the measured rate applied to the full flagged set, not a fresh score of every window.

**"Why doesn't the quantum curve cross the classical line if we just wait?"**
Because this circuit is not a harder quantum computation as the queue grows. Angle encoding is one rotation per qubit and no entanglement. A laptop computes that overlap directly. What grows is the number of circuits, one per pair of windows. XGBoost grows with the number of windows. The kernel grows with the square of the training set. The gap widens.

**"You tried a hybrid. What happened?"**
The tree scored every window. Ninety-three ambiguous ones went to the quantum kernel. PR-AUC went from 0.198 to 0.196. Same false-positive line. The second model was not better on the rows the first model was unsure about.

**"Why PR-AUC, not ROC-AUC or accuracy?"**
Accuracy is about 91 percent if you call everything legitimate and catch nothing. On the same 3,000 windows ROC-AUC reads 0.66 for the tree while precision is still about one in five. PR-AUC's floor is the prevalence, 0.093, so 0.18 is visibly "twice random" rather than "comfortably above 0.5."

**"What about the papers that quote a 1 percent false-positive rate?"**
That point is real if you barely flag anything. At roughly 77 percent precision our recall is about 2.5 percent, for the tree and for the tuned kernel. At 90 percent recall the false-positive rate is about 80 percent for both. Low false positives with no recall is not a win.

**"Amplitude or ZZ instead of angle?"**
On the simulator they sit in the same band until you look at hardware. Amplitude's noise drift was 0.16. ZZ needs 18 to 38 two-qubit gates. Angle is depth 2 and zero two-qubit gates, which is why it is the circuit we froze. After bandwidth tuning, angle still leads ZZ.

**"What did the quantum computer actually compute?"**
Not the decision. A fidelity. Two windows become angles, the overlap circuit runs, and the fraction of all-zero shots is the similarity. A classical SVM turns those similarities into a rank. The trust layer may only clear.

**"Show me the demo."**
Open the dashboard. It is the real feature table: the rank, the calibration, and an audit line per alert. The hardware table is the other half of the demo: the same frozen batch on three machines.

**"This does nothing for a real business. Why should finance care?"**
Nothing today — correct. The most useful thing we can give a finance decision-maker is the honest truth about where quantum stands on their problem, so they invest at the right time instead of chasing hype. A rigorous "not yet, here is how close and why" de-risks a decision worth millions. That is the value, and it is real.

**"It is all theoretical — a feasibility study."**
Yes. And in 2026 almost all quantum work is, including the work at the companies on this panel. We did honest frontier characterization on real financial data and real hardware. The event asks for technical implementation using quantum, not a shipped product. Feasibility done rigorously is the deliverable. "Theoretical" is foundational — the honest first step before anyone should spend money.

**"A laptop simulates six qubits instantly — why run on a QPU at all?"**
For this circuit the QPU was not necessary; a laptop does it. We ran it for three reasons: to prove the full pipeline executes on real machines (de-risking the integration for the day advantage exists), to learn which configurations survive real noise (you can only confirm that on metal — the robust one does, the tuned one does not), and for credibility over a simulator-only claim. Feasibility and characterization, not a necessity claim.

**"Angle encoding with no entanglement is classically simulable — so is this even quantum?"**
Honestly, for this circuit there is no advantage and it is simulable. We chose it because it is the only NISQ-robust option. The chip genuinely computed the fidelities; our contribution is the honest benchmark and the hardware characterization, not a claim that this circuit is classically hard. Advantage would live in deeper, structured circuits on better hardware — future work, stated as future work.

**"Features are the ceiling, so the quantum kernel is irrelevant — just do better features classically."**
Agreed that better features lift everyone, and that is the near-term win, done classically. We are not claiming the quantum kernel is the business lever. We characterized it honestly: at the right bandwidth it is level with classical, and it runs on hardware. A research result, not a product claim.

**"Aren't you overclaiming — this doesn't solve AML?"**
Correct, and we never claim to. Our scope is deliberately narrow. We do not detect laundering, we do not replace the bank's flagging system, and we do not clear the whole queue. We take the windows that are already flagged and auto-clear only the ones we are confidently sure are innocent — removing analyst busywork so they spend time on the uncertain cases. A small, bounded, honest slice, not a global solution.
