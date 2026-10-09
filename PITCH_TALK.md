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
- Tuned hardware, c=0.1, 1,600 circuits, simulator **0.231**: Garnet **0.183**, IBM Fez **0.216**, Q50 **0.265**. Average **0.221**. Say "held on average." Do not say Q50 improved the score.
- Safe auto-clear at ≤1% error: **0%** for both. The queue is not cleared today.
- Speed: the tree is **faster** — ~0.1 s vs ~18 s for the kernel on 3,000 windows; a real chip takes minutes. Quantum is slower. Say it before they ask.

---

## The talk

**① The stuck place** 〔the hook — why anyone should listen〕

Let's be honest with each other for a second. Everyone in this room already knows that in 2026, on real data, a quantum kernel usually can't beat a well-tuned classical one. So why are we even here? Why isn't this a *Classical* × Finance hackathon?

Here's the honest answer. People love to draw quantum as a curve that eventually crosses the classical line. On a problem like ours, that curve actually bends the wrong way — more data makes the kernel more *expensive*, and no more accurate. That's the stuck place. And instead of hiding from it, we made it the whole project: one real financial dataset, and three questions — how close can a quantum kernel get, why was it losing, and does the fix survive once you leave the simulator?

**② The testbed, in one breath** 〔business, kept short〕

The data is IBM's anti-money-laundering set — five million payments, about 388,000 flagged account-weeks, almost all of them innocent people. The bank already has something that flags. What it doesn't have is permission to *remove* work. So we built a trust layer that only clears an account when it's sure it's clean — it never accuses anyone, and it logs every call. And the quantum computer does exactly one job inside it: how similar are two account-weeks?

**③ Thirty seconds on how it works** 〔the math — makes everything after it land〕

That one job is the whole story, so give me thirty seconds. We turn each account into a quantum state. The kernel is just how much two of those states overlap — identical accounts, overlap one; opposite accounts, zero. Now here's the catch, and remember it: if you rotate the data too far, every account spreads to opposite corners and they all look equally different. The kernel goes blind. How far you rotate is one number — the bandwidth. Hold that word.

**④ The fair loss** 〔progress, and why we kept going〕

Okay — the numbers. We ran the fair fight first, before we touched anything quantum. Same data, same 200 training windows, a properly-tuned classical tree. The tree scored 0.19. Our quantum kernel scored 0.12. Random guessing here is 0.09, so it's real signal — it even beats the classical RBF kernel — but it loses to the tree. We could've stopped and shown you a prettier slide. Instead we asked why, and tried every repair: four encodings, training the circuit, a hybrid. Ninety-three windows moved; the score didn't. It really looked stuck.

**⑤ Bandwidth — the fix** 〔the technical heart〕

Then we found it. Remember the bandwidth — how far we rotate? Ours was wrong. We'd stretched every feature across the full range, and the kernel was blind, exactly like I showed you. We shrank the angles to about a tenth — and the same kernel jumped from 0.12 to 0.18. On the same splits, the tree sits at 0.17. The error bars overlap, so I'll say it plainly: quantum didn't *beat* the tree — it drew level. And we refused to find that scale by fiddling knobs. We derived it — one over root n-times-d — which predicts 0.083, and the measured peak sits right there. No trial and error. That's the curve climbing back toward the line.

**⑥ Three real machines** 〔quantum used in practice〕

But a simulator tie is cheap. So this weekend we put that kernel on three real quantum computers — IQM's Garnet, IBM, and VTT's own Q50 — and we did it properly: simulator, then noise model, then the chip, so we never spent device time on a guess. On the quiet six-qubit config, Garnet and IBM reproduced the simulator exactly — 0.111 to 0.111 — and Q50 moved it by a single point. On the tuned kernel, the three chips average right back onto the simulator: 0.22 against 0.23. It held on average, across three vendors. It runs — today — on real metal.

**⑦ The demo** 〔the live product〕

And it's all live. A working trust layer — the ranked queue, the auto-clear, an audit line for every decision, and the evaluation page with that three-machine table. Real data, real results.

**⑧ What we won't claim — and why that's the point** 〔close + honesty〕

So let me close by telling you what we *won't* claim. This doesn't empty the analyst's queue today — at the threshold we'd actually ship, both models safely clear zero. And the tree is faster — a tenth of a second against eighteen for the kernel. We're telling you that before you ask. Because the honest map *is* the contribution: we found why the kernel was stuck, we derived the fix instead of guessing it, and we proved it runs on three real machines. The curve hasn't crossed the line yet — but we found the line, and we know which way the curve is moving. Diagnose, derive, validate. Team Q-SIEVE. Thank you.

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

**"Did hardware really match, or did Q50 get lucky at 0.265?"**
Treat 0.265 as noise. Forty test points, one seed. The three hardware scores are 0.183, 0.216 and 0.265 around a simulator of 0.231. The average is 0.22. The clean statement is "held on average." The configuration with almost no drift is the untuned one: Garnet and IBM identical, Q50 down 0.011.

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
