# Q-SIEVE — spoken talk

**Prize to aim at:** Best Technical Implementation Utilising Quantum.
**Do not open with the bank.** Open with the stuck quantum kernel.
**Do not use `PITCH_SCRIPT.md`.** That draft still says the tree wins 0.19 to 0.12 and starts from the queue. Bandwidth changed that headline.
**Clock:** about 3 minutes. Speak slowly. The 〔brackets〕 are for you. Do not say them.

Numbers you may say out loud, and the ones you must not upgrade:

- Fair untuned benchmark: tree **0.19**, quantum **0.12**, classical RBF **0.097**, random floor **0.093**.
- After bandwidth, same splits, 5 seeds, 3,000 test windows: quantum **0.18 ± 0.04**, tree **0.17 ± 0.03**. A **tie**. Never “quantum wins.”
- Do not quote **0.23**. That is a smaller 1,500-window run. The number to defend is 0.18.
- Untuned hardware, 6 qubits, 656 circuits: Garnet **0.111 → 0.111**, IBM Marrakesh **0.111 → 0.111**, Q50 **0.111 → 0.100** (drift 0.018, 358 s).
- Tuned hardware, c=0.1, 1,600 circuits, simulator **0.231**: Garnet **0.183**, IBM Fez **0.216**, Q50 **0.265**. Average **0.221**. Say “held on average.” Do not say Q50 improved the score.
- Safe auto-clear at ≤1% error: **0%** for both. The queue is not cleared today.

---

## The talk

**① The stuck place** 〔why anyone should listen〕

Let me start where quantum actually is, not where the slides usually start.

In 2026, on real data, a quantum kernel usually cannot beat a well-tuned classical model. People draw a curve where quantum eventually crosses that line. On this kind of problem the curve bends the other way: more data makes the kernel more expensive, and it does not become more accurate. That is the stuck place. We took it as the whole project. One real financial dataset. Find out exactly how close a quantum kernel can get, why it was losing, and whether the fix still exists once you leave the simulator.

**② The testbed, in one breath** 〔what / who / business, kept short〕

The dataset is IBM’s anti-money-laundering transactions. Five million payments, about 388,000 flagged account-weeks, almost all of them innocent. A bank already has a system that flags. What it does not have is permission to remove work. We built a trust layer that only clears an account when it is sure the account is legitimate. It never accuses anyone, and it logs the decision. The quantum computer does one job inside that layer: how similar are two account-weeks?

**③ The fair loss** 〔progress, and why we kept going〕

We ran the fair comparison before we touched the quantum side. Same data, same 200 training windows, a properly tuned tree. The tree scored 0.19. The quantum kernel scored 0.12. Random guessing here is 0.09, so the kernel is real signal, and it beats a classical RBF kernel at 0.097. It loses to the tree.

Then we tried the usual repairs. Four encodings. Training the circuit. A hybrid that let the tree take the easy windows and sent only the ambiguous ones to the quantum model. Ninety-three windows out of 3,000 moved. The score did not. Amplitude looked a little better on a perfect simulator and then fell apart under noise. The entangled ZZ map needed dozens of two-qubit gates and scored worse. It really looked stuck.

**④ Bandwidth** 〔the technical heart〕

The mistake was the scale of the rotations. We had stretched every feature across the full range, 0 to π. At that scale two different accounts rotate so far apart that every pair looks equally different. The kernel concentrates, and the signal disappears. We shrank the angles. At about a tenth of that range the same kernel went from 0.12 to 0.18. On those same splits the tree sat at 0.17. The error bars overlap. That is a tie. I will not tell you quantum beat the tree.

We also refused to find that scale by sweeping until something looked good. There is a geometric rule: one over the square root of qubits times features. On twelve features it predicts 0.083. The measured peak sits there. The usual automatic picker, kernel-target alignment, chose the worst setting, because only about 9 percent of the windows are suspicious and the alignment gets drowned by the innocent pairs. The formula worked. The textbook selector did not.

**⑤ Three real machines** 〔quantum used in practice〕

A simulator tie is cheap. This weekend the kernel went onto three real quantum computers. IQM Garnet, IBM, and VTT’s Q50, the last of those through LUMI. Simulator first, then a noise model, then the chip, so we did not spend device time on a guess.

The quiet configuration is six qubits and no entanglement. On that one, Garnet and IBM reproduced the simulator exactly, 0.111 to 0.111. Q50 moved it by one point, 0.111 to 0.100. That is the degradation report: two chips untouched, one chip costs a point.

The bandwidth-tuned kernel is the stronger one, and it is also noisier, because the angles are small and every similarity sits near the readout floor. Across the three chips the hardware scores average back onto the simulator: 0.22 against 0.23. A uniform readout shift cancels inside the SVM, which is why the ranking can hold while the raw kernel entries drift. We say “held on average.” With forty test points, a single chip bouncing up or down is the sample, not a discovery.

**⑥ What we will not claim** 〔effectiveness, honestly〕

We cannot tell you this clears the analyst’s queue today. At the rule we would actually ship, no more than one percent of the cleared accounts may be suspicious, both models clear nothing. The tree is also faster. A tenth of a second, against about eighteen seconds for the kernel, on the same three thousand windows. A real chip takes minutes, and that is for a few dozen windows, not the full queue. Six qubits is deliberate. The similarity is the fraction of shots that come back all zeros, and every extra qubit multiplies another chance of a bad readout. Twelve features stay on the simulator. The hardware batch keeps the six that separate the classes.

**⑦ Close**

So the business idea does not retire the analyst. The technical result is the method. We found why the kernel was stuck, derived the scale instead of searching for it, and ran it on three vendors, including the chip that moved. The dashboard is up if you want to open a real alert. Thank you.

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

**“So quantum still doesn’t win?”**
Correct. 0.18 against 0.17, and the spread is larger than the gap. The claim is that an untuned kernel at 0.12 was an artifact of the rotation scale, and once that scale is set, quantum is statistically level with the tree. A tie, reported as a tie.

**“Why is this the technical prize rather than the business one?”**
Because the queue does not shrink at the safe threshold. The work that will survive questions is the diagnosis, the formula, and the hardware degradation report. We would rather win on that than on a clearance rate we do not have.

**“Is 1 over square-root of n-d your invention?”**
No. Bandwidth and concentration are Shaydulin and Wild. What we showed on this dataset is that the geometric rule lands on the measured peak, and kernel-target alignment does not, because of the class imbalance. We report the selector that failed.

**“Did hardware really match, or did Q50 get lucky at 0.265?”**
Treat 0.265 as noise. Forty test points, one seed. The three hardware scores are 0.183, 0.216 and 0.265 around a simulator of 0.231. The average is 0.22. The clean statement is “held on average.” The configuration with almost no drift is the untuned one: Garnet and IBM identical, Q50 down 0.011.

**“Why only six qubits when the model has twelve features?”**
The kernel entry is the all-zero shot rate. At six qubits an identical pair already comes back around 0.74 instead of 1. That 0.26 is readout, and it barely changes with circuit depth. Twelve qubits would push that floor toward 0.5, so we would be ranking readout noise. The twelve-feature model stays on the simulator. Hardware keeps the six strongest features.

**“Why not score all 388,000 windows?”**
The quantum kernel compares pairs. Training at a few hundred windows is already the limit we held every model to, so the comparison stays fair. Scoring the full queue on a QPU, at the rate Garnet ran 656 circuits, is on the order of weeks. The 300,000 figure is the measured rate applied to the full flagged set, not a fresh score of every window.

**“Why doesn’t the quantum curve cross the classical line if we just wait?”**
Because this circuit is not a harder quantum computation as the queue grows. Angle encoding is one rotation per qubit and no entanglement. A laptop computes that overlap directly. What grows is the number of circuits, one per pair of windows. XGBoost grows with the number of windows. The kernel grows with the square of the training set. The gap widens.

**“You tried a hybrid. What happened?”**
The tree scored every window. Ninety-three ambiguous ones went to the quantum kernel. PR-AUC went from 0.198 to 0.196. Same false-positive line. The second model was not better on the rows the first model was unsure about.

**“Why PR-AUC, not ROC-AUC or accuracy?”**
Accuracy is about 91 percent if you call everything legitimate and catch nothing. On the same 3,000 windows ROC-AUC reads 0.66 for the tree while precision is still about one in five. PR-AUC’s floor is the prevalence, 0.093, so 0.18 is visibly “twice random” rather than “comfortably above 0.5.”

**“What about the papers that quote a 1 percent false-positive rate?”**
That point is real if you barely flag anything. At roughly 77 percent precision our recall is about 2.5 percent, for the tree and for the tuned kernel. At 90 percent recall the false-positive rate is about 80 percent for both. Low false positives with no recall is not a win.

**“Amplitude or ZZ instead of angle?”**
On the simulator they sit in the same band until you look at hardware. Amplitude’s noise drift was 0.16. ZZ needs 18 to 38 two-qubit gates. Angle is depth 2 and zero two-qubit gates, which is why it is the circuit we froze. After bandwidth tuning, angle still leads ZZ.

**“What did the quantum computer actually compute?”**
Not the decision. A fidelity. Two windows become angles, the overlap circuit runs, and the fraction of all-zero shots is the similarity. A classical SVM turns those similarities into a rank. The trust layer may only clear.

**“Show me the demo.”**
Open the dashboard. It is the real feature table: the rank, the calibration, and an audit line per alert. The hardware table is the other half of the demo: the same frozen batch on three machines.
