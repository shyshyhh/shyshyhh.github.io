---
title: 'Lesson sensitivity without validated lesson selection'
description: 'In separate Boolean-assay experiments, a 360M recipient responded to lesson conditions while tested source interfaces up to 7B did not clear their selection gates.'
date: 2026-07-21
draft: false
---

Across separate Boolean-assay experiments, a 360M recipient was sensitive to lesson conditions. Tested frozen source interfaces up to 7B did not clear their lesson-selection gates.

These are two different results. The recipient got LoRA updates on mechanically constructed lessons. The source models stayed frozen and only scored labels and ranked candidate lessons. We did not test one model learning from a lesson it could not judge.

This was a sequence of experiments frozen one at a time, not one preregistration for the whole project. We ran them on a single RTX 3090 and kept every failed gate as written.

## Can the recipient learn from eight examples?

We used hidden Boolean rules over 10-bit inputs because every answer is exactly right or wrong. Each lesson has eight worked examples. The 360M recipient gets a real LoRA update and is then tested on a separate, label-balanced set.

In the first recipient experiment, a mechanically chosen oracle lesson beat the registered self-study control by 5.16 balanced-accuracy points. Its separately constructed one-sided 90% bounds were +1.93 (lower) and +8.53 (upper). A wrong-concept control lost 6.97 points and passed its negative-control gate. That showed the update assay could detect aggregate differences between lesson conditions. It did not show that a model could choose the useful condition.

We ran a fresh panel to isolate input choice while holding the lesson's 4/4 label balance fixed. One arm greedily chose inputs that ruled out as many candidate Boolean rules as possible. The other chose inputs at random. Greedy selection beat matched random by 3.90 points, with a one-sided 90% lower bound of +0.59 points. The gain was concentrated in conjunction and decision-list tasks; parity and DNF were roughly flat.

![D-008 registered point contrasts](/figures/teaching/decomposition.svg)
*Greedy-minus-random was +3.90 points (lower bound +0.59). Flipped-minus-random was -21.07 points (upper bound -16.06). The flipped arm reused the greedy inputs, so that second contrast mixes input selection and label corruption.*

The safe reading is simple: input choice mattered in this panel, and corrupting the labels was much worse. This was a different panel from the first experiment, so I cannot add the bars up and claim they explain the original 5.16 points.

## Can a source model pick the right eight?

We needed a working selector before spending GPU time on the actual teaching crossover. Frozen source models saw a Boolean task, first as an explicit rule and later as 16 labeled demonstrations. Each model scored new labels and ranked 16 candidate lessons.

Because the registered hypothesis space contained exactly 1,025 rules, we could score one proxy exactly: how much each lesson shrank that fixed set. This exact-H reduction was not a fresh recipient's learning outcome.

The small base and instruct checkpoints failed that calibration. We then kept the protocol fixed and tested Qwen2.5 Instruct at 1.5B, 3B, and 7B parameters.

Their registered fixed-zero classifications all missed the comprehension gate. Mean balanced accuracy was 0.500 at 1.5B, 0.546 at 3B, and 0.500 at 7B. A post-hoc look at the answer priors showed why the outer two looked so odd: the 1.5B model answered `1` on all 2,048 queries, while the 7B model answered `0` on 2,047 of 2,048. The 3B model answered `1` about three quarters of the time.

The observed mean per-concept AUCs were 0.6025, 0.6345, and 0.5680. All three were above 0.5, which is consistent with some score discrimination despite the failed binary classifications. These were exploratory diagnostics, not registered tests, and they did not validate lesson selection.

![D-009C answer priors and score diagnostics by scale](/figures/teaching/collapse.svg)
*The answer counts are post-hoc failure diagnostics. Mean per-concept AUC is exploratory; fixed-zero balanced accuracy governed the registered decision. Neither measure established a usable lesson selector.*

Fixed-threshold accuracy still governed the result, but it was incomplete as a diagnosis. Both label scores can sit on the same side of zero while their ordering carries descriptive signal. That gave us one concrete last test: calibrate the threshold without labels, then freeze everything else.

## One fresh test, then stop

The final run used the 3B model on 16 fresh concepts. Within this new design, the sole calibrated-versus-control intervention was a label-free rule fixed before inference: use the median log-odds on each balanced query pool as that concept's threshold. Fixed zero remained the paired control. We also sealed an exact randomization test before the run.

Calibration raised mean balanced accuracy from 0.550 to 0.576. The comprehension group still failed because median accuracy reached 0.594, below the required 0.60. The calibration mechanism also failed: mean gain was +0.025 across 9 of 16 positive concepts, short of +0.030 and 10 of 16. The selected lesson's median exact-H advantage over the mean candidate was +0.209 bits, below +0.250. The exact random-choice test returned p=0.122, above 0.05.

![D-009D gate audit](/figures/teaching/gates.svg)
*All four hard-gate groups failed, although individual clauses and both safeguards passed. The decision required every hard group to pass.*

The run took 317 seconds and used 6.5 GiB of the RTX 3090. We did not retry it or adjust the threshold afterward. Median accuracy missed by 0.006, so the gate failed. That narrow miss does not establish a sharp boundary in what the model can express; this source channel simply missed its pre-set standard.

In exploratory family summaries, calibrated comprehension and exact-H ranking advantage were strongest on different task families. There were only four concepts per family and no registered alignment test. This is heterogeneity, not evidence of anti-alignment or a demonstrated cause of the gate failure.

## Change the task before scaling up

I would change the domain before adding another model scale. Boolean rule induction is deliberately clean, but it is an odd fit for a pretrained language model. We already have a nonce-word API task family that keeps exact grading while looking more like material these models saw during pretraining.

Success there would make a Boolean-domain or interface mismatch more plausible. Another failure would narrow the possibilities, but it still would not separate model scale from architecture, prompting, or a bad proxy. I would not run the crossover until a selector passes. Then comes the experiment I actually care about: after a model learns something, can it choose a lesson that helps a fresh model learn it too?
