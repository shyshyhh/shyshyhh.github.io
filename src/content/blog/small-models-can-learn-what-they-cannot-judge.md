---
title: 'Small models can learn what they cannot judge'
description: 'Five pre-registered experiments on one RTX 3090. Fine-tuning on good lessons works (+5.2 points). Asking the same models to judge lessons fails at every scale from 360M to 7B, and the failure mode is not what accuracy numbers suggest.'
date: 2026-07-21
draft: false
---

Can a small language model teach an even smaller one? I ran five pre-registered experiments on this question last night (models from 360M to 7B, one RTX 3090). The answer split in half: small models measurably learn from a good lesson, but they can't tell a good lesson from a bad one. They pass the class and can't grade a single homework.

The failure mode turned out to be more interesting than the result. Here's the short version.

## Lessons work, mostly because the answers are right

Setup: synthetic reasoning tasks (hidden Boolean rules over 10-bit inputs, so every answer is exactly gradable), a 360M student, and real weight updates. A "lesson" is eight worked examples, and the student is fine-tuned on it. The control student gets the same token budget of generic study material.

Good lessons teach: +5.2 accuracy points over self-study. Corrupted lessons hurt (-7.0). So the instrument works. Then we ran a falsification panel to take the effect apart: one arm flips the answers on byte-identical examples, another randomizes example choice while keeping the answer balance fixed.

![What makes a lesson teach](/figures/teaching/decomposition.svg)
*Nearly all of a lesson's value is its answers being right. Smart example selection adds +3.9 points; flipping the answers on the same examples costs -21.1.*

So a lesson at this scale is mostly a container for correct answers. Choosing examples well is real but small. Being wrong is catastrophic. Not shocking in hindsight, but now it's measured, and the +3.9 selection effect matters later.

## What they said vs. what they knew

Next step for the bigger question (does being taught make you a better teacher?): the model has to act as a teacher. Shown a task, it should pick informative examples, or at least rate candidate lessons better than chance. We tested five models (SmolLM, Qwen, Pythia, base and instruct), then walked up Qwen's instruct ladder: 1.5B, 3B, 7B.

Every one failed the comprehension check. But look at *how*:

![What they said vs what they knew](/figures/teaching/collapse.svg)
*Left: the 1.5B answered "1" to all 2,048 queries; the 7B answered "0" to 2,047 of 2,048. Right: ranking the same models' raw confidences (AUC) shows real signal at every scale. They knew more than they said.*

The 1.5B answered "1" two thousand and forty-eight times in a row. The 7B answered "0" to all but one. Scored as accuracy, both are exactly at chance. Case closed, except accuracy is a threshold measurement, and the thresholds were parked in absurd places by instruction tuning. Rank the raw confidence scores instead (AUC, which ignores the threshold entirely) and there's signal at every scale: 0.60, 0.63, 0.57. The capability was present and unexpressed.

Two things I take from this. First, I don't fully trust fixed-threshold evaluations of small models anymore (the model can know the answer and still never say it). Second, note the 7B is *worse* than the 3B on both measures. Bigger did not help here, which killed my first hypothesis for what to try next.

## Missing by 0.006

If the threshold is the problem, move the threshold. Final experiment: re-read the 3B's answers using its own median confidence as the cutoff. No labels needed, no prompt changes, one run. We froze four pass/fail gates before the model loaded, including an exact combinatorial test that its lesson choices beat random selection (computed in exact rational arithmetic, because a floating-point tie had already burned us once earlier in the night).

![The four gates](/figures/teaching/gates.svg)
*Calibration recovered real accuracy (0.550 to 0.576 mean) and still missed every pre-registered bar. The comprehension gate failed at 0.594 against a 0.60 threshold.*

The calibration did what we predicted (+2.5 points mean accuracy) and failed anyway. Median accuracy came in at 0.594 against a 0.60 bar. Missed by 0.006. The other gates failed by similarly thin margins. We had pre-committed that a near-miss is a miss (no threshold-nudging after seeing data, no "one more run"), so that's a fail, and the local search is closed by our own rules. It cost ~10 GPU-minutes to find out. Fwiw, a 0.006 miss against frozen gates tells you more than a lucky pass would have: whatever lesson-judging ability exists at 3B sits right at the edge of expressibility, not comfortably inside it.

One result I still can't explain: the 3B's comprehension was best on exactly the task families where its lesson-judging was worst (conjunctions vs. decision lists), and vice versa. Understanding a task and knowing how to teach it didn't just fail to correlate. They anti-aligned.

## What's next

What I believe now that I didn't yesterday: teaching effects through real weight updates are measurable and decomposable on a single consumer GPU; "small models can't do X" claims need a check of what the model knows *below* its answer threshold; and the learn/judge gap doesn't close by 7B, at least not through any answer interface we tried.

The open question: is this about model scale, or about our deliberately alien tasks? Boolean rule induction is close to worst-case for LM priors. A friendlier domain (we have a nonce-word API task family built and waiting) might open the teacher channel at sizes I can run locally. That experiment costs $0 and is probably next.

Process note: the five designs, sealed artifacts, exact-arithmetic gates, and two independent audits of every number were executed overnight by a pair of AI research agents holding each other to pre-registered rules while I slept. They caught three real bugs in each other's statistics. That setup deserves its own post.

*Code, sealed run artifacts, and the full pre-registration trail will be released with the paper.*
