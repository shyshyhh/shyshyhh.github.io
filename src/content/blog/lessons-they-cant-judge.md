---
title: 'Learning from lessons they cannot judge'
description: 'Small language models measurably learn from good lessons — but they cannot tell a good lesson from a bad one. Five pre-registered experiments in one night, and the most useful one failed by 0.006.'
date: 2026-07-21
draft: false
---

Can a small language model teach an even smaller one? I spent last night finding out, and the answer split cleanly in half: **small models can *learn* from a well-made lesson, but they can't *judge* one** — they're students who pass the class and then can't grade a single homework. The way they failed turned out to be more interesting than a success would have been.

Some context: I'm interested in whether "teaching ability" is a real, measurable trait of a language model — whether being taught well makes you a better teacher, the missing step between ordinary self-training and actual recursive self-improvement. Before you can ask that question you need two boring-sounding things: proof that lessons teach at all, and a model that can act as a teacher. The first turned out fine. The second is where the night got strange.

## Lessons work — but not for the reason you'd hope

The setup: tiny synthetic reasoning tasks (hidden Boolean rules over 10-bit inputs) where every answer is exactly gradable, a 360M-parameter student, and real weight updates — the student is actually fine-tuned on each lesson, not just prompted with it. A "lesson" is eight worked examples. We compare against a student who gets the same token budget of generic study material.

Good lessons genuinely teach: +5.2 accuracy points over self-study, and deliberately corrupted lessons *hurt* (−7.0). So the instrument works. But when we tore the effect apart with a controlled falsification panel — same examples, only the answers flipped; same answer balance, only the example choice randomized — the anatomy was lopsided:

![What makes a lesson teach](/figures/teaching/decomposition.svg)
*Almost all of a lesson's power is in its answers being right. Choosing examples intelligently adds a real but small bonus (+3.9); corrupting the answers on identical examples is catastrophic (−21.1).*

In other words: at this scale, a lesson is mostly a vehicle for correct answers. Pedagogy — the careful choice of *which* examples to show — is real, measurable, and worth about a sixth as much as simply not being wrong.

## Then we asked the models to be teachers

Here's where it broke. For the bigger question, a model has to *produce* judgments: shown a task, pick informative examples, or rate candidate lessons. We tested five small models (SmolLM, Qwen, Pythia — base and instruction-tuned variants), then scaled up through Qwen's 1.5B, 3B, and 7B instruct models. Every single one failed our comprehension check — and the *way* they failed is the finding:

![Bigger models said less, not more](/figures/teaching/collapse.svg)
*Left: the 1.5B model answered "1" to all 2,048 questions; the 7B answered "0" to all but one. Right: threshold-free measurement (AUC) shows real signal hiding at every scale — the models knew more than they said.*

The 1.5B model answered "1" two thousand and forty-eight times in a row. The 7B — a model four times larger — answered "0" with the same fanaticism, breaking character exactly once. Read naively, that's chance-level comprehension, case closed. But accuracy is a *threshold* measurement, and when we instead ranked the models' raw confidence scores (AUC, which ignores where the threshold sits), every scale showed genuine signal — the 3B most of all. The models knew more than they said. Their instruction tuning had simply parked the answer threshold somewhere absurd, and everything they knew was trapped behind it.

I find this genuinely unsettling as an evaluation lesson: a fixed-threshold benchmark would have called these models incapable when the capability was measurably present, just inexpressible. And it's not monotonic in scale — the 7B was *worse* than the 3B — so "just use a bigger model" isn't obviously the answer either.

## The fix that failed by 0.006

If the threshold is the problem, move the threshold. We froze one final experiment: re-read the 3B's answers using its own median confidence as the cutoff — no labels needed, no prompt tricks, one run, four pass/fail gates locked in before the model loaded (including an exact combinatorial test that the model's lesson choices beat random selection — computed as exact fractions, because we'd already been burned once by floating-point ties).

![The four gates, one run, no retries](/figures/teaching/gates.svg)
*Calibration recovered real accuracy (0.550 → 0.576 mean) — and still missed every pre-registered bar. The comprehension gate failed by 0.006.*

The calibration worked, in the sense that it did exactly what we predicted: mean accuracy rose about 2.5 points. And it failed, in the sense that mattered: the pre-registered median bar was 0.60 and the model landed at 0.594. Missed by 0.006. Three other gates failed by similarly thin margins.

We'd agreed in advance that a near-miss is a miss — no nudging thresholds after seeing data, no "one more run." So the local search is closed, permanently, by our own rules. It cost about ten GPU-minutes to find out, and honestly, a 0.006 miss against frozen gates tells you more than a lucky pass would have: whatever teaching judgment exists at 3B, it's right at the edge of expressibility, not comfortably inside it.

One more oddity worth reporting because nobody would design it: the 3B's *comprehension* was best on the task families where its *lesson-judging* was worst, and vice versa. Understanding a task and knowing how to teach it didn't just fail to correlate — they anti-aligned.

## What I believe now

Three things I didn't believe yesterday. First, teaching effects through real weight updates are measurable and decomposable with hobby-scale hardware — a single RTX 3090 ran every experiment here. Second, "small models can't do X" claims deserve deep suspicion until someone checks what the model knows *below* its answer threshold. Third, the learn/judge asymmetry looks fundamental at this scale: consuming pedagogy is easy, producing it is not, and the gap doesn't close by 7B — at least not through any answer interface we tried.

The open question I care about: is the wall about model *scale*, or about our deliberately alien tasks? Boolean rule induction is close to worst-case for language-model priors; a friendlier domain might open the teacher channel at sizes I can run locally. That experiment costs nothing and is probably next.

A note on process, because it shaped everything: all of this — five pre-registered designs, sealed artifacts, exact-arithmetic decision gates, and two independent audits of every result — was executed overnight by a pair of AI research agents holding each other to the rules while I slept. They caught three real bugs in each other's statistics along the way. That workflow deserves its own post.

*Code, sealed run artifacts, and the full pre-registration trail will be released with the paper.*
