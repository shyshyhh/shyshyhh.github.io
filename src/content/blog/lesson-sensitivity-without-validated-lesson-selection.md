---
title: 'Can a model teach the next one?'
description: 'Before asking whether model-to-model teaching can compound, we tested two simpler links: learning from a short lesson and choosing a useful one.'
date: 2026-07-21
draft: false
---

Could a model learn something today, then teach it to another model tomorrow? If that worked repeatedly, each generation could pass useful knowledge to the next instead of always depending on human-written lessons or a larger teacher.

That is what *compounding* would mean in this project: learning would produce a better teacher, which would help the next model learn, and so on. We are not there yet. Before such a chain is worth running, two simpler links have to work. A learner must improve from a short lesson, and a would-be teacher must recognize which lesson will help someone else.

We tested those links separately. The first worked in our controlled task: a 360M model changed after studying eight examples. The second did not: separate models from 360M to 7B failed our tests for choosing useful examples. Because that second link failed, we never ran the actual model-teaches-model experiment.

This distinction matters. We did not show one model learning from a lesson it could not judge. We found a working learner experiment and, separately, a failed lesson selector.

## How do you grade a lesson?

Teaching in ordinary language is hard to score. If a model writes a lesson about recursion or Roman history, deciding whether it is the *best* lesson usually requires a human or another language model. That would put a fuzzy judge inside the experiment we were trying to measure.

So we built a tiny classroom with an exact answer key. Each task hides a rule that labels strings of ten zeros and ones as either `0` or `1`. A lesson contains eight worked examples. Since we know the rule, we can grade every answer and count how many wrong rules a lesson eliminates. The tasks are artificial, but nothing is left to taste.

The learner gets a real weight update on those examples and then takes a separate test. Everything ran on one RTX 3090. We froze each experiment before its run and kept failed gates as written.

## Did eight examples help?

Yes. The learner studied a lesson chosen with access to the hidden rule and scored 5.16 points higher than a self-study control with the same update budget. Balanced accuracy gives equal weight to `0` and `1`; its conservative lower bound was +1.93 points. A lesson from the wrong concept moved performance 6.97 points in the other direction.

So eight examples could move a 360M model after an actual weight update. This validated the learner side of the setup. It did not show that another model could have found those examples.

Next we asked what part of the lesson mattered. In a fresh panel, every lesson had four examples of each label. One method chose examples that ruled out many wrong rules; the control chose at random. Careful choice added 3.90 points, with a lower bound of +0.59. Most of the gain came from two of the four task families.

![D-008 registered point contrasts](/figures/teaching/decomposition.svg)
*Choosing examples to eliminate more wrong rules beat random choice by 3.90 points (lower bound +0.59). Flipping their labels put the learner 21.07 points below the random control, but that comparison changes both example choice and correctness.*

Input choice mattered in this panel, and corrupting the labels was much worse. But this was a different panel from the first experiment. I cannot add the bars together and claim they explain the original 5.16 points.

## Could a model choose the lesson?

Now the job changes. Instead of learning from eight examples, a model whose weights stay fixed has to *choose* them for someone else. We showed each model the hidden task, first as a written rule and later through 16 demonstrations. It then scored new answers and ranked 16 possible lessons.

Our stand-in for lesson quality was how many of 1,025 possible rules a lesson ruled out. Better lessons leave fewer wrong rules alive. We could compute this exactly, but the real test would be whether a fresh learner improves. We had not reached that experiment.

The first small base and chat-tuned models did not pass. We then kept the test fixed and tried Qwen2.5 Instruct at 1.5B, 3B, and 7B parameters.

At the normal decision cutoff, all three missed our basic check that they understood the hidden task. Mean balanced accuracy was 0.500 at 1.5B, 0.546 at 3B, and 0.500 at 7B. A later look at their answer habits showed why the outer two looked so odd: the 1.5B model answered `1` on all 2,048 questions, while the 7B model answered `0` on 2,047 of 2,048. The 3B model answered `1` about three quarters of the time.

The binary answers hid some structure in the raw scores. AUC ignores the cutoff and asks whether the correct answer tends to score higher. Random is 0.5; we observed 0.6025, 0.6345, and 0.5680. This was a later diagnostic, not a registered test, and it did not rescue lesson selection.

![D-009C answer priors and score diagnostics by scale](/figures/teaching/collapse.svg)
*The 1.5B model always said `1`; the 7B model almost always said `0`. Their score ordering carried some above-random signal (blue), but no scale produced a usable lesson selector.*

The original accuracy rule still governed the result, but it was an incomplete diagnosis. Both answer scores can sit on the same side of a cutoff even when the correct one tends to be higher. That gave us one reasonable final test: move the cutoff without looking at the answers, then freeze everything else.

## Move the cutoff once

The final run used the 3B model on 16 fresh tasks. Before loading the model, we fixed one rule: split each balanced set at the model's own median `0`-versus-`1` score. This moved the cutoff without using the answer key. The usual zero cutoff remained the control.

We also fixed four questions that all had to receive a yes. Could the model label the task? Did moving the cutoff actually help? Did its chosen lessons eliminate more wrong rules than an average lesson? Did those choices beat random selection under an exact test?

Moving the cutoff raised mean balanced accuracy from 0.550 to 0.576. It still missed all four groups. Median accuracy reached 0.594 instead of the required 0.60. The average gain was +0.025 instead of +0.030, and only 9 of 16 tasks improved instead of 10. The chosen lesson's advantage over an average candidate was +0.209 bits instead of +0.250. The exact random-choice test returned p=0.122, above our 0.05 cutoff.

![D-009D gate audit](/figures/teaching/gates.svg)
*Green marks a passed criterion and red a miss. Some checks passed, but every hard group needed to pass, so the overall result failed.*

The run took 317 seconds and used 6.5 GiB of the RTX 3090. We did not retry it or adjust the threshold afterward. Median accuracy missed by 0.006, so the gate failed. That narrow miss does not establish a sharp boundary in what the model can express; this source channel simply missed its pre-set standard.

The model also did better at labeling some kinds of rules and better at choosing lessons for other kinds. That pattern may matter, but there were only four tasks per family and we had not registered a test for it. For now it is a clue, not a result.

## What would I try next?

I would change the kind of task before adding another model size. Boolean rules gave us an exact answer key, but they are unlike most of the text a language model sees during training. We already have a task built around made-up words and simple API instructions. It still has exact answers, but looks more like ordinary language.

Success there would make the Boolean tasks or their interface a more plausible culprit. Another failure would narrow the possibilities, but would not tell us whether the limit came from model size, model design, the prompt, or our stand-in for lesson quality. I would not run the teaching chain until a selector passes. Then comes the experiment I actually care about: after a model learns something, can it choose a lesson that helps a fresh model learn it too?
