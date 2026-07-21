---
title: 'Can a model teach the next one?'
description: 'A 360M model learned from eight examples. Models up to 7B could not reliably choose the lesson. Here is how that happened.'
date: 2026-07-21
draft: false
---

Last night I gave a single RTX 3090 a slightly odd homework assignment: can one small language model teach the next one?

By morning, the student side worked and the teacher side did not. A 360M model improved after studying eight examples. Separate models from 360M to 7B could not reliably choose a useful set of eight.

That was not yet the experiment I wanted to run. The longer-term idea is a chain: teach model A something new, let A choose a lesson for model B, then see whether B becomes a better teacher for model C. A small model normally needs lessons from people or a larger model. If learning could also make it better at teaching, some of that work might pass forward.

But there is no point building the chain if a model cannot choose a decent lesson. So we split the job in two. First, can a model learn from eight examples? Second, can a model pick the eight examples that another learner should see?

One boring but important sentence: these were separate experiments. We did not teach one model and then watch it teach the next. We were checking whether that experiment was ready. It wasn't.

## A tiny classroom

We needed lessons we could grade without asking another language model for its opinion. So the classroom was deliberately weird.

Each task hides a rule over ten switches, each either on or off. Think of something like “answer `1` only when the first and fourth switches are on.” A lesson is eight worked examples. The learner studies them, gets a real weight update, and then takes a separate test.

Because we know the hidden rule, every answer has an answer key. We can also count how many possible rules remain after reading a lesson. The tasks are not realistic. That is the deal: we give up realism for a classroom where nothing depends on vibes.

## The student learns

The first lesson was chosen with access to the hidden rule. After studying it, the 360M learner scored 5.16 points higher than a self-study control with the same update budget. Even the conservative lower bound was +1.93 points. Give it a lesson from the wrong concept and performance moved 6.97 points the other way.

Good. The student could learn. Then we found an annoying catch.

The useful lessons often had a nicer balance of `0` and `1` answers than the control lessons. Maybe the model was not benefiting from clever example choice as much as it looked. Maybe it just liked the label balance.

So we ran a fresh panel where every lesson had four examples of each label. Now the only question was *which* examples to show. Choosing examples that eliminated many wrong rules beat random choice by 3.90 points. Most of that gain came from two kinds of rule; the other two were roughly flat.

![Which part of the lesson mattered?](/figures/teaching/decomposition.svg)
*Once label balance was fixed, careful example choice helped by 3.90 points. The red bar is a harsher mixed test: it keeps those inputs, flips their answers, and compares them with the random control.*

The red bar is large, but don't add these bars together. This was a different set of tasks from the first experiment, and flipping the answers changes both correctness and the comparison against random. The safe result is smaller: example choice mattered, but it was not the whole lesson.

## The models take the teacher's seat

Now we swapped seats. A model whose weights stayed fixed saw the hidden task, either as a written rule or through 16 examples. We gave it 16 possible lessons and asked it to rank them.

Our temporary grade for a lesson was simple: how many of 1,025 possible rules did it rule out? Fewer surviving wrong rules meant a better grade. This was only a screening test. The real grade would come from giving the chosen lesson to a fresh learner, but running that experiment made no sense until the screen worked.

The first small base and chat models failed. We then tried Qwen2.5 Instruct at 1.5B, 3B, and 7B parameters. I expected the larger models to make the story cleaner. They did not.

The 1.5B model answered `1` on all 2,048 questions. The 7B model went the other way and answered `0` on 2,047 of 2,048. The 3B model at least used both answers, though it still missed our bar. Not much of a teaching faculty.

![The answers collapsed while the raw scores retained some ordering](/figures/teaching/collapse.svg)
*The 1.5B model always said `1`; the 7B model almost always said `0`. Ignoring the final cutoff, their raw score rankings (blue) still sat above the 0.5 reference.*

The blue line is why we did not stop immediately. AUC ignores the final `0`/`1` cutoff and asks whether the correct answer tends to receive a higher score. Random is 0.5. We saw 0.6025, 0.6345, and 0.5680.

Those numbers came from a later diagnostic, not a test we had promised in advance. They did not turn the models into useful teachers. But they suggested a specific failure: perhaps the scores carried some ordering while the cutoff sat in a dumb place.

## One last shot

We went back to the 3B model with 16 fresh tasks. Before loading it, we fixed one simple rule: put the cutoff at the model's own median score on each balanced set. No answer key was used to place it. We also wrote down four groups of checks, and all four had to pass.

Moving the cutoff helped. Mean accuracy rose from 0.550 to 0.576. It still failed every hard group. The closest miss was median accuracy, 0.594 against a required 0.60. Its lesson ranking also missed, and the exact test against random choice gave p=0.122 rather than less than 0.05.

![The final 3B run missed every hard gate group](/figures/teaching/gates.svg)
*Some individual checks passed (green), but every hard group needed to pass. Each group contained at least one miss.*

A miss by 0.006 is exactly the kind of result that whispers “one more tweak.” We had frozen the rule before the run, so we left the line where it was. The run took 317 seconds, used 6.5 GiB of VRAM, and was not retried.

There is one loose thread. The 3B model labeled some kinds of rules best, but chose its best lessons for different kinds. There were only four tasks per family and we had not planned that comparison, so for now it stays a clue.

## What next?

I would change the classroom before buying a bigger teacher. Boolean rules give us a perfect answer key, but they look nothing like ordinary language. We already have a task built around made-up words and simple API instructions. It stays exactly gradable while looking more like something a language model might naturally learn.

If a selector passes there, we can finally run the experiment I wanted in the first place: teach one model, let it choose a lesson, and see what the next model learns. If it fails again, I still won't know whether scale, prompts, or our lesson grade is to blame. But at least we will have asked in a less alien classroom.
