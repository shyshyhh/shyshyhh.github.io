---
title: 'A GPT you can take apart'
description: 'A trained eight-wide decoder you can rotate, expand, and inspect down to every attention score, cached key/value, major activation tensor, and learned weight.'
date: 2026-07-29
draft: false
experience: gpt-architecture
lede: 'Modern LLM diagrams hide their hardest parts behind a few labeled boxes. I trained an eight-wide decoder so you can open every layer, rotate every two-number head, and inspect every weight.'
ogImage: '/figures/a-gpt-you-can-take-apart/og.png'
---

I kept getting stuck at the same point when reading about modern language models. The block diagram made sense: tokens go in, repeated Transformer blocks do something, logits come out. Then the paper said grouped-query attention reduces KV-cache traffic, or RoPE puts position into queries and keys, and my mental picture collapsed back into labeled rectangles.

I wanted a model I could pick up. Rotate the whole stack, open one block, click one token, then go all the way down to a two-number attention head. If four query heads share two key/value heads, those wires should literally meet. If RoPE rotates vectors, I should be able to watch them turn.

The model above is that attempt: a trained decoder small enough for every learned weight and every major forward-pass tensor to stay visible, while keeping the parts that show up in modern open LLMs.

## First, don’t build it

Before writing code, I looked for an existing version.

[Brendan Bycroft’s LLM Visualization](https://bbycroft.net/llm/) is gorgeous. You can descend from the full model to individual additions and multiplications, and the transitions make the architecture feel like one connected machine. Its walkthrough centers on the GPT-2-era stack, though, before several components now common in open models.

[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) runs a real GPT-2 model in the browser and explains the forward pass with a clean 2D interface. It is probably the easiest place to start if GPT itself is new to you.

[TokenPrint](https://github.com/Sudharsanselvaraj/Token-Print) was the closest match I found: modern architectures, 3D, and live inference. When I looked, it was a source repository rather than a finished hosted experience I could hand to someone on a phone.

So there was still a useful gap. I wanted the continuity of Bycroft’s 3D walkthrough, the live numbers of Transformer Explainer, and a small modern decoder whose important state could fit on screen without turning into a wall of matrix cells.

## Smaller than nano

The code owes an obvious debt to Andrej Karpathy’s [minGPT](https://github.com/karpathy/minGPT) and [nanoGPT](https://github.com/karpathy/nanoGPT). Those projects showed that GPT becomes much less mysterious when the implementation is kept short enough to read. But small code is not necessarily a small visual scene. Even an educational model with 64-dimensional activations and 128-wide hidden layers produces too many numbers to inspect at once.

This one goes lower:

- 4 decoder layers
- 8 numbers per token
- 4 query heads
- 2 key/value heads
- 2 numbers per head
- a 16-wide gated MLP
- 23 tokens in the vocabulary

Inside a block, the largest matrices are only \(8 \times 16\). You can expand all four layers, isolate one layer, or open a single operation without the browser giving up.

The important part is that the data flow is real. The browser runs the forward pass using frozen weights produced by the [training script](https://github.com/shyshyhh/shyshyhh.github.io/blob/main/scripts/train-nano-gpt.py). Its decode control appends one token using the saved per-layer keys and values instead of replaying the prefix. The cells, attention scores, rotations, cache entries, residual updates, and output probabilities are not decorative animation.

*A Transformer becomes legible when its dimensions are small enough to see every number.*

## Modern, with an asterisk

I don’t know the exact architecture inside current commercial GPT products. OpenAI’s [GPT-4 technical report](https://arxiv.org/abs/2303.08774), for example, explicitly withholds architectural details. “Modern GPT” here means a decoder assembled from published components used across open models such as [Llama 3](https://arxiv.org/abs/2407.21783), not a reverse-engineered copy of ChatGPT.

Each block uses pre-normalization: [RMSNorm](https://arxiv.org/abs/1910.07467) rescales the current token state before attention and again before the MLP. Queries and keys receive [rotary position embeddings](https://arxiv.org/abs/2104.09864), or RoPE. Attention uses [grouped-query attention](https://arxiv.org/abs/2305.13245), where several query heads share fewer key and value heads. The MLP uses [SwiGLU](https://arxiv.org/abs/2002.05202), a learned gate that controls which hidden features pass through. Input and output embeddings are tied, following the same weight-sharing idea studied by [Press and Wolf](https://arxiv.org/abs/1608.05859).

These names are easy to collect and surprisingly hard to connect. The point of the explorer is to keep the connections on screen.

## Attention you can hold

Try the default prompt, `the cat sat on the`, and open an attention block.

For each token, a query says what this position is looking for. A key says what each earlier token offers. Their dot product becomes an attention score. Softmax turns the scores into weights, and those weights mix the value vectors, which carry the actual payload. The causal mask blocks every token to the right, so the model cannot peek at words it is supposed to predict.

Multi-head attention repeats this in several learned spaces. Here there are four query heads, but only two key/value heads. Query heads 1 and 2 share the first key/value head; query heads 3 and 4 share the second. That is GQA in its smallest nontrivial form. You can see both the separate questions and the shared memory they consult.

The sharing matters during generation. Once a token has passed through a layer, its keys and values can be saved in the KV cache. When the next token arrives, the model reuses them instead of recomputing the whole prefix. Fewer key/value heads means fewer vectors stored for every token in every layer. In a large model and a long conversation, that difference adds up quickly.

RoPE is unusually nice at this scale. Each head has only two numbers, so a query or key is literally a point in a plane. RoPE rotates that point by an angle based on token position before attention compares it with other points. Large models do the same thing across many independent two-dimensional pairs. Here you can watch the basic operation without hiding it behind notation.

## Follow the scratchpad

Attention is only half of a decoder block. Its output gets added to the residual stream, the running eight-number state carried by each token. The MLP then reads a normalized copy of that state, expands it to 16 numbers, gates those numbers with SwiGLU, shrinks them back to eight, and adds another update.

Neither branch replaces the token state. Each writes an update into the same scratchpad. Collapse and expand the layers and you can follow that scratchpad from the initial embedding to the final normalized state.

At the end, the model compares the last token state with every vocabulary embedding. Those dot products become logits, and softmax turns the logits into next-token probabilities. Because the input and output embeddings are tied, the same learned table that gives a word its starting coordinates also helps decide whether that word should come next.

## Did the toy learn anything?

I trained the model for 1,800 steps on 110 sequences from a synthetic five-subject micro-language. Its entire vocabulary has 23 tokens. This is closer to a tiny grammar exercise than English, which is exactly what keeps the model inspectable.

It reached 85.97% next-token accuracy across the training corpus and got all 5 canonical completion checks right. Given `the cat sat on the`, it predicts `mat`; given `the owl waited near the`, it predicts `tree`.

The gap between 85.97% and 100% comes from intentional ambiguous alternatives. Some prefixes have more than one valid next token, while argmax accuracy accepts only the particular target written in that sequence. The model can choose another valid continuation and still be counted wrong. I kept that ambiguity because a perfectly deterministic lookup table would make a cleaner score and a worse language model.

Training matters here. Random weights can show the plumbing, but every attention pattern and output is arbitrary. This checkpoint has learned enough structure that changing a subject, verb, or final noun changes the computation in a coherent way.

## What this helps with

This explorer is useful for building a mechanical model of inference. You can see why attention needs queries, keys, and values; why several heads can look at the same tokens differently; how GQA changes the KV cache; where RoPE enters; what the MLP adds; and how the residual stream connects all four layers.

It does not explain why billion-parameter models gain broad knowledge or new capabilities. It does not show a production tokenizer, distributed training, long-context behavior, mixture-of-experts routing, instruction tuning, RLHF, or serving across many devices. The checkpoint is fixed, so you are watching inference rather than learning. Attention weights are also a computation trace, not a complete account of why a model chose an answer, a distinction made carefully in [“Attention is not Explanation”](https://aclanthology.org/N19-1357/).

The next useful step would be letting the reader perturb a weight or train this tiny model in the browser and watch its behavior change. For now, try breaking its little language and see where the prediction bends.
