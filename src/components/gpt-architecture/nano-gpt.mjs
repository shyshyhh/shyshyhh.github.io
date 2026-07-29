/*
 * Browser replay of a deliberately tiny trained decoder-only transformer.
 *
 * Every value is inspectable: d_model=8, four 2-D query heads, two KV heads,
 * four layers, and a 16-wide SwiGLU. The weights are trained on the tiny
 * synthetic language in scripts/train-nano-gpt.py.
 */

import TRAINED_WEIGHTS from './nano-gpt-weights.mjs';

export const NANO_CONFIG = Object.freeze(TRAINED_WEIGHTS.config);
export const DEFAULT_PROMPT = 'the cat sat on the';
export const MINI_VOCAB = Object.freeze(TRAINED_WEIGHTS.vocab);
export const TRAINING_INFO = Object.freeze(TRAINED_WEIGHTS.training);
export const MODEL_WEIGHTS = TRAINED_WEIGHTS;
export const MASKED_ATTENTION_SCORE = -1e9;

const EPSILON = 1e-6;

function matVec(vector, matrix) {
  const output = Array(matrix[0].length).fill(0);
  for (let row = 0; row < vector.length; row += 1) {
    for (let column = 0; column < output.length; column += 1) {
      output[column] += vector[row] * matrix[row][column];
    }
  }
  return output;
}

function add(left, right) {
  return left.map((value, index) => value + right[index]);
}

function dot(left, right) {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

function softmax(values) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) =>
    Number.isFinite(value) ? Math.exp(value - maximum) : 0
  );
  const total = exponentials.reduce((sum, value) => sum + value, 0) || 1;
  return exponentials.map((value) => value / total);
}

function magnitude(vector) {
  return Math.sqrt(dot(vector, vector));
}

function rmsNorm(vector, gain) {
  const meanSquare =
    vector.reduce((sum, value) => sum + value * value, 0) / vector.length;
  const denominator = Math.sqrt(meanSquare + EPSILON);
  return vector.map((value, index) => (value / denominator) * gain[index]);
}

function silu(value) {
  return value / (1 + Math.exp(-value));
}

function splitHeads(vector, headCount, headDimension) {
  return Array.from({ length: headCount }, (_, head) =>
    vector.slice(head * headDimension, (head + 1) * headDimension)
  );
}

function rotatePair(vector, angle) {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [
    vector[0] * cosine - vector[1] * sine,
    vector[0] * sine + vector[1] * cosine,
  ];
}

function applyRope(head, position, base = NANO_CONFIG.ropeBase) {
  const rotated = [];
  for (let pair = 0; pair < head.length / 2; pair += 1) {
    const frequency = base ** (-(2 * pair) / head.length);
    const values = head.slice(pair * 2, pair * 2 + 2);
    rotated.push(...rotatePair(values, position * frequency));
  }
  return rotated;
}

function tokenEmbedding(token) {
  const normalized = token.toLowerCase();
  const index = MINI_VOCAB.indexOf(normalized);
  const unknownIndex = MINI_VOCAB.indexOf('<unk>');
  return [...TRAINED_WEIGHTS.embedding[index >= 0 ? index : unknownIndex]];
}

function outputDistribution(lastState) {
  const logits = MINI_VOCAB.map((token) =>
    dot(lastState, tokenEmbedding(token))
  );
  const probabilities = softmax(logits);
  const predictions = MINI_VOCAB.map((token, index) => ({
    token,
    logit: logits[index],
    probability: probabilities[index],
  }))
    .filter(({ token }) => token !== '<unk>')
    .sort((left, right) => right.probability - left.probability);

  return { logits, probabilities, predictions };
}

function cloneCacheTensor(tensor) {
  return tensor.map((position) =>
    position.map((head) => [...head])
  );
}

function cacheFromReplay(result) {
  return {
    kind: 'nano-gpt-kv-cache',
    version: 1,
    tokens: [...result.tokens],
    layers: result.layers.map((layer) => ({
      key: cloneCacheTensor(layer.key),
      value: cloneCacheTensor(layer.value),
    })),
  };
}

function normalizeDecodeToken(token) {
  if (typeof token !== 'string') {
    throw new TypeError('The decoded token must be a string.');
  }

  const normalized = token.trim().toLowerCase();
  if (MINI_VOCAB.includes(normalized)) return normalized;

  const matches = normalized.match(/[a-z0-9]+(?:'[a-z0-9]+)?|[.,!?;:]/g);
  if (matches?.length !== 1 || matches[0] !== normalized) {
    throw new Error(
      `decodeNanoGPTToken expects exactly one token; received "${token}".`
    );
  }
  return normalized;
}

function assertCacheHead(head, path) {
  if (!Array.isArray(head) || head.length !== NANO_CONFIG.headDim) {
    throw new Error(
      `${path} must have shape [${NANO_CONFIG.headDim}].`
    );
  }
  head.forEach((value, dimension) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new Error(`${path}[${dimension}] must be a finite number.`);
    }
  });
}

function assertCacheTensor(tensor, tokenCount, path) {
  if (!Array.isArray(tensor) || tensor.length !== tokenCount) {
    throw new Error(
      `${path} must have shape [${tokenCount}, ${NANO_CONFIG.kvHeads}, ${NANO_CONFIG.headDim}].`
    );
  }
  tensor.forEach((position, positionIndex) => {
    if (!Array.isArray(position) || position.length !== NANO_CONFIG.kvHeads) {
      throw new Error(
        `${path}[${positionIndex}] must contain ${NANO_CONFIG.kvHeads} KV heads.`
      );
    }
    position.forEach((head, headIndex) =>
      assertCacheHead(head, `${path}[${positionIndex}][${headIndex}]`)
    );
  });
}

function assertNanoGPTCache(cache) {
  if (
    !cache ||
    cache.kind !== 'nano-gpt-kv-cache' ||
    cache.version !== 1
  ) {
    throw new Error('Expected a version 1 nano-GPT KV cache.');
  }
  if (!Array.isArray(cache.tokens) || cache.tokens.length === 0) {
    throw new Error('The KV cache must contain at least one token.');
  }
  if (
    !Array.isArray(cache.layers) ||
    cache.layers.length !== NANO_CONFIG.layers
  ) {
    throw new Error(
      `The KV cache must contain ${NANO_CONFIG.layers} layers.`
    );
  }

  const tokenCount = cache.tokens.length;
  cache.layers.forEach((layer, layerIndex) => {
    if (!layer || typeof layer !== 'object') {
      throw new Error(`cache.layers[${layerIndex}] must be an object.`);
    }
    assertCacheTensor(layer.key, tokenCount, `cache.layers[${layerIndex}].key`);
    assertCacheTensor(
      layer.value,
      tokenCount,
      `cache.layers[${layerIndex}].value`
    );
  });

  return tokenCount;
}

export function tokenizePrompt(prompt) {
  const normalized = prompt.trim().toLowerCase();
  const matches = normalized.match(
    /[a-z0-9]+(?:'[a-z0-9]+)?|[.,!?;:]/g
  );
  const tokens = matches?.length
    ? matches
    : normalized
      ? ['<unk>']
      : DEFAULT_PROMPT.split(' ');

  return tokens.slice(0, NANO_CONFIG.maxTokens);
}

export function queryHeadToKvHead(queryHead, kvHeads = NANO_CONFIG.kvHeads) {
  const groupSize = NANO_CONFIG.queryHeads / kvHeads;
  return Math.floor(queryHead / groupSize);
}

export function runNanoGPT(prompt = DEFAULT_PROMPT) {
  const tokens = tokenizePrompt(prompt);
  let residual = tokens.map(tokenEmbedding);
  const layers = [];

  for (let layerIndex = 0; layerIndex < NANO_CONFIG.layers; layerIndex += 1) {
    const weights = TRAINED_WEIGHTS.layers[layerIndex];
    const input = residual.map((vector) => [...vector]);
    const normalizedAttention = input.map((vector) =>
      rmsNorm(vector, weights.normAttention)
    );

    const query = normalizedAttention.map((vector, position) =>
      splitHeads(
        matVec(vector, weights.query),
        NANO_CONFIG.queryHeads,
        NANO_CONFIG.headDim
      ).map((head) => applyRope(head, position))
    );
    const key = normalizedAttention.map((vector, position) =>
      splitHeads(
        matVec(vector, weights.key),
        NANO_CONFIG.kvHeads,
        NANO_CONFIG.headDim
      ).map((head) => applyRope(head, position))
    );
    const value = normalizedAttention.map((vector) =>
      splitHeads(
        matVec(vector, weights.value),
        NANO_CONFIG.kvHeads,
        NANO_CONFIG.headDim
      )
    );

    const attentionWeights = Array.from(
      { length: NANO_CONFIG.queryHeads },
      () => Array.from({ length: tokens.length }, () => Array(tokens.length).fill(0))
    );
    const attentionScores = Array.from(
      { length: NANO_CONFIG.queryHeads },
      () =>
        Array.from(
          { length: tokens.length },
          () => Array(tokens.length).fill(MASKED_ATTENTION_SCORE)
        )
    );

    const attentionHeads = tokens.map((_, queryPosition) =>
      Array.from({ length: NANO_CONFIG.queryHeads }, (_, queryHead) => {
        const kvHead = queryHeadToKvHead(queryHead);
        const scores = tokens.map((__, keyPosition) => {
          if (keyPosition > queryPosition) return MASKED_ATTENTION_SCORE;
          return (
            dot(query[queryPosition][queryHead], key[keyPosition][kvHead]) /
            Math.sqrt(NANO_CONFIG.headDim)
          );
        });
        const probabilities = softmax(scores);
        attentionScores[queryHead][queryPosition] = scores;
        attentionWeights[queryHead][queryPosition] = probabilities;

        const mixed = Array(NANO_CONFIG.headDim).fill(0);
        probabilities.forEach((probability, keyPosition) => {
          for (
            let dimension = 0;
            dimension < NANO_CONFIG.headDim;
            dimension += 1
          ) {
            mixed[dimension] +=
              probability * value[keyPosition][kvHead][dimension];
          }
        });
        return mixed;
      })
    );

    const attentionUpdate = attentionHeads.map((heads) =>
      matVec(heads.flat(), weights.attentionOutput)
    );
    const afterAttention = input.map((vector, position) =>
      add(vector, attentionUpdate[position])
    );

    const normalizedMlp = afterAttention.map((vector) =>
      rmsNorm(vector, weights.normMlp)
    );
    const gate = normalizedMlp.map((vector) => matVec(vector, weights.gate));
    const up = normalizedMlp.map((vector) => matVec(vector, weights.up));
    const swiglu = gate.map((gateVector, position) =>
      gateVector.map((value, dimension) => silu(value) * up[position][dimension])
    );
    const mlpUpdate = swiglu.map((vector) => matVec(vector, weights.down));
    const output = afterAttention.map((vector, position) =>
      add(vector, mlpUpdate[position])
    );

    layers.push({
      input,
      normalizedAttention,
      query,
      key,
      value,
      attentionScores,
      attentionWeights,
      attentionHeads,
      attentionUpdate,
      afterAttention,
      normalizedMlp,
      gate,
      up,
      swiglu,
      mlpUpdate,
      output,
      norms: {
        input: input.map(magnitude),
        attentionUpdate: attentionUpdate.map(magnitude),
        afterAttention: afterAttention.map(magnitude),
        mlpUpdate: mlpUpdate.map(magnitude),
        output: output.map(magnitude),
      },
      parameters: weights,
    });

    residual = output;
  }

  const final = residual.map((vector) =>
    rmsNorm(vector, TRAINED_WEIGHTS.finalNorm)
  );
  const lastState = final.at(-1);
  const { logits, probabilities, predictions } =
    outputDistribution(lastState);

  return {
    config: NANO_CONFIG,
    prompt,
    tokens,
    embeddings: tokens.map(tokenEmbedding),
    layers,
    final,
    logits,
    probabilities,
    predictions,
    parameters: TRAINED_WEIGHTS,
    training: TRAINING_INFO,
  };
}

/**
 * Run the full prompt once and retain the RoPE-rotated K/V tensors needed for
 * incremental decoding. Cache layer tensors have shape [T, KV heads, head dim].
 */
export function prefillNanoGPT(prompt = DEFAULT_PROMPT) {
  const result = runNanoGPT(prompt);
  const cache = cacheFromReplay(result);

  return {
    phase: 'prefill',
    ...result,
    cache,
    cacheEntries: cache.layers,
  };
}

/**
 * Apply one token without replaying prior tokens. Returned layer tensors keep
 * runNanoGPT's token axis, but contain only the new token:
 * Q [1, Q heads, head dim], K/V [1, KV heads, head dim], and attention
 * [Q heads, 1, total cached tokens].
 */
export function decodeNanoGPTToken(cache, token) {
  const position = assertNanoGPTCache(cache);
  if (position >= NANO_CONFIG.maxTokens) {
    throw new Error(
      `The nano-GPT context is full (${NANO_CONFIG.maxTokens} tokens).`
    );
  }

  const decodedToken = normalizeDecodeToken(token);
  const embedding = tokenEmbedding(decodedToken);
  let residual = [...embedding];
  const layers = [];
  const cacheLayers = [];

  for (let layerIndex = 0; layerIndex < NANO_CONFIG.layers; layerIndex += 1) {
    const weights = TRAINED_WEIGHTS.layers[layerIndex];
    const input = [...residual];
    const normalizedAttention = rmsNorm(input, weights.normAttention);
    const query = splitHeads(
      matVec(normalizedAttention, weights.query),
      NANO_CONFIG.queryHeads,
      NANO_CONFIG.headDim
    ).map((head) => applyRope(head, position));
    const key = splitHeads(
      matVec(normalizedAttention, weights.key),
      NANO_CONFIG.kvHeads,
      NANO_CONFIG.headDim
    ).map((head) => applyRope(head, position));
    const value = splitHeads(
      matVec(normalizedAttention, weights.value),
      NANO_CONFIG.kvHeads,
      NANO_CONFIG.headDim
    );

    const cachedKey = [
      ...cloneCacheTensor(cache.layers[layerIndex].key),
      key.map((head) => [...head]),
    ];
    const cachedValue = [
      ...cloneCacheTensor(cache.layers[layerIndex].value),
      value.map((head) => [...head]),
    ];
    cacheLayers.push({ key: cachedKey, value: cachedValue });

    const attentionScores = Array.from(
      { length: NANO_CONFIG.queryHeads },
      (_, queryHead) => {
        const kvHead = queryHeadToKvHead(queryHead);
        return cachedKey.map(
          (cachedHead) =>
            dot(query[queryHead], cachedHead[kvHead]) /
            Math.sqrt(NANO_CONFIG.headDim)
        );
      }
    );
    const attentionRows = attentionScores.map(softmax);
    const attentionHeads = attentionRows.map(
      (probabilities, queryHead) => {
        const kvHead = queryHeadToKvHead(queryHead);
        const mixed = Array(NANO_CONFIG.headDim).fill(0);
        probabilities.forEach((probability, keyPosition) => {
          for (
            let dimension = 0;
            dimension < NANO_CONFIG.headDim;
            dimension += 1
          ) {
            mixed[dimension] +=
              probability * cachedValue[keyPosition][kvHead][dimension];
          }
        });
        return mixed;
      }
    );

    const attentionUpdate = matVec(
      attentionHeads.flat(),
      weights.attentionOutput
    );
    const afterAttention = add(input, attentionUpdate);
    const normalizedMlp = rmsNorm(afterAttention, weights.normMlp);
    const gate = matVec(normalizedMlp, weights.gate);
    const up = matVec(normalizedMlp, weights.up);
    const swiglu = gate.map(
      (gateValue, dimension) => silu(gateValue) * up[dimension]
    );
    const mlpUpdate = matVec(swiglu, weights.down);
    const output = add(afterAttention, mlpUpdate);

    layers.push({
      input: [input],
      normalizedAttention: [normalizedAttention],
      query: [query],
      key: [key],
      value: [value],
      attentionScores: attentionScores.map((row) => [row]),
      attentionWeights: attentionRows.map((row) => [row]),
      attentionHeads: [attentionHeads],
      attentionUpdate: [attentionUpdate],
      afterAttention: [afterAttention],
      normalizedMlp: [normalizedMlp],
      gate: [gate],
      up: [up],
      swiglu: [swiglu],
      mlpUpdate: [mlpUpdate],
      output: [output],
      norms: {
        input: [magnitude(input)],
        attentionUpdate: [magnitude(attentionUpdate)],
        afterAttention: [magnitude(afterAttention)],
        mlpUpdate: [magnitude(mlpUpdate)],
        output: [magnitude(output)],
      },
      parameters: weights,
    });

    residual = output;
  }

  const finalState = rmsNorm(residual, TRAINED_WEIGHTS.finalNorm);
  const { logits, probabilities, predictions } =
    outputDistribution(finalState);
  const tokens = [...cache.tokens, decodedToken];
  const nextCache = {
    kind: 'nano-gpt-kv-cache',
    version: 1,
    tokens,
    layers: cacheLayers,
  };

  return {
    phase: 'decode',
    config: NANO_CONFIG,
    prompt: tokens.join(' '),
    token: decodedToken,
    position,
    tokens,
    embeddings: [embedding],
    layers,
    final: [finalState],
    logits,
    probabilities,
    predictions,
    cache: nextCache,
    cacheEntries: nextCache.layers,
    parameters: TRAINED_WEIGHTS,
    training: TRAINING_INFO,
  };
}

export function kvCacheBytes({
  layers = 32,
  tokens = 8192,
  kvHeads = 8,
  headDim = 128,
  bytesPerElement = 2,
  batch = 1,
} = {}) {
  return (
    2 * batch * layers * tokens * kvHeads * headDim * bytesPerElement
  );
}

export function formatBytes(bytes) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = value >= 100 || unit === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

export function vectorMagnitude(vector) {
  return magnitude(vector);
}
