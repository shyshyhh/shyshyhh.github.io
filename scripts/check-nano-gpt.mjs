#!/usr/bin/env node

const EXPECTED_CONFIG = Object.freeze({
  layers: 4,
  dModel: 8,
  hidden: 16,
  queryHeads: 4,
  kvHeads: 2,
  headDim: 2,
});

const PROBABILITY_TOLERANCE = 1e-6;
const CACHE_PARITY_TOLERANCE = 1e-12;
const ATTENTION_SUM_TOLERANCE = 1e-10;
const MAGNITUDE_TOLERANCE = 1e-10;

function check(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function pass(message) {
  console.log(`PASS ${message}`);
}

function assertFiniteNumber(value, path) {
  check(
    typeof value === 'number' && Number.isFinite(value),
    `${path} must be a finite number; received ${String(value)}`
  );
}

function assertFiniteTree(value, path) {
  if (typeof value === 'number') {
    assertFiniteNumber(value, path);
    return;
  }

  if (Array.isArray(value)) {
    value.forEach((item, index) => assertFiniteTree(item, `${path}[${index}]`));
    return;
  }

  if (value && typeof value === 'object') {
    Object.entries(value).forEach(([key, item]) =>
      assertFiniteTree(item, `${path}.${key}`)
    );
  }
}

function assertVector(value, length, path) {
  check(Array.isArray(value), `${path} must be an array`);
  check(
    value.length === length,
    `${path} must have length ${length}; received ${value.length}`
  );
  value.forEach((item, index) => assertFiniteNumber(item, `${path}[${index}]`));
}

function assertMatrix(value, rows, columns, path) {
  check(Array.isArray(value), `${path} must be an array`);
  check(
    value.length === rows,
    `${path} must have ${rows} rows; received ${value.length}`
  );
  value.forEach((row, rowIndex) =>
    assertVector(row, columns, `${path}[${rowIndex}]`)
  );
}

function matVec(vector, matrix) {
  const output = Array(matrix[0].length).fill(0);
  for (let row = 0; row < vector.length; row += 1) {
    for (let column = 0; column < output.length; column += 1) {
      output[column] += vector[row] * matrix[row][column];
    }
  }
  return output;
}

function splitHeads(vector, headCount, headDimension) {
  return Array.from({ length: headCount }, (_, head) =>
    vector.slice(head * headDimension, (head + 1) * headDimension)
  );
}

function magnitude(vector) {
  return Math.sqrt(
    vector.reduce((sum, component) => sum + component * component, 0)
  );
}

function assertSameMagnitude(before, after, path) {
  const beforeMagnitude = magnitude(before);
  const afterMagnitude = magnitude(after);
  const tolerance =
    MAGNITUDE_TOLERANCE * Math.max(1, beforeMagnitude, afterMagnitude);
  check(
    Math.abs(beforeMagnitude - afterMagnitude) <= tolerance,
    `${path} changed magnitude from ${beforeMagnitude} to ${afterMagnitude}`
  );
}

function assertDistinctVectors(left, right, path) {
  check(left.length === right.length, `${path} vectors have different lengths`);
  const maximumDifference = Math.max(
    ...left.map((value, index) => Math.abs(value - right[index]))
  );
  check(
    maximumDifference > 1e-9,
    `${path} unexpectedly produced identical final states`
  );
}

function assertClose(left, right, path, tolerance = CACHE_PARITY_TOLERANCE) {
  check(
    Math.abs(left - right) <= tolerance,
    `${path} differs: ${left} vs ${right} (limit ${tolerance})`
  );
}

function assertCloseTree(left, right, path) {
  if (typeof left === 'number' || typeof right === 'number') {
    assertFiniteNumber(left, `${path}.left`);
    assertFiniteNumber(right, `${path}.right`);
    assertClose(left, right, path);
    return;
  }

  check(Array.isArray(left), `${path}.left must be an array`);
  check(Array.isArray(right), `${path}.right must be an array`);
  check(
    left.length === right.length,
    `${path} lengths differ: ${left.length} vs ${right.length}`
  );
  left.forEach((value, index) =>
    assertCloseTree(value, right[index], `${path}[${index}]`)
  );
}

async function main() {
  const {
    MODEL_WEIGHTS,
    MASKED_ATTENTION_SCORE,
    NANO_CONFIG,
    TRAINING_INFO,
    decodeNanoGPTToken,
    formatBytes,
    kvCacheBytes,
    prefillNanoGPT,
    queryHeadToKvHead,
    runNanoGPT,
    tokenizePrompt,
  } = await import('../src/components/gpt-architecture/nano-gpt.mjs');

  for (const [key, expected] of Object.entries(EXPECTED_CONFIG)) {
    check(
      NANO_CONFIG[key] === expected,
      `config.${key} must be exactly ${expected}; received ${NANO_CONFIG[key]}`
    );
  }
  pass('config is 4L / d=8 / hidden=16 / 4Q / 2KV / head_dim=2');

  check(
    JSON.stringify(tokenizePrompt('🤖')) === JSON.stringify(['<unk>']),
    'unsupported-only prompts must map visibly to <unk>'
  );
  check(
    tokenizePrompt('one two three four five six seven eight nine').length ===
      NANO_CONFIG.maxTokens,
    'overlong prompts must stop at the configured context length'
  );
  pass('tokenization exposes unsupported input and enforces the 8-token context');

  const vocabularySize = MODEL_WEIGHTS.vocab.length;
  assertMatrix(
    MODEL_WEIGHTS.embedding,
    vocabularySize,
    EXPECTED_CONFIG.dModel,
    'weights.embedding'
  );
  assertVector(
    MODEL_WEIGHTS.finalNorm,
    EXPECTED_CONFIG.dModel,
    'weights.finalNorm'
  );
  check(
    MODEL_WEIGHTS.layers.length === EXPECTED_CONFIG.layers,
    `weights.layers must contain ${EXPECTED_CONFIG.layers} layers`
  );

  const expectedLayerKeys = [
    'attentionOutput',
    'down',
    'gate',
    'key',
    'normAttention',
    'normMlp',
    'query',
    'up',
    'value',
  ];

  MODEL_WEIGHTS.layers.forEach((layer, layerIndex) => {
    const path = `weights.layers[${layerIndex}]`;
    const actualLayerKeys = Object.keys(layer).sort();
    check(
      JSON.stringify(actualLayerKeys) === JSON.stringify(expectedLayerKeys),
      `${path} parameter keys must be ${expectedLayerKeys.join(', ')}; received ${actualLayerKeys.join(', ')}`
    );

    assertVector(
      layer.normAttention,
      EXPECTED_CONFIG.dModel,
      `${path}.normAttention`
    );
    assertVector(layer.normMlp, EXPECTED_CONFIG.dModel, `${path}.normMlp`);
    assertMatrix(
      layer.query,
      EXPECTED_CONFIG.dModel,
      EXPECTED_CONFIG.queryHeads * EXPECTED_CONFIG.headDim,
      `${path}.query`
    );
    assertMatrix(
      layer.key,
      EXPECTED_CONFIG.dModel,
      EXPECTED_CONFIG.kvHeads * EXPECTED_CONFIG.headDim,
      `${path}.key`
    );
    assertMatrix(
      layer.value,
      EXPECTED_CONFIG.dModel,
      EXPECTED_CONFIG.kvHeads * EXPECTED_CONFIG.headDim,
      `${path}.value`
    );
    assertMatrix(
      layer.attentionOutput,
      EXPECTED_CONFIG.queryHeads * EXPECTED_CONFIG.headDim,
      EXPECTED_CONFIG.dModel,
      `${path}.attentionOutput`
    );
    assertMatrix(
      layer.gate,
      EXPECTED_CONFIG.dModel,
      EXPECTED_CONFIG.hidden,
      `${path}.gate`
    );
    assertMatrix(
      layer.up,
      EXPECTED_CONFIG.dModel,
      EXPECTED_CONFIG.hidden,
      `${path}.up`
    );
    assertMatrix(
      layer.down,
      EXPECTED_CONFIG.hidden,
      EXPECTED_CONFIG.dModel,
      `${path}.down`
    );
  });
  assertFiniteTree(MODEL_WEIGHTS, 'weights');
  pass('all trained parameter shapes and values are valid');

  check(
    Array.isArray(TRAINING_INFO.checks) && TRAINING_INFO.checks.length === 5,
    `training metadata must contain exactly five checks; received ${TRAINING_INFO.checks?.length}`
  );

  let maximumProbabilityDelta = 0;
  const canonicalRuns = TRAINING_INFO.checks.map((trainingCheck, checkIndex) => {
    check(
      trainingCheck.predicted === trainingCheck.expected,
      `training check ${checkIndex + 1} metadata predicts ${trainingCheck.predicted}, expected ${trainingCheck.expected}`
    );

    const result = runNanoGPT(trainingCheck.prompt);
    assertVector(
      result.logits,
      MODEL_WEIGHTS.vocab.length,
      `runs[${checkIndex}].logits`
    );
    assertVector(
      result.probabilities,
      MODEL_WEIGHTS.vocab.length,
      `runs[${checkIndex}].probabilities`
    );
    const topPrediction = result.predictions[0];
    check(
      topPrediction?.token === trainingCheck.expected,
      `"${trainingCheck.prompt}" predicted ${topPrediction?.token}, expected ${trainingCheck.expected}`
    );

    const probabilityDelta = Math.abs(
      topPrediction.probability - trainingCheck.probability
    );
    maximumProbabilityDelta = Math.max(
      maximumProbabilityDelta,
      probabilityDelta
    );
    check(
      probabilityDelta <= PROBABILITY_TOLERANCE,
      `"${trainingCheck.prompt}" probability differs from Python metadata by ${probabilityDelta} (limit ${PROBABILITY_TOLERANCE})`
    );

    assertFiniteTree(result, `runs[${checkIndex}]`);
    return result;
  });
  pass(
    `five canonical predictions match Python metadata (max Δ=${maximumProbabilityDelta.toExponential(2)})`
  );

  canonicalRuns.forEach((result, runIndex) => {
    const tokenCount = result.tokens.length;
    result.layers.forEach((layer, layerIndex) => {
      check(
        layer.attentionWeights.length === EXPECTED_CONFIG.queryHeads,
        `runs[${runIndex}].layers[${layerIndex}] must expose ${EXPECTED_CONFIG.queryHeads} attention heads`
      );
      check(
        layer.attentionScores.length === EXPECTED_CONFIG.queryHeads,
        `runs[${runIndex}].layers[${layerIndex}] must expose ${EXPECTED_CONFIG.queryHeads} attention score heads`
      );

      layer.attentionWeights.forEach((head, headIndex) => {
        assertMatrix(
          layer.attentionScores[headIndex],
          tokenCount,
          tokenCount,
          `runs[${runIndex}].layers[${layerIndex}].attentionScores[${headIndex}]`
        );
        assertMatrix(
          head,
          tokenCount,
          tokenCount,
          `runs[${runIndex}].layers[${layerIndex}].attentionWeights[${headIndex}]`
        );
        head.forEach((row, queryPosition) => {
          const sum = row.reduce((total, value) => total + value, 0);
          check(
            Math.abs(sum - 1) <= ATTENTION_SUM_TOLERANCE,
            `attention row sum is ${sum} at run ${runIndex}, layer ${layerIndex}, head ${headIndex}, query ${queryPosition}`
          );
          row.forEach((weight, keyPosition) => {
            check(
              weight >= 0 && weight <= 1,
              `attention weight ${weight} is outside [0, 1]`
            );
            if (keyPosition > queryPosition) {
              check(
                layer.attentionScores[headIndex][queryPosition][keyPosition] ===
                  MASKED_ATTENTION_SCORE,
                `masked attention score must equal ${MASKED_ATTENTION_SCORE}`
              );
              check(
                weight === 0,
                `causal mask leaked weight ${weight} from query ${queryPosition} to future key ${keyPosition}`
              );
            }
          });
        });
      });
    });
  });
  pass('attention is finite, normalized, and strictly causal');

  /*
   * Raw pre-RoPE Q/K heads are not exported directly. Reconstructing them from
   * normalizedAttention and the exposed projection matrices makes this an
   * indirect norm-preservation check rather than comparing post-RoPE tensors.
   */
  canonicalRuns.forEach((result, runIndex) => {
    result.layers.forEach((layer, layerIndex) => {
      layer.normalizedAttention.forEach((normalized, position) => {
        const rawQueries = splitHeads(
          matVec(normalized, layer.parameters.query),
          EXPECTED_CONFIG.queryHeads,
          EXPECTED_CONFIG.headDim
        );
        const rawKeys = splitHeads(
          matVec(normalized, layer.parameters.key),
          EXPECTED_CONFIG.kvHeads,
          EXPECTED_CONFIG.headDim
        );

        rawQueries.forEach((raw, head) =>
          assertSameMagnitude(
            raw,
            layer.query[position][head],
            `RoPE Q run ${runIndex}, layer ${layerIndex}, position ${position}, head ${head}`
          )
        );
        rawKeys.forEach((raw, head) =>
          assertSameMagnitude(
            raw,
            layer.key[position][head],
            `RoPE K run ${runIndex}, layer ${layerIndex}, position ${position}, head ${head}`
          )
        );
      });
    });
  });
  pass('RoPE preserves every reconstructed Q/K head magnitude');

  const gqaMap = Array.from(
    { length: EXPECTED_CONFIG.queryHeads },
    (_, queryHead) => queryHeadToKvHead(queryHead, EXPECTED_CONFIG.kvHeads)
  );
  check(
    JSON.stringify(gqaMap) === JSON.stringify([0, 0, 1, 1]),
    `GQA mapping must be [0,0,1,1]; received [${gqaMap.join(',')}]`
  );
  pass('GQA mapping is [0,0,1,1]');

  const llama31CacheBytes = kvCacheBytes({
    layers: 32,
    tokens: 8192,
    kvHeads: 8,
    headDim: 128,
    bytesPerElement: 2,
    batch: 1,
  });
  check(
    llama31CacheBytes === 1024 ** 3,
    `Llama 3.1 8B KV cache must be 1 GiB; received ${llama31CacheBytes} bytes`
  );
  check(
    formatBytes(llama31CacheBytes) === '1.00 GiB',
    `1 GiB formatting must be "1.00 GiB"; received "${formatBytes(llama31CacheBytes)}"`
  );
  pass('Llama 3.1 8B 8192-token BF16 KV cache is 1.00 GiB');

  const topTokens = canonicalRuns.map((result) => result.predictions[0].token);
  check(
    new Set(topTokens).size === canonicalRuns.length,
    `canonical prompts must have distinct top predictions; received ${topTokens.join(', ')}`
  );
  for (let left = 0; left < canonicalRuns.length; left += 1) {
    for (let right = left + 1; right < canonicalRuns.length; right += 1) {
      assertDistinctVectors(
        canonicalRuns[left].final.at(-1),
        canonicalRuns[right].final.at(-1),
        `canonical runs ${left} and ${right}`
      );
    }
  }
  pass('canonical prompts produce five distinct model outputs');

  let maximumCacheDelta = 0;
  TRAINING_INFO.checks.forEach((trainingCheck, checkIndex) => {
    const prefill = prefillNanoGPT(trainingCheck.prompt);
    check(prefill.phase === 'prefill', 'prefill phase must be "prefill"');
    check(
      prefill.cache.layers.length === EXPECTED_CONFIG.layers,
      `prefill cache ${checkIndex} must contain ${EXPECTED_CONFIG.layers} layers`
    );

    prefill.cache.layers.forEach((entry, layerIndex) => {
      const tokenCount = prefill.tokens.length;
      check(
        entry.key.length === tokenCount && entry.value.length === tokenCount,
        `prefill cache ${checkIndex}, layer ${layerIndex} must contain ${tokenCount} positions`
      );
      assertCloseTree(
        entry.key,
        prefill.layers[layerIndex].key,
        `prefill cache ${checkIndex}, layer ${layerIndex}.key`
      );
      assertCloseTree(
        entry.value,
        prefill.layers[layerIndex].value,
        `prefill cache ${checkIndex}, layer ${layerIndex}.value`
      );
    });

    const cacheBefore = JSON.stringify(prefill.cache);
    const decoded = decodeNanoGPTToken(
      prefill.cache,
      trainingCheck.expected
    );
    const full = runNanoGPT(
      `${trainingCheck.prompt} ${trainingCheck.expected}`
    );
    check(
      JSON.stringify(prefill.cache) === cacheBefore,
      `decode ${checkIndex} mutated its input cache`
    );
    check(
      decoded.position === prefill.tokens.length,
      `decode ${checkIndex} position must equal prefill length`
    );
    check(
      decoded.cache.tokens.length === prefill.tokens.length + 1,
      `decode ${checkIndex} cache did not grow by one token`
    );

    decoded.layers.forEach((layer, layerIndex) => {
      const fullLayer = full.layers[layerIndex];
      const lastPosition = full.tokens.length - 1;
      assertCloseTree(
        layer.key[0],
        fullLayer.key[lastPosition],
        `decode ${checkIndex}, layer ${layerIndex}.newKey`
      );
      assertCloseTree(
        layer.value[0],
        fullLayer.value[lastPosition],
        `decode ${checkIndex}, layer ${layerIndex}.newValue`
      );
      assertCloseTree(
        layer.output[0],
        fullLayer.output[lastPosition],
        `decode ${checkIndex}, layer ${layerIndex}.output`
      );
      assertCloseTree(
        decoded.cache.layers[layerIndex].key,
        fullLayer.key,
        `decode ${checkIndex}, layer ${layerIndex}.cachedKey`
      );
      assertCloseTree(
        decoded.cache.layers[layerIndex].value,
        fullLayer.value,
        `decode ${checkIndex}, layer ${layerIndex}.cachedValue`
      );

      layer.attentionWeights.forEach((head, headIndex) => {
        check(
          head.length === 1 && head[0].length === full.tokens.length,
          `decode attention ${checkIndex}, layer ${layerIndex}, head ${headIndex} has the wrong shape`
        );
        const sum = head[0].reduce((total, weight) => total + weight, 0);
        assertClose(
          sum,
          1,
          `decode attention ${checkIndex}, layer ${layerIndex}, head ${headIndex} sum`,
          ATTENTION_SUM_TOLERANCE
        );
        assertCloseTree(
          head[0],
          fullLayer.attentionWeights[headIndex][lastPosition],
          `decode attention ${checkIndex}, layer ${layerIndex}, head ${headIndex}`
        );
      });
    });

    assertCloseTree(
      decoded.final[0],
      full.final.at(-1),
      `decode ${checkIndex}.final`
    );
    assertCloseTree(
      decoded.logits,
      full.logits,
      `decode ${checkIndex}.logits`
    );
    assertCloseTree(
      decoded.probabilities,
      full.probabilities,
      `decode ${checkIndex}.probabilities`
    );
    decoded.probabilities.forEach((probability, index) => {
      maximumCacheDelta = Math.max(
        maximumCacheDelta,
        Math.abs(probability - full.probabilities[index])
      );
    });
  });
  pass(
    `incremental KV-cache decode matches five full replays (max Δ=${maximumCacheDelta.toExponential(2)})`
  );

  const chainedPrefill = prefillNanoGPT('the cat sat on the');
  const chainedFirst = decodeNanoGPTToken(chainedPrefill.cache, 'mat');
  const chainedSecond = decodeNanoGPTToken(chainedFirst.cache, '.');
  const chainedFull = runNanoGPT('the cat sat on the mat .');
  assertCloseTree(
    chainedSecond.final[0],
    chainedFull.final.at(-1),
    'two-step cached decode.final'
  );
  assertCloseTree(
    chainedSecond.logits,
    chainedFull.logits,
    'two-step cached decode.logits'
  );
  check(
    chainedSecond.cache.layers.every(
      (entry) =>
        entry.key.length === chainedFull.tokens.length &&
        entry.value.length === chainedFull.tokens.length
    ),
    'two-step cache must contain every prefix position'
  );
  pass('incremental cache chains across two decode steps without replay');
}

main().catch((error) => {
  console.error(`FAIL nano-GPT verification\n${error.stack ?? error}`);
  process.exitCode = 1;
});
