import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@radix-ui/themes/components/button';
import * as SegmentedControl from '@radix-ui/themes/components/segmented-control';
import * as TextField from '@radix-ui/themes/components/text-field';
import GPTScene, { HEAD_COLORS } from './GPTScene.jsx';
import TensorInspector from './TensorInspector.jsx';
import {
  DEFAULT_PROMPT,
  MASKED_ATTENTION_SCORE,
  MODEL_WEIGHTS,
  NANO_CONFIG,
  TRAINING_INFO,
  decodeNanoGPTToken,
  formatBytes,
  kvCacheBytes,
  prefillNanoGPT,
  queryHeadToKvHead,
  tokenizePrompt,
} from './nano-gpt.mjs';
import './gpt-explorer.css';

const MODES = [
  { id: 'stack', label: 'Stack', color: '#2563eb' },
  { id: 'attention', label: 'Attention', color: '#7c3aed' },
  { id: 'rope', label: 'RoPE', color: '#4f46e5' },
  { id: 'gqa', label: 'GQA', color: '#b45309' },
  { id: 'cache', label: 'KV cache', color: '#047857' },
  { id: 'mlp', label: 'SwiGLU', color: '#d14f3f' },
  { id: 'weights', label: 'Weights', color: '#2563eb' },
];

const MODE_COPY = {
  stack: {
    eyebrow: 'THE WHOLE MACHINE',
    title: 'A residual stream with two kinds of updates',
    summary:
      'Each block reads a normalized copy, lets attention move information between tokens, lets SwiGLU transform each token, then adds both updates back.',
    formula: 'x ← x + Attn(RMSNorm(x));  x ← x + MLP(RMSNorm(x))',
  },
  attention: {
    eyebrow: 'TOKEN MIXING',
    title: 'Four heads, four different mixtures',
    summary:
      'A query matches allowed keys. Softmax turns those matches into weights, and the weights mix values. Every number below comes from the selected trained layer.',
    formula: 'softmax((QKᵀ / √d_head) + causal mask) · V',
  },
  rope: {
    eyebrow: 'POSITION AS PHASE',
    title: 'Rotate before comparing',
    summary:
      'RoPE rotates every 2-D pair in Q and K by a position-dependent angle. Equal relative offsets create equal relative rotations. V is left alone.',
    formula: 'q′ₘ = R(mω)q;  k′ₙ = R(nω)k',
  },
  gqa: {
    eyebrow: 'SHARED MEMORY HEADS',
    title: 'Many questions, fewer memories',
    summary:
      'Query heads stay independent. GQA only shares the key/value heads they consult, reducing the tensors that generation must keep and read.',
    formula: 'KV head(h) = floor(h / (Hq / Hkv))',
  },
  cache: {
    eyebrow: 'INCREMENTAL DECODING',
    title: 'Compute the prompt once',
    summary:
      'Prefill writes one K and V per prompt token into every layer. Decode appends one newcomer, whose query scans the saved keys and mixes the saved values.',
    formula: 'cache elements = 2 · L · T · Hkv · d_head',
  },
  mlp: {
    eyebrow: 'TOKEN-WISE COMPUTE',
    title: 'A learned signed gate',
    summary:
      'One projection proposes features. Another passes through SiLU and gates them element by element. A final projection writes the update back to eight dimensions.',
    formula: 'Wdown(SiLU(xWgate) ⊙ (xWup))',
  },
  weights: {
    eyebrow: 'NO HIDDEN WALL OF NUMBERS',
    title: 'Every trained parameter fits',
    summary:
      'Pick any projection in any layer. Positive weights rise; negative weights drop. Click a cell to read the exact value used by the live forward pass.',
    formula: 'largest block projection: 8 × 16 = 128 trainable numbers',
  },
};

const TOUR = [
  {
    mode: 'stack',
    title: 'Start with the scratchpad',
    body: 'Each token owns an 8-number residual stream. Open the stack and follow one token through all four blocks.',
    exploded: true,
  },
  {
    mode: 'attention',
    title: 'Split one query four ways',
    body: 'The selected token forms four different 2-D queries. Pick a head and watch its causal mixture change.',
  },
  {
    mode: 'rope',
    title: 'Give the heads position',
    body: 'The two numbers in a head make a real plane, so the rotary position operation is a literal rotation.',
  },
  {
    mode: 'gqa',
    title: 'Share only K and V',
    body: 'Switch between MHA, GQA, and MQA. The query count stays fixed while the memory count shrinks.',
  },
  {
    mode: 'cache',
    title: 'Keep the reusable work',
    body: 'Run prefill, then generate one token at a time. Old queries disappear; old keys and values stay.',
  },
  {
    mode: 'mlp',
    title: 'Gate features inside each token',
    body: 'Attention has mixed tokens. SwiGLU now works on each token independently and writes another residual update.',
  },
  {
    mode: 'weights',
    title: 'Nothing up the sleeve',
    body: 'The tiny dimensions let the full learned matrices remain clickable. This is the exact checkpoint, not a sketch.',
  },
];

const PARAMETER_OPTIONS = [
  { id: 'query', label: 'Wq · query', color: '#168aa1', kind: 'parameter' },
  { id: 'key', label: 'Wk · key', color: '#765ba7', kind: 'parameter' },
  { id: 'value', label: 'Wv · value', color: '#25896f', kind: 'parameter' },
  { id: 'attentionOutput', label: 'Wo · attention out', color: '#4f6fae', kind: 'parameter' },
  { id: 'gate', label: 'Wgate · SwiGLU gate', color: '#765ba7', kind: 'parameter' },
  { id: 'up', label: 'Wup · SwiGLU content', color: '#168aa1', kind: 'parameter' },
  { id: 'down', label: 'Wdown · MLP out', color: '#cf6258', kind: 'parameter' },
  { id: 'normAttention', label: 'RMS gain · attention', color: '#4f6fae', kind: 'parameter' },
  { id: 'normMlp', label: 'RMS gain · MLP', color: '#4f6fae', kind: 'parameter' },
  { id: 'embedding', label: 'tied embedding table', color: '#b9811f', kind: 'parameter' },
  { id: 'finalNorm', label: 'final RMS gain', color: '#25896f', kind: 'parameter' },
];

const ACTIVATION_OPTIONS = [
  { id: 'actEmbeddings', label: 'input embeddings', color: '#b9811f', kind: 'activation' },
  { id: 'actInput', label: 'block input / residual', color: '#168aa1', kind: 'activation' },
  { id: 'actNormAttention', label: 'RMSNorm before attention', color: '#4f6fae', kind: 'activation' },
  { id: 'actQuery', label: 'Q · all query heads', color: '#168aa1', kind: 'activation' },
  { id: 'actKey', label: 'K · all KV heads', color: '#765ba7', kind: 'activation' },
  { id: 'actValue', label: 'V · all KV heads', color: '#25896f', kind: 'activation' },
  { id: 'actAttentionScores', label: 'attention scores · selected head', color: '#765ba7', kind: 'activation' },
  { id: 'actAttentionWeights', label: 'attention weights · selected head', color: '#765ba7', kind: 'activation' },
  { id: 'actAttentionHeads', label: 'mixed head outputs', color: '#b9811f', kind: 'activation' },
  { id: 'actAttentionUpdate', label: 'attention residual update', color: '#4f6fae', kind: 'activation' },
  { id: 'actAfterAttention', label: 'residual after attention', color: '#168aa1', kind: 'activation' },
  { id: 'actNormMlp', label: 'RMSNorm before MLP', color: '#4f6fae', kind: 'activation' },
  { id: 'actGate', label: 'SwiGLU gate preactivation', color: '#765ba7', kind: 'activation' },
  { id: 'actUp', label: 'SwiGLU content projection', color: '#168aa1', kind: 'activation' },
  { id: 'actSwiglu', label: 'SwiGLU gated features', color: '#b9811f', kind: 'activation' },
  { id: 'actMlpUpdate', label: 'MLP residual update', color: '#cf6258', kind: 'activation' },
  { id: 'actOutput', label: 'block output / residual', color: '#25896f', kind: 'activation' },
  { id: 'actFinal', label: 'final RMS-normalized state', color: '#25896f', kind: 'activation' },
  { id: 'actLogits', label: 'last-token vocabulary logits', color: '#b9811f', kind: 'activation' },
  { id: 'actProbabilities', label: 'last-token vocabulary probabilities', color: '#25896f', kind: 'activation' },
];

const TENSOR_OPTIONS = [...PARAMETER_OPTIONS, ...ACTIVATION_OPTIONS];

const PRESETS = [
  'the cat sat on the',
  'the dog slept on the',
  'the fox ran under the',
  'the owl waited near the',
  'the robot worked with the',
];

function rawPromptTokens(prompt) {
  return (
    prompt
      .trim()
      .toLowerCase()
      .match(/[a-z0-9]+(?:'[a-z0-9]+)?|[.,!?;:]/g) ?? []
  );
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(media.matches);
    update();
    media.addEventListener?.('change', update);
    return () => media.removeEventListener?.('change', update);
  }, []);

  return reduced;
}

function supportsWebGL() {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(
      window.WebGLRenderingContext &&
        (canvas.getContext('webgl2') || canvas.getContext('webgl'))
    );
  } catch {
    return false;
  }
}

function signed(value, digits = 3) {
  if (!Number.isFinite(value)) return '—';
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
}

function percent(value) {
  if (value >= 0.9995) return '>99.9%';
  if (value < 0.001) return '<0.1%';
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

function norm(vector) {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

function flattenHeads(tensor) {
  return tensor.map((token) => token.flat());
}

function PanelSection({ label, children, className = '' }) {
  return (
    <section className={`gptx-panel-section ${className}`}>
      <p className="gptx-panel-label">{label}</p>
      {children}
    </section>
  );
}

function DimensionStrip() {
  return (
    <div className="gptx-dimensions" aria-label="Nano GPT dimensions">
      {[
        ['layers', '4'],
        ['d model', '8'],
        ['Q / KV', '4 / 2'],
        ['head', '2'],
        ['MLP', '16'],
      ].map(([label, value]) => (
        <div className="gptx-dimension" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function PredictionBars({ predictions }) {
  const visible = predictions.slice(0, 4);
  const maximum = visible[0]?.probability || 1;
  return (
    <div className="gptx-bars" aria-label="Top next-token predictions">
      {visible.map((prediction, index) => (
        <div className="gptx-bar" key={prediction.token}>
          <span className="gptx-bar-rank">0{index + 1}</span>
          <span className="gptx-bar-label">
            {prediction.token === '<eos>' ? '⟨eos⟩' : prediction.token}
          </span>
          <span className="gptx-bar-track" aria-hidden="true">
            <span
              className="gptx-bar-fill"
              style={{
                '--gptx-bar-width': `${Math.max(
                  2,
                  (prediction.probability / maximum) * 100
                )}%`,
              }}
            />
          </span>
          <span className="gptx-bar-value">{percent(prediction.probability)}</span>
        </div>
      ))}
    </div>
  );
}

function VectorStrip({ label, values, color = '#168aa1', limit }) {
  const shown = values.slice(0, limit ?? values.length);
  const maximum = Math.max(...shown.map((value) => Math.abs(value)), 0.001);
  return (
    <div className="gptx-vector">
      <span className="gptx-vector-label">{label}</span>
      <div className="gptx-vector-cells">
        {shown.map((value, index) => (
          <span
            key={`${label}-${index}`}
            className={`gptx-vector-cell ${
              value < 0 ? 'gptx-vector-cell--negative' : ''
            }`}
            style={{
              '--gptx-vector-strength': Math.abs(value) / maximum,
              '--gptx-vector-color': color,
            }}
            title={`${label}[${index}] = ${value}`}
          >
            {signed(value, 2)}
          </span>
        ))}
      </div>
    </div>
  );
}

function AttentionGrid({
  model,
  selectedLayer,
  selectedHead,
  selectedToken,
  onTokenSelect,
}) {
  const weights = model.layers[selectedLayer].attentionWeights[selectedHead];
  return (
    <div className="gptx-attention-grid-wrap">
      <div
        className="gptx-attention-grid"
        style={{ '--gptx-token-count': model.tokens.length }}
        aria-label={`Attention weights for layer ${selectedLayer + 1}, head ${
          selectedHead + 1
        }`}
      >
        <span />
        {model.tokens.map((token, index) => (
          <span className="gptx-attention-axis" key={`column-${token}-${index}`}>
            {token.slice(0, 3)}
          </span>
        ))}
        {weights.map((row, rowIndex) => (
          <React.Fragment key={`row-${rowIndex}`}>
            <button
              type="button"
              className={`gptx-attention-row ${
                rowIndex === selectedToken ? 'is-selected' : ''
              }`}
              onClick={() => onTokenSelect(rowIndex)}
              aria-label={`Select query token ${model.tokens[rowIndex]}`}
            >
              {model.tokens[rowIndex].slice(0, 4)}
            </button>
            {row.map((value, columnIndex) => {
              const masked = columnIndex > rowIndex;
              return (
                <button
                  type="button"
                  key={`${rowIndex}-${columnIndex}`}
                  className={`gptx-attention-cell ${
                    masked ? 'is-masked' : ''
                  } ${rowIndex === selectedToken ? 'is-query-row' : ''}`}
                  style={{ '--gptx-attention': value }}
                  onClick={() => onTokenSelect(rowIndex)}
                  aria-label={`${model.tokens[rowIndex]} attends to ${
                    model.tokens[columnIndex]
                  }: ${masked ? 'masked' : percent(value)}`}
                  title={
                    masked
                      ? 'future token masked'
                      : `${model.tokens[rowIndex]} → ${model.tokens[columnIndex]}: ${percent(value)}`
                  }
                >
                  <span>{masked ? '×' : Math.round(value * 100)}</span>
                </button>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function Segmented({ label, options, value, onChange }) {
  return (
    <SegmentedControl.Root
      className="gptx-segmented"
      value={value}
      onValueChange={(nextValue) => nextValue && onChange(nextValue)}
      size="3"
      aria-label={label}
    >
      {options.map((option) => (
        <SegmentedControl.Item
          key={option.value}
          value={option.value}
        >
          {option.label}
        </SegmentedControl.Item>
      ))}
    </SegmentedControl.Root>
  );
}

function MatrixSelection({
  matrixId,
  onMatrixChange,
  selectedLayer,
  matrixInfo,
  selectedCell,
  onCellSelect,
}) {
  return (
    <>
      <label className="gptx-select-label" htmlFor="gptx-matrix-select">
        inspect tensor
      </label>
      <select
        id="gptx-matrix-select"
        className="gptx-select"
        value={matrixId}
        onChange={(event) => onMatrixChange(event.target.value)}
      >
        <optgroup label="Trained parameters">
          {PARAMETER_OPTIONS.map((option) => (
            <option value={option.id} key={option.id}>
              {option.label}
            </option>
          ))}
        </optgroup>
        <optgroup label="Live activations">
          {ACTIVATION_OPTIONS.map((option) => (
            <option value={option.id} key={option.id}>
              {option.label}
            </option>
          ))}
        </optgroup>
      </select>
      <TensorInspector
        title={matrixInfo.label}
        subtitle={
          matrixId === 'embedding'
            ? '23-token table tied to the output classifier'
            : matrixId === 'finalNorm'
              ? 'shared after the final block'
              : matrixInfo.kind === 'activation'
                ? matrixInfo.scope
                : `trained layer ${selectedLayer + 1}`
        }
        matrix={matrixInfo.matrix}
        rowLabels={matrixInfo.rowLabels}
        columnLabels={matrixInfo.columnLabels}
        color={matrixInfo.color}
        onCellSelect={onCellSelect}
      />
      {selectedCell && (
        <p className="gptx-cell-readout">
          3D cell [{selectedCell.row}, {selectedCell.column}] ={' '}
          <code>{selectedCell.value}</code>
        </p>
      )}
    </>
  );
}

function Inspector({
  mode,
  model,
  selectedLayer,
  selectedHead,
  selectedToken,
  onTokenSelect,
  variant,
  onVariantChange,
  contextLength,
  onContextLengthChange,
  decodeStep,
  maxDecodeStep,
  cacheState,
  cacheTokens,
  playing,
  onPrefill,
  onNextToken,
  onPlayToggle,
  matrixId,
  onMatrixChange,
  matrixInfo,
  selectedCell,
  onCellSelect,
}) {
  const baseCopy = MODE_COPY[mode];
  const copy =
    mode === 'weights' && matrixInfo.kind === 'activation'
      ? {
          eyebrow: 'THE LIVE FORWARD PASS',
          title: 'Every major activation tensor fits',
          summary:
            'Switch from parameters to the values they produce. Token-indexed tensors use one row per token; head dimensions flatten into adjacent columns. Click any cell for its exact number.',
          formula: `${matrixInfo.rows} × ${matrixInfo.columns} = ${
            matrixInfo.rows * matrixInfo.columns
          } live values in the selected tensor`,
        }
      : baseCopy;
  const layer = model.layers[selectedLayer];
  const kvHead = queryHeadToKvHead(selectedHead);
  const selectedCacheLayer = cacheState.layers[selectedLayer];
  const cachedKeys = selectedCacheLayer.key.map((token) => token[kvHead]);
  const cachedValues = selectedCacheLayer.value.map((token) => token[kvHead]);
  const selectedAttention =
    layer.attentionWeights[selectedHead][selectedToken] ?? [];
  const gqaHeads = variant === 'mha' ? 4 : variant === 'mqa' ? 1 : 2;
  const gqaBytes = kvCacheBytes({
    layers: 4,
    tokens: contextLength,
    kvHeads: gqaHeads,
    headDim: 2,
  });
  const mhaBytes = kvCacheBytes({ tokens: contextLength, kvHeads: 32 });
  const modernGqaBytes = kvCacheBytes({ tokens: contextLength, kvHeads: 8 });

  return (
    <aside className="gptx-inspector">
      <div className="gptx-inspector-heading">
        <p className="gptx-eyebrow" style={{ '--gptx-mode': MODES.find((item) => item.id === mode)?.color }}>
          {copy.eyebrow}
        </p>
        <h2>{copy.title}</h2>
        <p>{copy.summary}</p>
      </div>

      <code className="gptx-formula">{copy.formula}</code>

      {mode === 'stack' && (
        <>
          <PanelSection label="nano checkpoint">
            <DimensionStrip />
          </PanelSection>
          <PanelSection label={`next token after “${model.tokens.at(-1)}”`}>
            <PredictionBars predictions={model.predictions} />
          </PanelSection>
          <PanelSection label={`layer ${selectedLayer + 1} · “${model.tokens[selectedToken]}”`}>
            <div className="gptx-delta">
              <span>
                attention update
                <strong>{norm(layer.attentionUpdate[selectedToken]).toFixed(3)}</strong>
              </span>
              <span>
                MLP update
                <strong>{norm(layer.mlpUpdate[selectedToken]).toFixed(3)}</strong>
              </span>
              <span>
                residual out
                <strong>{norm(layer.output[selectedToken]).toFixed(3)}</strong>
              </span>
            </div>
          </PanelSection>
        </>
      )}

      {mode === 'attention' && (
        <>
          <PanelSection label="causal attention map · values are %">
            <AttentionGrid
              model={model}
              selectedLayer={selectedLayer}
              selectedHead={selectedHead}
              selectedToken={selectedToken}
              onTokenSelect={onTokenSelect}
            />
          </PanelSection>
          <PanelSection label={`head ${selectedHead + 1} · query “${model.tokens[selectedToken]}”`}>
            <VectorStrip
              label="Q"
              values={layer.query[selectedToken][selectedHead]}
              color={HEAD_COLORS[selectedHead]}
            />
            <VectorStrip
              label={`K${kvHead + 1}`}
              values={layer.key[selectedToken][kvHead]}
              color="#765ba7"
            />
            <p className="gptx-footnote">
              Row sum {selectedAttention.reduce((sum, value) => sum + value, 0).toFixed(3)}.
              Attention weights are mixing coefficients, not a full explanation of the model’s reasoning.
            </p>
          </PanelSection>
        </>
      )}

      {mode === 'rope' && (
        <>
          <PanelSection label="actual rotated vectors">
            <VectorStrip
              label={`Q · pos ${selectedToken}`}
              values={layer.query[selectedToken][selectedHead]}
              color="#168aa1"
            />
            <VectorStrip
              label={`K · pos ${Math.max(0, selectedToken - 2)}`}
              values={layer.key[Math.max(0, selectedToken - 2)][kvHead]}
              color="#765ba7"
            />
          </PanelSection>
          <PanelSection label="what survives the rotation">
            <div className="gptx-delta">
              <span>
                Q magnitude
                <strong>{norm(layer.query[selectedToken][selectedHead]).toFixed(3)}</strong>
              </span>
              <span>
                K magnitude
                <strong>{norm(layer.key[Math.max(0, selectedToken - 2)][kvHead]).toFixed(3)}</strong>
              </span>
              <span>
                dimensions / head
                <strong>2</strong>
              </span>
            </div>
            <p className="gptx-footnote">
              Real Llama heads use 128 dimensions, paired into 64 clocks. This model uses one visible clock per head.
            </p>
          </PanelSection>
        </>
      )}

      {mode === 'gqa' && (
        <>
          <PanelSection label="attention variant">
            <Segmented
              label="Attention head sharing"
              value={variant}
              onChange={onVariantChange}
              options={[
                { value: 'mha', label: 'MHA · 4/4' },
                { value: 'gqa', label: 'GQA · 4/2' },
                { value: 'mqa', label: 'MQA · 4/1' },
              ]}
            />
          </PanelSection>
          <PanelSection label="selected mapping">
            <div className="gptx-mapping">
              {Array.from({ length: 4 }, (_, head) => {
                const kvCount = variant === 'mha' ? 4 : variant === 'mqa' ? 1 : 2;
                const mapped = Math.floor(head / (4 / kvCount));
                return (
                  <span key={head} className={head === selectedHead ? 'is-selected' : ''}>
                    Q{head + 1} <b>→</b> KV{mapped + 1}
                  </span>
                );
              })}
            </div>
            <p className="gptx-footnote">
              Comparison only: the trained checkpoint remains 4-query / 2-KV GQA.
            </p>
          </PanelSection>
          <PanelSection
            label={`theoretical toy BF16 cache · ${contextLength.toLocaleString()} tokens`}
          >
            <div className="gptx-cache-compare">
              <span>
                this setting
                <strong>{formatBytes(gqaBytes)}</strong>
              </span>
              <span>
                four-head MHA
                <strong>
                  {formatBytes(
                    kvCacheBytes({
                      layers: 4,
                      tokens: contextLength,
                      kvHeads: 4,
                      headDim: 2,
                    })
                  )}
                </strong>
              </span>
            </div>
            <p className="gptx-footnote">
              GQA groups heads, never tokens. Query projections remain independent.
            </p>
          </PanelSection>
        </>
      )}

      {mode === 'cache' && (
        <>
          <PanelSection label={decodeStep === 0 ? 'phase · prefill' : `phase · decode ${decodeStep}`}>
            <div className="gptx-cache-controls">
              <button type="button" onClick={onPrefill} className={decodeStep === 0 ? 'is-selected' : ''}>
                Prefill
              </button>
              <button type="button" onClick={onNextToken} disabled={decodeStep >= maxDecodeStep}>
                Next token
              </button>
              <button type="button" onClick={onPlayToggle} disabled={maxDecodeStep === 0}>
                {playing ? 'Pause' : 'Play'}
              </button>
            </div>
            <p className="gptx-footnote">
              {decodeStep === 0
                ? 'All prompt positions run in parallel under the causal mask.'
                : 'Only the newcomer runs through the stack; each layer appends its K and V.'}
            </p>
          </PanelSection>
          <PanelSection
            label={`exact cache · layer ${selectedLayer + 1} · KV head ${kvHead + 1}`}
          >
            <TensorInspector
              title="cached K"
              subtitle={`${cachedKeys.length} positions × ${NANO_CONFIG.headDim} dimensions`}
              matrix={cachedKeys}
              rowLabels={cacheTokens}
              columnLabels={['d0', 'd1']}
              color="#765ba7"
            />
            <TensorInspector
              title="cached V"
              subtitle="the values reused by the newcomer"
              matrix={cachedValues}
              rowLabels={cacheTokens}
              columnLabels={['d0', 'd1']}
              color="#25896f"
            />
          </PanelSection>
          <PanelSection label="production-scale comparison">
            <label className="gptx-range-label" htmlFor="gptx-context">
              context length
              <output>{contextLength.toLocaleString()} tokens</output>
            </label>
            <input
              id="gptx-context"
              className="gptx-range"
              type="range"
              min="1024"
              max="131072"
              step="1024"
              value={contextLength}
              onChange={(event) => onContextLengthChange(Number(event.target.value))}
            />
            <div className="gptx-cache-compare">
              <span>
                Llama 3.1 8B GQA
                <strong>{formatBytes(modernGqaBytes)}</strong>
              </span>
              <span>
                same shape with MHA
                <strong>{formatBytes(mhaBytes)}</strong>
              </span>
            </div>
            <p className="gptx-footnote">
              Batch 1, BF16 K/V only. The cache saves recomputation, but decode still scans a growing context.
            </p>
          </PanelSection>
        </>
      )}

      {mode === 'mlp' && (
        <>
          <PanelSection label={`layer ${selectedLayer + 1} · token “${model.tokens[selectedToken]}”`}>
            <VectorStrip
              label="SiLU(g)"
              values={layer.gate[selectedToken].map(
                (value) => value / (1 + Math.exp(-value))
              )}
              color="#765ba7"
            />
            <VectorStrip label="up" values={layer.up[selectedToken]} color="#168aa1" />
            <VectorStrip label="product" values={layer.swiglu[selectedToken]} color="#b9811f" />
          </PanelSection>
          <PanelSection label="residual write">
            <div className="gptx-delta">
              <span>
                input width
                <strong>8</strong>
              </span>
              <span>
                hidden width
                <strong>16</strong>
              </span>
              <span>
                update norm
                <strong>{norm(layer.mlpUpdate[selectedToken]).toFixed(3)}</strong>
              </span>
            </div>
            <p className="gptx-footnote">
              SiLU is a signed gate, not an on/off switch. Negative features can pass through.
            </p>
          </PanelSection>
        </>
      )}

      {mode === 'weights' && (
        <MatrixSelection
          matrixId={matrixId}
          onMatrixChange={onMatrixChange}
          selectedLayer={selectedLayer}
          matrixInfo={matrixInfo}
          selectedCell={selectedCell}
          onCellSelect={onCellSelect}
        />
      )}
    </aside>
  );
}

function ControlStrip({
  mode,
  model,
  selectedLayer,
  selectedHead,
  selectedToken,
  matrixId,
  onLayerSelect,
  onHeadSelect,
  onTokenSelect,
}) {
  const showsLayer = mode !== 'gqa';
  const showsHead =
    ['attention', 'rope', 'gqa', 'cache'].includes(mode) ||
    (mode === 'weights' &&
      ['actAttentionScores', 'actAttentionWeights'].includes(matrixId));
  const headCount =
    mode === 'cache' ? NANO_CONFIG.kvHeads : NANO_CONFIG.queryHeads;
  const displayedHead =
    mode === 'cache' ? queryHeadToKvHead(selectedHead) : selectedHead;
  return (
    <div className="gptx-control-strip">
      <div className="gptx-control-group gptx-control-group--tokens">
        <span className="gptx-control-label">token</span>
        <div className="gptx-token-buttons">
          {model.tokens.map((token, index) => (
            <button
              type="button"
              key={`${token}-${index}`}
              className={index === selectedToken ? 'is-selected' : ''}
              aria-pressed={index === selectedToken}
              onClick={() => onTokenSelect(index)}
            >
              <span>{index}</span>
              {token}
            </button>
          ))}
        </div>
      </div>

      {showsLayer && (
        <div className="gptx-control-group">
          <span className="gptx-control-label">layer</span>
          <div className="gptx-number-buttons">
            {Array.from({ length: NANO_CONFIG.layers }, (_, index) => (
              <button
                type="button"
                key={index}
                className={index === selectedLayer ? 'is-selected' : ''}
                aria-label={`Select layer ${index + 1}`}
                aria-pressed={index === selectedLayer}
                onClick={() => onLayerSelect(index)}
              >
                {index + 1}
              </button>
            ))}
          </div>
        </div>
      )}

      {showsHead && (
        <div className="gptx-control-group">
          <span className="gptx-control-label">
            {mode === 'cache' ? 'KV head' : 'head'}
          </span>
          <div className="gptx-head-buttons">
            {Array.from({ length: headCount }, (_, index) => (
              <button
                type="button"
                key={index}
                className={index === displayedHead ? 'is-selected' : ''}
                style={{
                  '--gptx-head-color':
                    HEAD_COLORS[
                      mode === 'cache'
                        ? index * (NANO_CONFIG.queryHeads / NANO_CONFIG.kvHeads)
                        : index
                    ],
                }}
                aria-label={`Select ${
                  mode === 'cache' ? 'KV' : 'attention'
                } head ${index + 1}`}
                aria-pressed={index === displayedHead}
                onClick={() =>
                  onHeadSelect(
                    mode === 'cache'
                      ? index * (NANO_CONFIG.queryHeads / NANO_CONFIG.kvHeads)
                      : index
                  )
                }
              >
                {index + 1}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function FallbackDiagram({ model }) {
  return (
    <div className="gptx-webgl-fallback">
      <p className="gptx-eyebrow">WEBGL IS UNAVAILABLE</p>
      <h3>The numbers still work.</h3>
      <div className="gptx-fallback-stack">
        {Array.from({ length: 4 }, (_, index) => (
          <div key={index}>
            <span>Layer {index + 1}</span>
            <small>RMS → attention → + → RMS → SwiGLU → +</small>
          </div>
        ))}
      </div>
      <p>
        The trained nano model predicts <strong>{model.predictions[0].token}</strong>.
        Use the inspector and controls beside this diagram to explore its exact tensors.
      </p>
    </div>
  );
}

export default function GPTExplorer() {
  const [mode, setMode] = useState('stack');
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [draftPrompt, setDraftPrompt] = useState(DEFAULT_PROMPT);
  const model = useMemo(() => prefillNanoGPT(prompt), [prompt]);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedToken, setSelectedToken] = useState(model.tokens.length - 1);
  const [exploded, setExploded] = useState(false);
  const [variant, setVariant] = useState('gqa');
  const [decodeStep, setDecodeStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [contextLength, setContextLength] = useState(8192);
  const [matrixId, setMatrixId] = useState('query');
  const [selectedCell, setSelectedCell] = useState(null);
  const [tourStep, setTourStep] = useState(null);
  const [resetKey, setResetKey] = useState(0);
  const [hasWebGL] = useState(() =>
    typeof window === 'undefined' ? false : supportsWebGL()
  );
  const tourTriggerRef = useRef(null);
  const tourCloseRef = useRef(null);
  const tourWasOpen = useRef(false);
  const reducedMotion = useReducedMotion();
  const activeToken = Math.min(selectedToken, model.tokens.length - 1);
  const draftTokens = rawPromptTokens(draftPrompt);
  const unknownTokens = [
    ...new Set(
      draftTokens.filter((token) => !MODEL_WEIGHTS.vocab.includes(token))
    ),
  ];
  const unsupportedOnly =
    Boolean(draftPrompt.trim()) && draftTokens.length === 0;
  const promptNotices = [
    draftTokens.length > NANO_CONFIG.maxTokens
      ? `only the first ${NANO_CONFIG.maxTokens} of ${draftTokens.length} tokens will run`
      : null,
    unknownTokens.length
      ? `${unknownTokens.join(', ')} ${unknownTokens.length === 1 ? 'uses' : 'use'} <unk>`
      : null,
    unsupportedOnly ? 'unsupported characters use <unk>' : null,
  ].filter(Boolean);

  useEffect(() => {
    setSelectedToken(model.tokens.length - 1);
    setDecodeStep(0);
    setPlaying(false);
  }, [model]);

  useEffect(() => {
    if (tourStep === null) {
      if (tourWasOpen.current) {
        tourWasOpen.current = false;
        tourTriggerRef.current?.focus();
      }
      return undefined;
    }

    let focusFrame;
    if (!tourWasOpen.current) {
      tourWasOpen.current = true;
      focusFrame = window.requestAnimationFrame(() => {
        tourCloseRef.current?.focus();
      });
    }
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setTourStep(null);
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      if (focusFrame) window.cancelAnimationFrame(focusFrame);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [tourStep]);

  const cacheTrace = useMemo(() => {
    const decodes = [];
    let current = model;
    const remaining = Math.max(0, NANO_CONFIG.maxTokens - model.tokens.length);
    for (let index = 0; index < Math.min(3, remaining); index += 1) {
      const next = current.predictions[0]?.token;
      if (!next) break;
      current = decodeNanoGPTToken(current.cache, next);
      decodes.push(current);
      if (next === '<eos>') break;
    }
    return { prefill: model, decodes };
  }, [model]);

  const maxDecodeStep = cacheTrace.decodes.length;
  const activeDecodeStep = Math.min(decodeStep, maxDecodeStep);
  const activeCacheResult =
    activeDecodeStep === 0
      ? cacheTrace.prefill
      : cacheTrace.decodes[activeDecodeStep - 1];
  const activeCache = activeCacheResult.cache;
  const cacheTokens = activeCache.tokens.map((token) =>
    token === '<eos>' ? '⟨eos⟩' : token
  );
  const cacheCount = activeCache.tokens.length;

  useEffect(() => {
    if (!playing || mode !== 'cache') return undefined;
    const timer = window.setInterval(() => {
      setDecodeStep((current) => {
        if (current >= maxDecodeStep) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, reducedMotion ? 1600 : 1050);
    return () => window.clearInterval(timer);
  }, [playing, mode, maxDecodeStep, reducedMotion]);

  useEffect(() => {
    if (tourStep === null) return;
    const step = TOUR[tourStep];
    setMode(step.mode);
    if (typeof step.exploded === 'boolean') setExploded(step.exploded);
    setResetKey((value) => value + 1);
  }, [tourStep]);

  const matrixInfo = useMemo(() => {
    const option =
      TENSOR_OPTIONS.find((candidate) => candidate.id === matrixId) ??
      TENSOR_OPTIONS[0];
    let matrix;
    let rowLabels;
    let columnLabels;
    let scope;
    const layer = model.layers[selectedLayer];

    if (matrixId === 'embedding') {
      matrix = MODEL_WEIGHTS.embedding;
      rowLabels = MODEL_WEIGHTS.vocab;
      scope = 'complete tied input / output table';
    } else if (matrixId === 'finalNorm') {
      matrix = MODEL_WEIGHTS.finalNorm;
      scope = 'shared after the final decoder block';
    } else if (matrixId === 'actEmbeddings') {
      matrix = model.embeddings;
      rowLabels = model.tokens;
      scope = 'live token embeddings';
    } else if (matrixId === 'actInput') {
      matrix = layer.input;
    } else if (matrixId === 'actNormAttention') {
      matrix = layer.normalizedAttention;
    } else if (matrixId === 'actQuery') {
      matrix = flattenHeads(layer.query);
    } else if (matrixId === 'actKey') {
      matrix = flattenHeads(layer.key);
    } else if (matrixId === 'actValue') {
      matrix = flattenHeads(layer.value);
    } else if (matrixId === 'actAttentionScores') {
      matrix = layer.attentionScores[selectedHead].map((row) =>
        row.map((value) =>
          value === MASKED_ATTENTION_SCORE
            ? Number.NEGATIVE_INFINITY
            : value
        )
      );
      rowLabels = model.tokens;
      columnLabels = model.tokens;
      scope = `live layer ${selectedLayer + 1}, query head ${selectedHead + 1}; masked cells are shown as −∞`;
    } else if (matrixId === 'actAttentionWeights') {
      matrix = layer.attentionWeights[selectedHead];
      rowLabels = model.tokens;
      columnLabels = model.tokens;
      scope = `live layer ${selectedLayer + 1}, query head ${selectedHead + 1}`;
    } else if (matrixId === 'actAttentionHeads') {
      matrix = flattenHeads(layer.attentionHeads);
    } else if (matrixId === 'actAttentionUpdate') {
      matrix = layer.attentionUpdate;
    } else if (matrixId === 'actAfterAttention') {
      matrix = layer.afterAttention;
    } else if (matrixId === 'actNormMlp') {
      matrix = layer.normalizedMlp;
    } else if (matrixId === 'actGate') {
      matrix = layer.gate;
    } else if (matrixId === 'actUp') {
      matrix = layer.up;
    } else if (matrixId === 'actSwiglu') {
      matrix = layer.swiglu;
    } else if (matrixId === 'actMlpUpdate') {
      matrix = layer.mlpUpdate;
    } else if (matrixId === 'actOutput') {
      matrix = layer.output;
    } else if (matrixId === 'actFinal') {
      matrix = model.final;
      scope = 'live state after the final RMSNorm';
    } else if (matrixId === 'actLogits') {
      matrix = [model.logits];
      columnLabels = MODEL_WEIGHTS.vocab;
      scope = `live vocabulary logits after “${model.tokens.at(-1)}”`;
    } else if (matrixId === 'actProbabilities') {
      matrix = [model.probabilities];
      columnLabels = MODEL_WEIGHTS.vocab;
      scope = `live softmax probabilities after “${model.tokens.at(-1)}”`;
    } else {
      matrix = MODEL_WEIGHTS.layers[selectedLayer][matrixId];
    }
    const rows = Array.isArray(matrix?.[0]) ? matrix.length : 1;
    const columns = Array.isArray(matrix?.[0]) ? matrix[0].length : matrix.length;

    if (!rowLabels && option.kind === 'activation' && rows === model.tokens.length) {
      rowLabels = model.tokens;
    }
    if (!scope && option.kind === 'activation') {
      scope = `live layer ${selectedLayer + 1} · all ${model.tokens.length} prompt tokens`;
    }

    return {
      ...option,
      matrix,
      rows,
      columns,
      scope,
      rowLabels:
        rowLabels ??
        (rows > 1 ? Array.from({ length: rows }, (_, index) => `r${index}`) : undefined),
      columnLabels:
        columnLabels ??
        Array.from({ length: columns }, (_, index) => `c${index}`),
    };
  }, [matrixId, model, selectedHead, selectedLayer]);

  useEffect(() => {
    setSelectedCell(null);
  }, [matrixId, selectedHead, selectedLayer]);

  const submitPrompt = (event) => {
    event.preventDefault();
    if (!draftPrompt.trim()) return;
    const nextPrompt = draftPrompt.trim();
    setSelectedToken(tokenizePrompt(nextPrompt).length - 1);
    setPrompt(nextPrompt);
    setMode('stack');
    setResetKey((value) => value + 1);
  };

  const choosePreset = (value) => {
    setDraftPrompt(value);
    setSelectedToken(tokenizePrompt(value).length - 1);
    setPrompt(value);
    setMode('stack');
    setResetKey((key) => key + 1);
  };

  const changeMode = (nextMode) => {
    setMode(nextMode);
    setTourStep(null);
    setResetKey((value) => value + 1);
  };

  const matrixForScene = matrixInfo.matrix;
  const activeCopy =
    mode === 'weights' && matrixInfo.kind === 'activation'
      ? { title: 'Every major activation tensor fits' }
      : MODE_COPY[mode];
  const currentTour = tourStep === null ? null : TOUR[tourStep];

  return (
    <div className="gptx-theme">
      <div className={`gptx-shell gptx-shell--${mode}`}>
      <header className="gptx-topbar">
        <div className="gptx-brand">
          <span className="gptx-brand-mark" aria-hidden="true">
            8
          </span>
          <span>
            <strong>Nano GPT Lab</strong>
            <small>Modern decoder microscope</small>
          </span>
        </div>

        <nav className="gptx-mode-nav-wrap" aria-label="Explorer chapters">
          <SegmentedControl.Root
            className="gptx-mode-nav"
            value={mode}
            onValueChange={(nextMode) => nextMode && changeMode(nextMode)}
            size="3"
            aria-label="Explorer chapters"
          >
            {MODES.map((item) => (
              <SegmentedControl.Item
                key={item.id}
                value={item.id}
                style={{ '--gptx-mode-color': item.color }}
              >
                {item.label}
              </SegmentedControl.Item>
            ))}
          </SegmentedControl.Root>
        </nav>

        <div className="gptx-top-actions">
          <Button
            type="button"
            size="3"
            className="gptx-tour-button"
            ref={tourTriggerRef}
            aria-expanded={tourStep !== null}
            aria-controls="gptx-guided-tour"
            onClick={() => setTourStep(0)}
          >
            Start guided tour
          </Button>
        </div>
      </header>

      <div className="gptx-prompt-row">
        <form className="gptx-prompt-form" onSubmit={submitPrompt}>
          <label htmlFor="gptx-prompt">Try a prompt</label>
          <TextField.Root
            id="gptx-prompt"
            size="3"
            value={draftPrompt}
            onChange={(event) => setDraftPrompt(event.target.value)}
            maxLength={72}
            spellCheck="false"
          />
          <Button type="submit" size="3">
            Run
          </Button>
        </form>
        <div className="gptx-preset-row" aria-label="Example prompts">
          <span>Examples</span>
          {PRESETS.map((preset) => (
            <Button
              type="button"
              size="2"
              variant={prompt === preset ? 'soft' : 'ghost'}
              key={preset}
              className={prompt === preset ? 'is-selected' : ''}
              aria-label={`Use prompt: ${preset}`}
              onClick={() => choosePreset(preset)}
            >
              {preset.split(' ')[1]}
            </Button>
          ))}
        </div>
        {promptNotices.length ? (
          <span className="gptx-prompt-warning" role="status">
            {promptNotices.join(' · ')}
          </span>
        ) : (
          <span className="gptx-trained-badge">
            <i aria-hidden="true" />
            trained checkpoint · 23 tokens
          </span>
        )}
      </div>

      <div className="gptx-main">
        <div className="gptx-stage">
          {hasWebGL ? (
            <GPTScene
              mode={mode}
              model={model}
              selectedLayer={selectedLayer}
              selectedToken={activeToken}
              selectedHead={selectedHead}
              exploded={exploded}
              variant={variant}
              cacheTokens={cacheTokens}
              prefillCount={model.tokens.length}
              cacheCount={cacheCount}
              matrix={matrixForScene}
              matrixColor={matrixInfo.color}
              selectedCell={selectedCell}
              onLayerSelect={setSelectedLayer}
              onTokenSelect={setSelectedToken}
              onHeadSelect={setSelectedHead}
              onCellSelect={setSelectedCell}
              reducedMotion={reducedMotion}
              resetKey={resetKey}
              accessibleLabel={`${activeCopy.title}. Selected layer ${
                selectedLayer + 1
              }, token ${model.tokens[activeToken]}, head ${selectedHead + 1}.`}
            />
          ) : (
            <FallbackDiagram model={model} />
          )}

          <div className="gptx-stage-hud gptx-stage-hud--top">
            <span>Live trained model</span>
            <span>
              Layer {selectedLayer + 1} ·{' '}
              {mode === 'cache'
                ? `KV head ${queryHeadToKvHead(selectedHead) + 1}`
                : `Head ${selectedHead + 1}`}{' '}
              · Token {activeToken}
            </span>
          </div>
          <div className="gptx-stage-toolbar" aria-label="3D view controls">
            {mode === 'stack' && (
              <Button
                type="button"
                size="2"
                variant="surface"
                className={exploded ? 'is-selected' : ''}
                aria-pressed={exploded}
                onClick={() => setExploded((value) => !value)}
              >
                {exploded ? 'Collapse layers' : 'Expand layers'}
              </Button>
            )}
            <Button
              type="button"
              size="2"
              variant="surface"
              onClick={() => setResetKey((value) => value + 1)}
              aria-label="Reset 3D camera"
            >
              Reset view
            </Button>
          </div>
          <div className="gptx-drag-hint">
            <span aria-hidden="true">↻</span>
            Drag to orbit · scroll to zoom · click to inspect
          </div>

          {currentTour && (
            <section
              className="gptx-tour-card"
              id="gptx-guided-tour"
              role="dialog"
              aria-modal="false"
              aria-labelledby="gptx-tour-title"
            >
              <div className="gptx-tour-meta">
                <span>
                  guided tour · {tourStep + 1}/{TOUR.length}
                </span>
                <button
                  type="button"
                  ref={tourCloseRef}
                  onClick={() => setTourStep(null)}
                  aria-label="Close guided tour"
                >
                  ×
                </button>
              </div>
              <h3 id="gptx-tour-title">{currentTour.title}</h3>
              <p>{currentTour.body}</p>
              <div className="gptx-tour-actions">
                <button
                  type="button"
                  disabled={tourStep === 0}
                  onClick={() => setTourStep((value) => Math.max(0, value - 1))}
                >
                  Back
                </button>
                <button
                  type="button"
                  className="gptx-tour-next"
                  onClick={() => {
                    if (tourStep === TOUR.length - 1) setTourStep(null);
                    else setTourStep((value) => value + 1);
                  }}
                >
                  {tourStep === TOUR.length - 1 ? 'Explore freely' : 'Next'}
                </button>
              </div>
            </section>
          )}
        </div>

        <Inspector
          mode={mode}
          model={model}
          selectedLayer={selectedLayer}
          selectedHead={selectedHead}
          selectedToken={activeToken}
          onTokenSelect={setSelectedToken}
          variant={variant}
          onVariantChange={setVariant}
          contextLength={contextLength}
          onContextLengthChange={setContextLength}
          decodeStep={activeDecodeStep}
          maxDecodeStep={maxDecodeStep}
          cacheState={activeCache}
          cacheTokens={cacheTokens}
          playing={playing}
          onPrefill={() => {
            setPlaying(false);
            setDecodeStep(0);
          }}
          onNextToken={() =>
            setDecodeStep((value) => Math.min(maxDecodeStep, value + 1))
          }
          onPlayToggle={() => {
            if (activeDecodeStep >= maxDecodeStep) setDecodeStep(0);
            setPlaying((value) => !value);
          }}
          matrixId={matrixId}
          onMatrixChange={setMatrixId}
          matrixInfo={matrixInfo}
          selectedCell={selectedCell}
          onCellSelect={setSelectedCell}
        />
      </div>

      <ControlStrip
        mode={mode}
        model={model}
        selectedLayer={selectedLayer}
        selectedHead={selectedHead}
        selectedToken={activeToken}
        matrixId={matrixId}
        onLayerSelect={setSelectedLayer}
        onHeadSelect={setSelectedHead}
        onTokenSelect={setSelectedToken}
      />

      <footer className="gptx-footer">
        <span>
          real inference · {TRAINING_INFO.steps.toLocaleString()} training steps ·{' '}
          {(TRAINING_INFO.tokenAccuracy * 100).toFixed(2)}% token accuracy
        </span>
        <span>
          modern open-decoder pattern, not the undisclosed architecture of GPT-4/5
        </span>
      </footer>

      <p className="gptx-sr-status" aria-live="polite">
        {activeCopy.title}. Layer {selectedLayer + 1}, head {selectedHead + 1},
        token {model.tokens[activeToken]}. Top prediction{' '}
        {model.predictions[0].token}.
      </p>
      </div>
    </div>
  );
}
