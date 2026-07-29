import React, { useMemo, useState } from 'react';
import './gpt-primer.css';

const STEP_COUNT = 6;

const NEXT_LABELS = [
  'Split it into tokens',
  'Give them numbers',
  'Run one block',
  'Repeat the block',
  'Make the prediction',
  'Open the advanced lab',
];

function probability(value) {
  if (value >= 0.9995) return '>99.9%';
  if (value < 0.001) return '<0.1%';
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

function Sentence({ tokens, cursor = false }) {
  return (
    <div className="gptx-primer-sentence" aria-label={tokens.join(' ')}>
      {tokens.map((token, index) => (
        <span key={`${token}-${index}`}>{token}</span>
      ))}
      {cursor && <i aria-hidden="true" />}
    </div>
  );
}

function TokenSlots({ tokens }) {
  return (
    <div
      className="gptx-primer-tokens"
      aria-label={`${tokens.length} tokens`}
      data-axis-contract="sequence=left-to-right"
    >
      {tokens.map((token, index) => (
        <div key={`${token}-${index}`}>
          <span>{index}</span>
          <strong>{token}</strong>
        </div>
      ))}
    </div>
  );
}

function Scratchpads({ tokens, embeddings }) {
  const maximum = Math.max(
    ...embeddings.flatMap((vector) => vector.map((value) => Math.abs(value))),
    0.001
  );

  return (
    <div
      className="gptx-primer-scratchpads"
      aria-label="Each token paired with an eight-number scratchpad"
      data-axis-contract="sequence=left-to-right"
      data-vector-size="8"
    >
      {tokens.map((token, tokenIndex) => (
        <div
          className={tokenIndex === tokens.length - 1 ? 'is-selected' : ''}
          key={`${token}-${tokenIndex}`}
        >
          <strong>{token}</strong>
          <span className="gptx-primer-vector" aria-hidden="true">
            {embeddings[tokenIndex].map((value, valueIndex) => (
              <i
                className={value < 0 ? 'is-negative' : ''}
                key={valueIndex}
                style={{
                  '--gptx-primer-value': `${Math.max(
                    12,
                    (Math.abs(value) / maximum) * 100
                  )}%`,
                }}
              />
            ))}
          </span>
          <small>8 numbers</small>
        </div>
      ))}
    </div>
  );
}

function OneBlock() {
  const operations = [
    ['Look around', 'attention'],
    ['Keep + add', 'residual'],
    ['Think alone', 'MLP'],
    ['Keep + add', 'residual'],
  ];

  return (
    <ol
      className="gptx-primer-one-block"
      aria-label="One transformer block"
      data-axis-contract="compute=top-to-bottom"
    >
      {operations.map(([plain, technical], index) => (
        <li key={`${plain}-${index}`}>
          <span>{index + 1}</span>
          <strong>{plain}</strong>
          <small>{technical}</small>
        </li>
      ))}
    </ol>
  );
}

function RepeatedBlocks() {
  return (
    <div
      className="gptx-primer-block-chain"
      aria-label="The same transformer block repeated four times"
      data-axis-contract="compute=top-to-bottom"
    >
      {Array.from({ length: 4 }, (_, index) => (
        <React.Fragment key={index}>
          <div>
            <strong>Block {index + 1}</strong>
            <span>share context</span>
            <span>edit each token</span>
          </div>
          {index < 3 && <i aria-hidden="true">→</i>}
        </React.Fragment>
      ))}
    </div>
  );
}

function Prediction({ token, predictions }) {
  return (
    <div className="gptx-primer-prediction">
      <div className="gptx-primer-final-state">
        <strong>{token}</strong>
        <span aria-hidden="true">
          {Array.from({ length: 8 }, (_, index) => (
            <i key={index} />
          ))}
        </span>
        <small>final 8 numbers</small>
      </div>
      <span className="gptx-primer-prediction-arrow" aria-hidden="true">
        →
      </span>
      <ol>
        {predictions.slice(0, 3).map((prediction, index) => (
          <li key={prediction.token}>
            <span>{String(index + 1).padStart(2, '0')}</span>
            <strong>{prediction.token}</strong>
            <i aria-hidden="true">
              <b style={{ width: `${Math.max(2, prediction.probability * 100)}%` }} />
            </i>
            <small>{probability(prediction.probability)}</small>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function GPTPrimer({ model, onOpenLab }) {
  const [step, setStep] = useState(0);
  const tokens = model.tokens;
  const content = useMemo(
    () => [
      {
        title: null,
        body: 'What comes next?',
        visual: <Sentence tokens={tokens} cursor />,
      },
      {
        title: 'First: five token slots.',
        body: 'GPT reads positions, not a sentence.',
        visual: <TokenSlots tokens={tokens} />,
      },
      {
        title: 'Each token gets 8 numbers.',
        body: 'The little cube was only this scratchpad.',
        visual: (
          <Scratchpads tokens={tokens} embeddings={model.embeddings} />
        ),
      },
      {
        title: 'One block edits those numbers.',
        body: 'It shares context, edits each token, and keeps the old state.',
        visual: <OneBlock />,
      },
      {
        title: 'The same block runs four times.',
        body: '“L1–L4” was only shorthand for these four repeats.',
        visual: <RepeatedBlocks />,
      },
      {
        title: 'The last token picks the next one.',
        body: 'Its final 8 numbers become one score per possible word.',
        visual: (
          <Prediction
            token={tokens.at(-1)}
            predictions={model.predictions}
          />
        ),
      },
    ],
    [model.embeddings, model.predictions, tokens]
  );
  const current = content[step];

  const advance = () => {
    if (step === STEP_COUNT - 1) onOpenLab();
    else setStep((value) => value + 1);
  };

  return (
    <div
      className="gptx-primer-shell"
      data-primer-step={step + 1}
      data-flow-contract="sequence-right;compute-down"
    >
      <header className="gptx-primer-header">
        <div className="gptx-brand">
          <span className="gptx-brand-mark" aria-hidden="true">
            8
          </span>
          <span>
            <strong>Nano GPT</strong>
            <small>one idea at a time</small>
          </span>
        </div>
        <span className="gptx-primer-count">
          {step + 1} / {STEP_COUNT}
        </span>
        <button type="button" onClick={onOpenLab}>
          Advanced lab
        </button>
      </header>

      <main className="gptx-primer-main" aria-live="polite">
        <div className={`gptx-primer-step gptx-primer-step--${step}`} key={step}>
          <div className="gptx-primer-copy">
            {current.title && <h2>{current.title}</h2>}
            <p>{current.body}</p>
          </div>
          <div className="gptx-primer-visual">{current.visual}</div>
        </div>
      </main>

      <footer className="gptx-primer-footer">
        <button
          type="button"
          className="gptx-primer-back"
          onClick={() => setStep((value) => Math.max(0, value - 1))}
          disabled={step === 0}
        >
          Back
        </button>
        <div className="gptx-primer-progress" aria-label={`Step ${step + 1} of ${STEP_COUNT}`}>
          {Array.from({ length: STEP_COUNT }, (_, index) => (
            <i className={index <= step ? 'is-complete' : ''} key={index} />
          ))}
        </div>
        <button
          type="button"
          className="gptx-primer-next"
          onClick={advance}
        >
          {NEXT_LABELS[step]} <span aria-hidden="true">→</span>
        </button>
      </footer>
    </div>
  );
}
