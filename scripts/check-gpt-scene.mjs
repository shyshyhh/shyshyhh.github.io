#!/usr/bin/env node

import {
  buildStackSceneBlueprint,
  projectStackSceneTo2D,
  validateStackSceneBlueprint,
} from '../src/components/gpt-architecture/stack-scene-blueprint.mjs';

const vocabulary = [
  'the',
  'robot',
  'worked',
  'with',
  'code',
  'and',
  'learned',
  'fast',
];
const tokenSets = Array.from({ length: 8 }, (_, index) =>
  vocabulary.slice(0, index + 1)
);

function representativeNorms(layerCount, tokenCount) {
  const valueCount = layerCount * tokenCount;
  return Array.from({ length: layerCount }, (_, layerIndex) =>
    Array.from({ length: tokenCount }, (_, tokenIndex) => {
      const flatIndex = layerIndex * tokenCount + tokenIndex;
      return flatIndex / Math.max(valueCount - 1, 1);
    })
  );
}

let checked = 0;

for (const tokens of tokenSets) {
  const layerTokenNorms = representativeNorms(4, tokens.length);
  const normValues = layerTokenNorms.flat();
  if (
    !normValues.includes(0) ||
    !normValues.includes(1) ||
    !normValues.some((value) => value > 0 && value < 1)
  ) {
    throw new Error('representative norm matrix must cover 0, intermediate, and 1');
  }

  for (let selectedLayer = 0; selectedLayer < 4; selectedLayer += 1) {
    for (const expanded of [false, true]) {
      for (const selectedToken of [...new Set([0, tokens.length - 1])]) {
        const scene = buildStackSceneBlueprint({
          tokens,
          layerCount: 4,
          layerTokenNorms,
          selectedLayer,
          selectedToken,
          expanded,
        });
        const errors = validateStackSceneBlueprint(scene);
        if (errors.length) {
          throw new Error(
            [
              `Invalid stack scene for ${tokens.length} tokens, layer ${
                selectedLayer + 1
              }, ${expanded ? 'open' : 'collapsed'}:`,
              ...errors.map((error) => `- ${error}`),
            ].join('\n')
          );
        }

        const projection = projectStackSceneTo2D(scene);
        if (
          projection.horizontalAxis.meaning !== 'token sequence position' ||
          projection.horizontalAxis.direction !== 'left-to-right' ||
          projection.verticalAxis.meaning !== 'model compute depth' ||
          projection.verticalAxis.direction !== 'top-to-bottom'
        ) {
          throw new Error('2D projection lost the stack axis contract');
        }
        const projectedBranches = projection.paths.filter(
          (path) => path.kind === 'parallel-branch'
        );
        if (
          expanded &&
          (projectedBranches.length !== 2 ||
            projectedBranches.some(
              (branch) =>
                branch.depthOffsets[0] !== 0 ||
                branch.depthOffsets.at(-1) !== 0 ||
                !branch.depthOffsets.some((depth) => depth !== 0)
            ))
        ) {
          throw new Error('2D projection lost residual branch depth');
        }
        if (!expanded && projectedBranches.length !== 0) {
          throw new Error('collapsed 2D projection exposed residual branches');
        }
        checked += 1;
      }
    }
  }
}

console.log(
  `PASS ${checked} stack layouts preserve x=sequence →, y=compute ↓, z=branches`
);
