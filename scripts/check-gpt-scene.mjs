#!/usr/bin/env node

import {
  buildStackSceneBlueprint,
  projectStackSceneTo2D,
  validateStackSceneBlueprint,
} from '../src/components/gpt-architecture/stack-scene-blueprint.mjs';

const tokenSets = [
  ['the'],
  ['the', 'robot', 'worked', 'with'],
  ['the', 'robot', 'worked', 'with', 'code', 'and', 'learned', 'fast'],
];

let checked = 0;

for (const tokens of tokenSets) {
  for (let selectedLayer = 0; selectedLayer < 4; selectedLayer += 1) {
    for (const expanded of [false, true]) {
      for (const selectedToken of [0, tokens.length - 1]) {
        const scene = buildStackSceneBlueprint({
          tokens,
          layerCount: 4,
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
          projection.verticalAxis.meaning !== 'model compute depth'
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
  `PASS ${checked} stack layouts preserve x=sequence, y=compute, z=branches`
);
