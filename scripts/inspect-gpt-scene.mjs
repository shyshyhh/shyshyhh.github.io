#!/usr/bin/env node

import {
  buildStackSceneBlueprint,
  inspectStackSceneBlueprint,
  projectStackSceneTo2D,
  validateStackSceneBlueprint,
} from '../src/components/gpt-architecture/stack-scene-blueprint.mjs';

const argumentsSet = new Set(process.argv.slice(2));
const readInteger = (prefix, fallback) => {
  const argument = [...argumentsSet].find((value) =>
    value.startsWith(`${prefix}=`)
  );
  return argument ? Number.parseInt(argument.slice(prefix.length + 1), 10) : fallback;
};

const tokenCount = readInteger('--tokens', 6);
const selectedLayer = readInteger('--layer', 3) - 1;
const selectedToken = readInteger('--token', tokenCount) - 1;
const expanded = !argumentsSet.has('--collapsed');
const vocabulary = [
  'the',
  'robot',
  'worked',
  'with',
  'code',
  'today',
  'and',
  'learned',
];
const tokens = Array.from(
  { length: tokenCount },
  (_, index) => vocabulary[index] ?? `t${index + 1}`
);

const scene = buildStackSceneBlueprint({
  tokens,
  layerCount: 4,
  selectedLayer,
  selectedToken,
  expanded,
});
const errors = validateStackSceneBlueprint(scene);

if (errors.length) {
  throw new Error(errors.join('\n'));
}

if (argumentsSet.has('--json')) {
  console.log(JSON.stringify(projectStackSceneTo2D(scene), null, 2));
} else {
  console.log(inspectStackSceneBlueprint(scene));
}
