export const STACK_SCENE_SCHEMA = 'gpt-stack-scene/v1';

export const STACK_AXES = Object.freeze({
  x: Object.freeze({
    meaning: 'token sequence position',
    direction: 'left-to-right',
  }),
  y: Object.freeze({
    meaning: 'model compute depth',
    direction: 'bottom-to-top',
  }),
  z: Object.freeze({
    meaning: 'parallel branch depth',
    direction: 'front-to-back',
  }),
});

export const STACK_OPERATION_STEPS = Object.freeze([
  Object.freeze({
    id: 'attention-norm',
    label: 'RMSNorm',
    shortLabel: 'RMS',
    colorKey: 'blue',
  }),
  Object.freeze({
    id: 'attention',
    label: 'Attention',
    shortLabel: 'ATTN',
    colorKey: 'violet',
  }),
  Object.freeze({
    id: 'attention-residual',
    label: 'Residual add',
    shortLabel: '+',
    colorKey: 'cyan',
  }),
  Object.freeze({
    id: 'mlp-norm',
    label: 'RMSNorm',
    shortLabel: 'RMS',
    colorKey: 'blue',
  }),
  Object.freeze({
    id: 'swiglu',
    label: 'SwiGLU',
    shortLabel: 'SWIGLU',
    colorKey: 'coral',
  }),
  Object.freeze({
    id: 'mlp-residual',
    label: 'Residual add',
    shortLabel: '+',
    colorKey: 'mint',
  }),
]);

const BASE_LAYER_GAP = 1.55;
const EXPANSION_PUSH = 1.78;
const OPERATION_GAP = 0.58;
const OPERATION_HALF_SPAN =
  ((STACK_OPERATION_STEPS.length - 1) * OPERATION_GAP) / 2;

function assertIntegerInRange(value, minimum, maximum, label) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(
      `${label} must be an integer from ${minimum} to ${maximum}; received ${value}`
    );
  }
}

function point(x, y, z = 0) {
  return Object.freeze([x, y, z]);
}

function size(x, y, z) {
  return Object.freeze([x, y, z]);
}

function extentCenter(minimum, maximum) {
  return (minimum + maximum) / 2;
}

/**
 * Pure geometry for the stack view. Rendering code is intentionally absent:
 * this object is the inspectable source of truth for both 3D and 2D views.
 */
export function buildStackSceneBlueprint({
  tokens,
  layerCount,
  layerTokenNorms,
  selectedLayer = 0,
  selectedToken = Math.max(tokens.length - 1, 0),
  expanded = false,
}) {
  if (!Array.isArray(tokens) || tokens.length === 0) {
    throw new TypeError('tokens must contain at least one token');
  }
  assertIntegerInRange(layerCount, 1, 64, 'layerCount');
  assertIntegerInRange(selectedLayer, 0, layerCount - 1, 'selectedLayer');
  assertIntegerInRange(selectedToken, 0, tokens.length - 1, 'selectedToken');
  if (
    layerTokenNorms !== undefined &&
    (!Array.isArray(layerTokenNorms) ||
      layerTokenNorms.length !== layerCount ||
      layerTokenNorms.some(
        (layer) =>
          !Array.isArray(layer) ||
          layer.length !== tokens.length ||
          layer.some(
            (value) =>
              typeof value !== 'number' ||
              !Number.isFinite(value) ||
              value < 0 ||
              value > 1
          )
      ))
  ) {
    throw new TypeError(
      'layerTokenNorms must be a layerCount × tokenCount matrix in [0, 1]'
    );
  }

  const tokenSpacing = Math.min(
    1.08,
    6.2 / Math.max(tokens.length - 1, 1)
  );
  const tokenX = (index) =>
    (index - (tokens.length - 1) / 2) * tokenSpacing;
  const plateWidth = Math.max(6.5, tokens.length * tokenSpacing + 1.5);

  const rawLayerY = Array.from({ length: layerCount }, (_, index) => {
    const centered = (index - (layerCount - 1) / 2) * BASE_LAYER_GAP;
    if (!expanded || index === selectedLayer) return centered;
    return centered + (index < selectedLayer ? -EXPANSION_PUSH : EXPANSION_PUSH);
  });

  const rawOperationY = expanded
    ? STACK_OPERATION_STEPS.map(
        (_, index) =>
          rawLayerY[selectedLayer] -
          OPERATION_HALF_SPAN +
          index * OPERATION_GAP
      )
    : [];

  const structuralMinimum = Math.min(...rawLayerY, ...rawOperationY);
  const structuralMaximum = Math.max(...rawLayerY, ...rawOperationY);
  const centerY = extentCenter(structuralMinimum, structuralMaximum);
  const layerY = rawLayerY.map((value) => value - centerY);
  const operationY = rawOperationY.map((value) => value - centerY);
  const minimumY = structuralMinimum - centerY;
  const maximumY = structuralMaximum - centerY;
  const inputY = minimumY - 1.18;
  const outputY = maximumY + 1.42;
  const railMinimumY = inputY + 0.16;
  const railMaximumY = outputY - 0.18;

  const tokenNodes = tokens.map((token, index) => ({
    id: `token-${index}`,
    index,
    label: token,
    position: point(tokenX(index), inputY, 0),
    labelPosition: point(tokenX(index), inputY - 0.4, 0),
    radius: index === selectedToken ? 0.16 : 0.11,
    selected: index === selectedToken,
  }));

  const layerNodes = layerY.map((y, index) => ({
    id: `layer-${index}`,
    index,
    label: `L${index + 1}`,
    expanded: expanded && index === selectedLayer,
    selected: index === selectedLayer,
    position: point(0, y, 0),
    labelPosition: point(-plateWidth / 2 - 0.7, y, 0),
    openLabelPosition: point(plateWidth / 2 + 0.78, y, 0),
    size: size(plateWidth, index === selectedLayer ? 0.25 : 0.16, 2.65),
    tokenCells: tokens.map((token, tokenIndex) => {
      const strength = layerTokenNorms?.[index]?.[tokenIndex] ?? 0.5;
      return {
        id: `layer-${index}-token-${tokenIndex}`,
        token,
        tokenIndex,
        strength,
        position: point(tokenX(tokenIndex), y + 0.18, 0),
        size: size(0.52, 0.28 + strength * 0.18, 0.52),
      };
    }),
  }));

  const operationNodes = expanded
    ? STACK_OPERATION_STEPS.map((step, index) => ({
        ...step,
        id: `layer-${selectedLayer}-${step.id}`,
        layerIndex: selectedLayer,
        sequenceIndex: index,
        position: point(0, operationY[index], 0),
        labelPosition: point(-plateWidth / 2 - 0.82, operationY[index], 0),
        size: size(plateWidth - 0.45, 0.11, 1.62),
        tokenCells: tokens.map((token, tokenIndex) => ({
          id: `layer-${selectedLayer}-${step.id}-token-${tokenIndex}`,
          token,
          tokenIndex,
          position: point(tokenX(tokenIndex), operationY[index] + 0.1, 0),
          size: size(0.42, 0.16, 0.42),
        })),
      }))
    : [];

  const expandedBracket = expanded
    ? (() => {
        const bracketX = plateWidth / 2 + 0.42;
        const bracketZ = 0.92;
        const minimumBracketY = operationNodes[0].position[1] - 0.27;
        const maximumBracketY =
          operationNodes[operationNodes.length - 1].position[1] + 0.27;
        return {
          id: `layer-${selectedLayer}-open-bracket`,
          colorKey: 'cyan',
          segments: [
            {
              id: 'spine',
              points: [
                point(bracketX, minimumBracketY, bracketZ),
                point(bracketX, maximumBracketY, bracketZ),
              ],
            },
            {
              id: 'entry-tick',
              points: [
                point(bracketX - 0.22, minimumBracketY, bracketZ),
                point(bracketX, minimumBracketY, bracketZ),
              ],
            },
            {
              id: 'exit-tick',
              points: [
                point(bracketX - 0.22, maximumBracketY, bracketZ),
                point(bracketX, maximumBracketY, bracketZ),
              ],
            },
          ],
        };
      })()
    : null;

  const selectedX = tokenX(selectedToken);
  const residualBranches =
    expanded && operationNodes.length === STACK_OPERATION_STEPS.length
      ? [
          {
            id: 'attention-residual-bypass',
            label: 'attention residual bypass',
            colorKey: 'cyan',
            points: [
              point(selectedX, operationNodes[0].position[1] - 0.25, 0),
              point(selectedX, operationNodes[0].position[1] - 0.25, 1.08),
              point(selectedX, operationNodes[2].position[1], 1.08),
              point(selectedX, operationNodes[2].position[1], 0),
            ],
          },
          {
            id: 'mlp-residual-bypass',
            label: 'MLP residual bypass',
            colorKey: 'mint',
            points: [
              point(selectedX, operationNodes[2].position[1] + 0.08, 0),
              point(selectedX, operationNodes[2].position[1] + 0.08, 1.08),
              point(selectedX, operationNodes[5].position[1], 1.08),
              point(selectedX, operationNodes[5].position[1], 0),
            ],
          },
        ]
      : [];

  const rails = tokens.map((token, index) => ({
    id: `token-${index}-compute-rail`,
    token,
    tokenIndex: index,
    axis: 'y',
    selected: index === selectedToken,
    from: point(tokenX(index), railMinimumY, 0),
    to: point(tokenX(index), railMaximumY, 0),
  }));

  const selectedFlowOrder = [
    tokenNodes[selectedToken].id,
    ...layerNodes.flatMap((layer) =>
      layer.expanded
        ? operationNodes.map((operation) => operation.id)
        : [layer.id]
    ),
    'model-output',
  ];

  return {
    schema: STACK_SCENE_SCHEMA,
    axes: STACK_AXES,
    expanded,
    selection: {
      layerIndex: selectedLayer,
      tokenIndex: selectedToken,
    },
    dimensions: {
      tokenSpacing,
      plateWidth,
      baseLayerGap: BASE_LAYER_GAP,
      expansionPush: expanded ? EXPANSION_PUSH : 0,
      operationGap: OPERATION_GAP,
    },
    bounds: {
      x: Object.freeze([-plateWidth / 2, plateWidth / 2]),
      y: Object.freeze([inputY - 0.45, outputY + 0.48]),
      z: Object.freeze([-1.35, expanded ? 1.2 : 1.35]),
      structuralY: Object.freeze([minimumY, maximumY]),
    },
    tokens: tokenNodes,
    layers: layerNodes,
    operations: operationNodes,
    expandedBracket,
    residualBranches,
    rails,
    flow: {
      axis: 'y',
      direction: 'increasing',
      selectedTokenX: selectedX,
      minimumY: railMinimumY,
      maximumY: railMaximumY,
      pulseRadius: 0.16,
      pulseDepth: 0.3,
      orderedNodeIds: selectedFlowOrder,
    },
    output: {
      id: 'model-output',
      position: point(0, outputY, 0),
      labelPosition: point(0, outputY + 0.4, 0),
      size: size(2.8, 0.42, 1.35),
    },
    labels: {
      sequenceAxis: {
        text: 'tokens →',
        position: point(0, inputY - 0.86, 0),
      },
      computeAxis: {
        text: 'model depth ↑',
        position: point(plateWidth / 2 - 0.62, maximumY + 0.54, 0),
      },
    },
    selectedNormPosition: point(
      selectedX,
      expanded
        ? operationNodes[operationNodes.length - 1].position[1] + 0.48
        : layerNodes[selectedLayer].position[1] + 0.72,
      expanded ? 0.86 : 0
    ),
  };
}

/**
 * Orthographic front elevation: X and Y remain visible; Z is flattened.
 * This is deliberately plain data so CI and future agents can inspect it.
 */
export function projectStackSceneTo2D(scene) {
  return {
    schema: `${scene.schema}/front-elevation`,
    plane: 'xy',
    horizontalAxis: scene.axes.x,
    verticalAxis: scene.axes.y,
    flattenedAxis: scene.axes.z,
    expanded: scene.expanded,
    selection: { ...scene.selection },
    bounds: {
      x: [...scene.bounds.x],
      y: [...scene.bounds.y],
    },
    selectedFlowOrder: [...scene.flow.orderedNodeIds],
    inputs: scene.tokens.map((token) => ({
      id: token.id,
      kind: 'token',
      label: token.label,
      tokenIndex: token.index,
      selected: token.selected,
      x: token.position[0],
      y: token.position[1],
    })),
    marks: [
      ...scene.layers
        .filter((layer) => !layer.expanded)
        .map((layer) => ({
          id: layer.id,
          kind: 'layer',
          label: layer.label,
          x: layer.position[0],
          y: layer.position[1],
          width: layer.size[0],
          height: layer.size[1],
        })),
      ...scene.layers
        .filter((layer) => !layer.expanded)
        .flatMap((layer) =>
          layer.tokenCells.map((cell) => ({
            id: cell.id,
            kind: 'layer-token-cell',
            layerIndex: layer.index,
            tokenIndex: cell.tokenIndex,
            strength: cell.strength,
            x: cell.position[0],
            y: cell.position[1],
            width: cell.size[0],
            height: cell.size[1],
          }))
        ),
      ...scene.operations.map((operation) => ({
        id: operation.id,
        kind: 'operation',
        label: operation.label,
        sequenceIndex: operation.sequenceIndex,
        x: operation.position[0],
        y: operation.position[1],
        width: operation.size[0],
        height: operation.size[1],
      })),
      ...scene.operations.flatMap((operation) =>
        operation.tokenCells.map((cell) => ({
          id: cell.id,
          kind: 'operation-token-cell',
          operationId: operation.id,
          sequenceIndex: operation.sequenceIndex,
          tokenIndex: cell.tokenIndex,
          x: cell.position[0],
          y: cell.position[1],
          width: cell.size[0],
          height: cell.size[1],
        }))
      ),
      {
        id: scene.output.id,
        kind: 'output',
        label: 'logits',
        x: scene.output.position[0],
        y: scene.output.position[1],
        width: scene.output.size[0],
        height: scene.output.size[1],
      },
    ],
    labels: [
      ...scene.tokens.map((token) => ({
        id: `${token.id}-label`,
        text: token.label,
        x: token.labelPosition[0],
        y: token.labelPosition[1],
      })),
      ...scene.layers
        .filter((layer) => !layer.expanded)
        .map((layer) => ({
          id: `${layer.id}-label`,
          text: layer.label,
          x: layer.labelPosition[0],
          y: layer.labelPosition[1],
        })),
      ...scene.operations.map((operation) => ({
        id: `${operation.id}-label`,
        text: `${operation.sequenceIndex + 1} · ${operation.shortLabel}`,
        x: operation.labelPosition[0],
        y: operation.labelPosition[1],
      })),
      ...(scene.expanded
        ? [
            {
              id: `layer-${scene.selection.layerIndex}-open-label`,
              text: `L${scene.selection.layerIndex + 1} · OPEN`,
              x: scene.layers[scene.selection.layerIndex].openLabelPosition[0],
              y: scene.layers[scene.selection.layerIndex].openLabelPosition[1],
            },
          ]
        : []),
      {
        id: 'output-label',
        text: 'logits',
        x: scene.output.labelPosition[0],
        y: scene.output.labelPosition[1],
      },
      {
        id: 'selected-norm-label',
        text: 'selected layer output norm',
        x: scene.selectedNormPosition[0],
        y: scene.selectedNormPosition[1],
        depthOffset: scene.selectedNormPosition[2],
      },
      ...Object.entries(scene.labels).map(([id, label]) => ({
        id: `${id}-label`,
        text: label.text,
        x: label.position[0],
        y: label.position[1],
      })),
    ],
    paths: [
      ...scene.rails.map((rail) => ({
        id: rail.id,
        kind: 'compute-flow',
        tokenIndex: rail.tokenIndex,
        selected: rail.selected,
        from: [rail.from[0], rail.from[1]],
        to: [rail.to[0], rail.to[1]],
      })),
      ...scene.residualBranches.map((branch) => ({
        id: branch.id,
        kind: 'parallel-branch',
        label: branch.label,
        points: branch.points.map((branchPoint) => [
          branchPoint[0],
          branchPoint[1],
        ]),
        depthOffsets: branch.points.map((branchPoint) => branchPoint[2]),
      })),
      ...(scene.expandedBracket
        ? scene.expandedBracket.segments.map((segment) => ({
            id: `${scene.expandedBracket.id}-${segment.id}`,
            kind: 'open-block-bracket',
            points: segment.points.map((segmentPoint) => [
              segmentPoint[0],
              segmentPoint[1],
            ]),
            depthOffsets: segment.points.map(
              (segmentPoint) => segmentPoint[2]
            ),
          }))
        : []),
    ],
  };
}

export function inspectStackSceneBlueprint(scene) {
  const projection = projectStackSceneTo2D(scene);
  const rows = projection.marks
    .filter(
      (mark) =>
        mark.kind === 'layer' ||
        mark.kind === 'operation' ||
        mark.kind === 'output'
    )
    .map((mark) => ({
      y: mark.y,
      label:
        mark.kind === 'operation'
          ? `${mark.sequenceIndex + 1}. ${mark.label}`
          : mark.label,
      kind: mark.kind,
    }))
    .sort((left, right) => right.y - left.y);

  return [
    `schema: ${scene.schema}`,
    `axes: x=${scene.axes.x.meaning} (${scene.axes.x.direction}); y=${scene.axes.y.meaning} (${scene.axes.y.direction}); z=${scene.axes.z.meaning} (${scene.axes.z.direction})`,
    `selection: layer ${scene.selection.layerIndex + 1}, token ${scene.selection.tokenIndex}; ${scene.expanded ? 'open' : 'collapsed'}`,
    'front elevation (x/y; z branches flattened):',
    ...rows.map(
      (row) =>
        `  y=${row.y.toFixed(2).padStart(6)}  ${row.kind.padEnd(9)} ${row.label}`
    ),
    `selected flow: ${scene.flow.orderedNodeIds.join(' -> ')}`,
  ].join('\n');
}

export function validateStackSceneBlueprint(scene) {
  const errors = [];
  const add = (condition, message) => {
    if (!condition) errors.push(message);
  };
  const almostEqual = (left, right) => Math.abs(left - right) < 1e-9;
  const strictlyIncreasingX = (nodes) =>
    nodes.every(
      (node, index) =>
        index === 0 || node.position[0] > nodes[index - 1].position[0]
    );
  const isHorizontalPlate = (node) =>
    node.size[0] > node.size[1] &&
    node.size[2] > node.size[1];
  const serialPlates = scene.layers.flatMap((layer) =>
    layer.expanded ? scene.operations : [layer]
  );
  const inspectableNodeIds = new Set([
    ...scene.tokens.map((token) => token.id),
    ...scene.layers.filter((layer) => !layer.expanded).map((layer) => layer.id),
    ...scene.operations.map((operation) => operation.id),
    scene.output.id,
  ]);

  add(scene.schema === STACK_SCENE_SCHEMA, 'unexpected scene schema');
  add(scene.flow.axis === 'y', 'compute flow must use the Y axis');
  add(
    scene.flow.direction === 'increasing',
    'compute flow must move from lower to higher Y'
  );
  add(
    scene.layers.every(
      (layer, index, layers) =>
        index === 0 || layer.position[1] > layers[index - 1].position[1]
    ),
    'layer centers must increase monotonically on Y'
  );
  add(
    strictlyIncreasingX(scene.tokens),
    'input tokens must increase left-to-right on X'
  );
  add(
    scene.layers.every(
      (layer) =>
        isHorizontalPlate(layer) &&
        strictlyIncreasingX(layer.tokenCells) &&
        layer.tokenCells.every(
          (cell) =>
            cell.strength >= 0 &&
            cell.strength <= 1 &&
            almostEqual(cell.size[1], 0.28 + cell.strength * 0.18) &&
            almostEqual(cell.position[1], layer.tokenCells[0].position[1]) &&
            almostEqual(cell.position[2], 0)
        )
    ),
    'every layer must be a thin horizontal plate with norm-scaled, left-to-right token cells'
  );
  add(
    scene.rails.every(
      (rail) =>
        rail.axis === 'y' &&
        rail.from[0] === rail.to[0] &&
        rail.from[2] === 0 &&
        rail.to[2] === 0 &&
        rail.to[1] > rail.from[1]
    ),
    'every token rail must be a straight, increasing-Y path at Z=0'
  );
  add(
    serialPlates.every(
      (plate) =>
        isHorizontalPlate(plate) &&
        scene.rails.every(
          (rail) =>
            rail.from[0] >= plate.position[0] - plate.size[0] / 2 &&
            rail.from[0] <= plate.position[0] + plate.size[0] / 2 &&
            0 >= plate.position[2] - plate.size[2] / 2 &&
            0 <= plate.position[2] + plate.size[2] / 2 &&
            rail.from[1] < plate.position[1] &&
            rail.to[1] > plate.position[1]
        )
    ),
    'every token rail must intersect every thin serial plate'
  );
  add(
    scene.flow.orderedNodeIds.every((id) => inspectableNodeIds.has(id)),
    'selected flow order must resolve to inspectable scene nodes'
  );

  if (scene.expanded) {
    add(
      scene.operations.length === STACK_OPERATION_STEPS.length,
      'the open block must contain every serial operation'
    );
    add(
      scene.operations.every(
        (operation, index, operations) =>
          operation.position[2] === 0 &&
          isHorizontalPlate(operation) &&
          strictlyIncreasingX(operation.tokenCells) &&
          operation.tokenCells.every(
            (cell) =>
              almostEqual(
                cell.position[1],
                operation.tokenCells[0].position[1]
              ) && almostEqual(cell.position[2], 0)
          ) &&
          (index === 0 ||
            operation.position[1] > operations[index - 1].position[1])
      ),
      'serial operations must be horizontal token rows increasing on Y at Z=0'
    );
    add(
      scene.operations.every(
        (operation) =>
          operation.position[0] - operation.size[0] / 2 <=
            scene.flow.selectedTokenX &&
          operation.position[0] + operation.size[0] / 2 >=
            scene.flow.selectedTokenX
      ),
      'the selected token rail must intersect every serial operation'
    );
    add(
      scene.residualBranches.length === 2 &&
        scene.residualBranches.every((branch, index) => {
          const expectedEnd = scene.operations[index === 0 ? 2 : 5];
          return (
            branch.points[0][2] === 0 &&
            branch.points.at(-1)[2] === 0 &&
            branch.points.every((branchPoint) =>
              almostEqual(branchPoint[0], scene.flow.selectedTokenX)
            ) &&
            branch.points.some((branchPoint) => branchPoint[2] !== 0) &&
            branch.points[0][1] < branch.points.at(-1)[1] &&
            almostEqual(
              branch.points.at(-1)[1],
              expectedEnd.position[1]
            )
          );
        }),
      'the open block must contain two correctly anchored Z-depth residual bypasses'
    );
    add(
      scene.expandedBracket?.segments.length === 3 &&
        scene.expandedBracket.segments.every((segment) =>
          segment.points.every((segmentPoint) => segmentPoint[2] !== 0)
        ),
      'the open-block bracket must be fully represented in blueprint geometry'
    );
  } else {
    add(
      scene.operations.length === 0,
      'a collapsed stack must not expose operation nodes'
    );
    add(
      scene.residualBranches.length === 0 && scene.expandedBracket === null,
      'a collapsed stack must not expose branch or bracket geometry'
    );
  }

  add(
    scene.flow.minimumY < scene.bounds.structuralY[0] &&
      scene.flow.maximumY > scene.bounds.structuralY[1],
    'the compute rail must span the full stack'
  );

  return errors;
}
