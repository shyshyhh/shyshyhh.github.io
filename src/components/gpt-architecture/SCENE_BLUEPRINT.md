# GPT visualization inspection contracts

## Primary block trace

The default block view is intentionally DOM, not WebGL. It exposes the live
computation as one input row followed by six operation rows. Every row contains
the full eight-number state for the selected token.

Tests can inspect it without screenshots:

```css
.gptx-block-trace[data-axis-contract="compute=top-to-bottom"]
.gptx-block-trace[data-vector-size="8"]
.gptx-block-trace[data-selected-operation]
.gptx-trace-operations > li
.gptx-trace-vector > span
```

The selected token and block are ordinary buttons. Clicking Attention or
SwiGLU opens the corresponding spatial lab; the primary forward-pass
explanation itself remains flat and literal.

## Legacy stack geometry contract

`stack-scene-blueprint.mjs` remains the inspectable specification for the old
3D stack and a regression fixture for layout rules. It is not mounted in the
default block view.

## Spatial contract

- `x` is token sequence position, left to right.
- `y` is model compute depth, top to bottom.
- `z` is reserved for parallel branches, such as residual bypasses.
- Serial operations must never advance on `z`.

An open transformer block therefore exposes this strictly decreasing `y`
sequence:

1. RMSNorm
2. attention
3. residual add
4. RMSNorm
5. SwiGLU
6. residual add

Exactly one blueprint operation carries `selected: true`, and its stable
`operationId` is also stored at `selection.operationId`. The rendered canvas
mirrors it in `data-selected-operation`, so tests can verify both the geometry
and the active drill-down without reading pixels.

## Inspect without rendering

Print a readable front elevation:

```sh
npm run inspect:scene -- --layer=3 --tokens=6
```

Print the same orthographic projection as JSON:

```sh
npm run inspect:scene -- --layer=3 --tokens=6 --json
```

Check every combination of open/closed state, selected block, and supported
token count:

```sh
npm run check:scene
```

Screenshots remain useful for presentation-only problems such as label
collisions. They are not needed to verify the default trace’s order, values, or
selection state.
