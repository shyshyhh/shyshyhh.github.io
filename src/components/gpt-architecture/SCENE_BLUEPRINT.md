# GPT stack scene blueprint

The stack view is data before it is WebGL. Its source of truth is
`stack-scene-blueprint.mjs`; `GPTScene.jsx` only turns that data into Three.js
objects.

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

Exactly one operation carries `selected: true`, and its stable
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

The hydrated page embeds the lossless current scene and its front elevation in:

```css
script[data-gptx-scene-blueprint="full"]
script[data-gptx-scene-blueprint="front-elevation"]
```

and exposes the directional axis contract
`x=sequence-right;y=compute-down;z=parallel-branches` on
`.gptx-canvas[data-axis-contract]`. Browser
tests can inspect those values without reading pixels or manipulating the
camera.

Screenshots remain useful for presentation-only problems such as label
collisions. They are not needed to verify geometry or flow semantics.
