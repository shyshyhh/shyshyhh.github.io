# GPT stack scene blueprint

The stack view is data before it is WebGL. Its source of truth is
`stack-scene-blueprint.mjs`; `GPTScene.jsx` only turns that data into Three.js
objects.

## Spatial contract

- `x` is token sequence position, left to right.
- `y` is model compute depth, bottom to top.
- `z` is reserved for parallel branches, such as residual bypasses.
- Serial operations must never advance on `z`.

An open transformer block therefore exposes this strictly increasing `y`
sequence:

1. RMSNorm
2. attention
3. residual add
4. RMSNorm
5. SwiGLU
6. residual add

## Inspect without rendering

Print a readable front elevation:

```sh
npm run inspect:scene -- --layer=3 --tokens=6
```

Print the same orthographic projection as JSON:

```sh
npm run inspect:scene -- --layer=3 --tokens=6 --json
```

Check every combination of open/closed state, selected layer, and supported
token count:

```sh
npm run check:scene
```

The hydrated page embeds the lossless current scene and its front elevation in:

```css
script[data-gptx-scene-blueprint="full"]
script[data-gptx-scene-blueprint="front-elevation"]
```

and exposes the axis contract on `.gptx-canvas[data-axis-contract]`. Browser
tests can inspect those values without reading pixels or manipulating the
camera.

Screenshots remain useful for presentation-only problems such as label
collisions. They are not needed to verify geometry or flow semantics.
