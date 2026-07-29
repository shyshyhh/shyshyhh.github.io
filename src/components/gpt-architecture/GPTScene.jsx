import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import {
  Float,
  Html,
  Line,
  OrbitControls,
  RoundedBox,
  Sparkles,
} from '@react-three/drei';
import * as THREE from 'three';

const COLORS = {
  cyan: '#70e8ff',
  blue: '#6d8dff',
  violet: '#ad83ff',
  coral: '#ff8274',
  gold: '#ffd166',
  mint: '#69f0bd',
  ink: '#0a1020',
  dim: '#34405d',
  white: '#f4f7ff',
};

export const HEAD_COLORS = [
  COLORS.cyan,
  COLORS.violet,
  COLORS.coral,
  COLORS.gold,
];

const CAMERA_PRESETS = {
  stack: { position: [10.5, 7.2, 12.5], target: [0, 0, 0] },
  attention: { position: [0, 1.2, 13.5], target: [0, 0.1, 0] },
  rope: { position: [0, 0.5, 13.2], target: [0, 0, 0] },
  gqa: { position: [0, 0.4, 12.8], target: [0, 0, 0] },
  cache: { position: [10.5, 6.8, 11.8], target: [0, 0, 0] },
  mlp: { position: [0, 1.2, 13], target: [0, 0, 0] },
  weights: { position: [9.5, 10.2, 11.8], target: [0, 0, 0] },
};

function SceneLabel({ children, position, tone = 'normal', className = '' }) {
  return (
    <Html
      center
      position={position}
      distanceFactor={10}
      style={{ pointerEvents: 'none' }}
      zIndexRange={[20, 0]}
    >
      <span
        aria-hidden="true"
        className={`gptx-world-label gptx-world-label--${tone} ${className}`}
      >
        {children}
      </span>
    </Html>
  );
}

function setPointer(active) {
  document.body.style.cursor = active ? 'pointer' : '';
}

function GlassBox({
  args,
  position,
  color = COLORS.cyan,
  active = false,
  opacity = 0.22,
  onClick,
  rotation,
  children,
}) {
  const [hovered, setHovered] = useState(false);
  const selected = active || hovered;

  return (
    <RoundedBox
      args={args}
      position={position}
      rotation={rotation}
      radius={Math.min(...args) * 0.16}
      smoothness={4}
      onClick={
        onClick
          ? (event) => {
              event.stopPropagation();
              onClick();
            }
          : undefined
      }
      onPointerOver={
        onClick
          ? (event) => {
              event.stopPropagation();
              setHovered(true);
              setPointer(true);
            }
          : undefined
      }
      onPointerOut={
        onClick
          ? () => {
              setHovered(false);
              setPointer(false);
            }
          : undefined
      }
    >
      <meshPhysicalMaterial
        color={color}
        emissive={color}
        emissiveIntensity={selected ? 0.42 : 0.09}
        transparent
        opacity={selected ? Math.min(opacity + 0.18, 0.72) : opacity}
        roughness={0.2}
        metalness={0.08}
        clearcoat={0.7}
        clearcoatRoughness={0.2}
        depthWrite={false}
      />
      {children}
    </RoundedBox>
  );
}

function DataOrb({
  position,
  color = COLORS.cyan,
  radius = 0.18,
  active = true,
  onClick,
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <mesh
      position={position}
      onClick={
        onClick
          ? (event) => {
              event.stopPropagation();
              onClick();
            }
          : undefined
      }
      onPointerOver={
        onClick
          ? (event) => {
              event.stopPropagation();
              setHovered(true);
              setPointer(true);
            }
          : undefined
      }
      onPointerOut={
        onClick
          ? () => {
              setHovered(false);
              setPointer(false);
            }
          : undefined
      }
      scale={hovered ? 1.2 : 1}
    >
      <sphereGeometry args={[radius, 24, 24]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={active ? 1.25 : 0.14}
        transparent
        opacity={active ? 0.98 : 0.32}
      />
      {active && (
        <pointLight color={color} intensity={1.5} distance={2.5} decay={2} />
      )}
    </mesh>
  );
}

function CameraRig({ mode, resetKey, controls, reducedMotion }) {
  const { camera } = useThree();
  const destination = useRef(new THREE.Vector3());
  const target = useRef(new THREE.Vector3());
  const moving = useRef(true);

  useEffect(() => {
    const preset = CAMERA_PRESETS[mode] ?? CAMERA_PRESETS.stack;
    destination.current.set(...preset.position);
    target.current.set(...preset.target);
    moving.current = true;
  }, [mode, resetKey]);

  useFrame((_, delta) => {
    if (!moving.current) return;
    if (reducedMotion) {
      camera.position.copy(destination.current);
      if (controls.current) {
        controls.current.target.copy(target.current);
        controls.current.update();
      }
      moving.current = false;
      return;
    }
    const alpha = 1 - Math.exp(-delta * 4.5);
    camera.position.lerp(destination.current, alpha);
    if (controls.current) {
      controls.current.target.lerp(target.current, alpha);
      controls.current.update();
    }
    if (
      camera.position.distanceTo(destination.current) < 0.035 &&
      (!controls.current ||
        controls.current.target.distanceTo(target.current) < 0.035)
    ) {
      moving.current = false;
    }
  });

  return null;
}

function FlowOrb({ x, minimum, maximum, color, reducedMotion }) {
  const ref = useRef();
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const progress = reducedMotion
      ? 0.72
      : (clock.getElapsedTime() * 0.16) % 1;
    ref.current.position.y = THREE.MathUtils.lerp(minimum, maximum, progress);
    ref.current.material.opacity =
      reducedMotion ? 0.9 : 0.55 + Math.sin(progress * Math.PI) * 0.45;
  });

  return (
    <mesh ref={ref} position={[x, minimum, 0.3]}>
      <sphereGeometry args={[0.16, 20, 20]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={2.2}
        transparent
      />
      <pointLight color={color} intensity={2} distance={2.5} />
    </mesh>
  );
}

function StackScene({
  model,
  selectedLayer,
  selectedToken,
  exploded,
  onLayerSelect,
  onTokenSelect,
  reducedMotion,
}) {
  const { tokens, layers } = model;
  const spacing = Math.min(1.08, 6.2 / Math.max(tokens.length - 1, 1));
  const tokenX = (index) => (index - (tokens.length - 1) / 2) * spacing;
  const layerGap = exploded ? 2.35 : 1.55;
  const bottom = -((layers.length - 1) * layerGap) / 2;
  const layerY = (index) => bottom + index * layerGap;
  const plateWidth = Math.max(6.5, tokens.length * spacing + 1.5);
  const selectedOutput = layers[selectedLayer].output;
  const maxNorm = Math.max(
    ...layers.flatMap((layer) => layer.norms.output),
    0.001
  );
  const operationRail = [
    { label: 'RMS', z: 2.45, color: COLORS.blue },
    { label: 'ATTN', z: 1.48, color: COLORS.violet },
    { label: '+', z: 0.5, color: COLORS.cyan },
    { label: 'RMS', z: -0.5, color: COLORS.blue },
    { label: 'SWIGLU', z: -1.48, color: COLORS.coral },
    { label: '+', z: -2.45, color: COLORS.mint },
  ];

  return (
    <group>
      {tokens.map((token, tokenIndex) => (
        <React.Fragment key={`rail-${token}-${tokenIndex}`}>
          <Line
            points={[
              [tokenX(tokenIndex), bottom - 1.15, 0],
              [tokenX(tokenIndex), layerY(layers.length - 1) + 1.3, 0],
            ]}
            color={tokenIndex === selectedToken ? COLORS.cyan : COLORS.dim}
            transparent
            opacity={tokenIndex === selectedToken ? 0.72 : 0.17}
            lineWidth={tokenIndex === selectedToken ? 1.8 : 0.7}
          />
          <SceneLabel
            position={[tokenX(tokenIndex), bottom - 1.45, 0]}
            tone={tokenIndex === selectedToken ? 'active' : 'muted'}
          >
            {token}
          </SceneLabel>
          <DataOrb
            position={[tokenX(tokenIndex), bottom - 1.05, 0]}
            color={tokenIndex === selectedToken ? COLORS.cyan : COLORS.blue}
            radius={tokenIndex === selectedToken ? 0.16 : 0.11}
            active={tokenIndex === selectedToken}
            onClick={() => onTokenSelect(tokenIndex)}
          />
        </React.Fragment>
      ))}

      <FlowOrb
        x={tokenX(selectedToken)}
        minimum={bottom - 0.9}
        maximum={layerY(layers.length - 1) + 1.2}
        color={COLORS.cyan}
        reducedMotion={reducedMotion}
      />

      {layers.map((layer, layerIndex) => {
        const active = selectedLayer === layerIndex;
        const y = layerY(layerIndex);
        return (
          <group key={`layer-${layerIndex}`}>
            <GlassBox
              args={[plateWidth, active ? 0.25 : 0.16, 2.65]}
              position={[0, y, 0]}
              color={active ? COLORS.cyan : COLORS.blue}
              active={active}
              opacity={active ? 0.24 : 0.09}
              onClick={() => onLayerSelect(layerIndex)}
            />
            <SceneLabel
              position={[-plateWidth / 2 - 0.7, y, 0]}
              tone={active ? 'active' : 'muted'}
            >
              L{layerIndex + 1}
            </SceneLabel>

            {tokens.map((token, tokenIndex) => {
              const norm = layer.norms.output[tokenIndex] / maxNorm;
              return (
                <GlassBox
                  key={`${layerIndex}-${tokenIndex}-${token}`}
                  args={[0.52, 0.28 + norm * 0.18, 0.52]}
                  position={[tokenX(tokenIndex), y + 0.18, 0]}
                  color={
                    tokenIndex === selectedToken
                      ? HEAD_COLORS[layerIndex % HEAD_COLORS.length]
                      : COLORS.blue
                  }
                  active={active && tokenIndex === selectedToken}
                  opacity={0.2 + norm * 0.26}
                  onClick={() => {
                    onLayerSelect(layerIndex);
                    onTokenSelect(tokenIndex);
                  }}
                />
              );
            })}

            {exploded &&
              active &&
              operationRail.map((operation) => (
                <group key={operation.label + operation.z}>
                  <GlassBox
                    args={[plateWidth - 0.9, 0.13, 0.5]}
                    position={[0, y, operation.z]}
                    color={operation.color}
                    opacity={0.22}
                  />
                  <SceneLabel
                    position={[
                      plateWidth / 2 - 0.1,
                      y + 0.08,
                      operation.z,
                    ]}
                    tone="active"
                  >
                    {operation.label}
                  </SceneLabel>
                </group>
              ))}
          </group>
        );
      })}

      <Float
        speed={reducedMotion ? 0 : 1.1}
        rotationIntensity={reducedMotion ? 0 : 0.08}
        floatIntensity={reducedMotion ? 0 : 0.18}
      >
        <GlassBox
          args={[2.8, 0.42, 1.35]}
          position={[0, layerY(layers.length - 1) + 1.35, 0]}
          color={COLORS.mint}
          opacity={0.28}
        />
        <SceneLabel
          position={[0, layerY(layers.length - 1) + 1.7, 0]}
          tone="active"
        >
          logits → {model.predictions[0].token}
        </SceneLabel>
      </Float>

      <SceneLabel
        position={[
          tokenX(selectedToken),
          layerY(selectedLayer) + 0.72,
          exploded ? -2.7 : 0,
        ]}
        tone="active"
      >
        ‖x‖ {Math.sqrt(selectedOutput[selectedToken].reduce((sum, value) => sum + value ** 2, 0)).toFixed(2)}
      </SceneLabel>
    </group>
  );
}

function AttentionScene({
  model,
  selectedLayer,
  selectedToken,
  selectedHead,
  onTokenSelect,
  onHeadSelect,
}) {
  const { tokens } = model;
  const layer = model.layers[selectedLayer];
  const spacing = Math.min(1.3, 7 / Math.max(tokens.length - 1, 1));
  const xForToken = (index) => (index - (tokens.length - 1) / 2) * spacing;
  const xForHead = (index) =>
    (index - (model.config.queryHeads - 1) / 2) * 1.55;

  return (
    <group>
      <SceneLabel position={[0, 3.35, 0]} tone="muted">
        query from “{tokens[selectedToken]}”
      </SceneLabel>
      <DataOrb position={[0, 2.9, 0]} color={COLORS.white} radius={0.22} />
      <Line
        points={[
          [0, 2.68, 0],
          [0, 2.25, 0],
        ]}
        color={COLORS.white}
        transparent
        opacity={0.45}
      />

      {Array.from({ length: model.config.queryHeads }, (_, head) => {
        const active = head === selectedHead;
        const headX = xForHead(head);
        return (
          <group key={`head-${head}`}>
            <Line
              points={[
                [0, 2.25, 0],
                [headX, 1.65, 0],
              ]}
              color={HEAD_COLORS[head]}
              transparent
              opacity={active ? 0.75 : 0.18}
              lineWidth={active ? 1.5 : 0.65}
            />
            <DataOrb
              position={[headX, 1.55, 0]}
              color={HEAD_COLORS[head]}
              radius={active ? 0.25 : 0.16}
              active={active}
              onClick={() => onHeadSelect(head)}
            />
            <SceneLabel
              position={[headX, 1.05, 0]}
              tone={active ? 'active' : 'muted'}
            >
              Q{head + 1}
            </SceneLabel>

            {tokens.map((_, keyPosition) => {
              const allowed = keyPosition <= selectedToken;
              const weight =
                layer.attentionWeights[head][selectedToken][keyPosition] ?? 0;
              if (!allowed || weight < 0.002) return null;
              return (
                <Line
                  key={`edge-${head}-${keyPosition}`}
                  points={[
                    [headX, 1.33, 0],
                    [xForToken(keyPosition), -1.65, 0],
                  ]}
                  color={HEAD_COLORS[head]}
                  transparent
                  opacity={(active ? 0.22 : 0.035) + weight * (active ? 1.4 : 0.25)}
                  lineWidth={active ? 0.8 + weight * 4.5 : 0.45}
                />
              );
            })}
          </group>
        );
      })}

      {tokens.map((token, tokenIndex) => {
        const future = tokenIndex > selectedToken;
        const weight =
          layer.attentionWeights[selectedHead][selectedToken][tokenIndex] ?? 0;
        return (
          <group key={`key-${token}-${tokenIndex}`}>
            <GlassBox
              args={[0.78, 0.34, 0.78]}
              position={[xForToken(tokenIndex), -1.9, 0]}
              color={
                future
                  ? COLORS.dim
                  : tokenIndex === selectedToken
                    ? COLORS.cyan
                    : HEAD_COLORS[selectedHead]
              }
              active={!future && weight === Math.max(...layer.attentionWeights[selectedHead][selectedToken])}
              opacity={future ? 0.07 : 0.12 + weight * 0.5}
              onClick={() => onTokenSelect(tokenIndex)}
            />
            <SceneLabel
              position={[xForToken(tokenIndex), -2.45, 0]}
              tone={future ? 'muted' : tokenIndex === selectedToken ? 'active' : 'normal'}
            >
              {token}
            </SceneLabel>
            <SceneLabel
              position={[xForToken(tokenIndex), -1.47, 0]}
              tone={future ? 'muted' : 'active'}
            >
              {future ? 'masked' : `${Math.round(weight * 100)}%`}
            </SceneLabel>
          </group>
        );
      })}

      <Line
        points={[
          [xForToken(selectedToken) + 0.45, -2.9, 0],
          [xForToken(tokens.length - 1) + 0.45, -2.9, 0],
        ]}
        color={COLORS.coral}
        transparent
        opacity={selectedToken < tokens.length - 1 ? 0.5 : 0}
        lineWidth={1.1}
      />
      {selectedToken < tokens.length - 1 && (
        <SceneLabel
          position={[
            (xForToken(selectedToken) + xForToken(tokens.length - 1)) / 2,
            -3.18,
            0,
          ]}
          tone="muted"
        >
          future blocked
        </SceneLabel>
      )}
    </group>
  );
}

function RopeClock({ position, vector, color, label, phaseLabel }) {
  const angle = Math.atan2(vector[1], vector[0]);
  const radius = 1.55;
  const endpoint = [Math.cos(angle) * radius, Math.sin(angle) * radius, 0.04];
  const circle = useMemo(
    () =>
      Array.from({ length: 65 }, (_, index) => {
        const theta = (index / 64) * Math.PI * 2;
        return [Math.cos(theta) * radius, Math.sin(theta) * radius, 0];
      }),
    []
  );

  return (
    <group position={position}>
      <Line
        points={circle}
        color={COLORS.dim}
        transparent
        opacity={0.55}
        lineWidth={1}
      />
      <Line
        points={[
          [-radius, 0, 0],
          [radius, 0, 0],
        ]}
        color={COLORS.dim}
        transparent
        opacity={0.22}
        lineWidth={0.6}
      />
      <Line
        points={[
          [0, -radius, 0],
          [0, radius, 0],
        ]}
        color={COLORS.dim}
        transparent
        opacity={0.22}
        lineWidth={0.6}
      />
      <Line
        points={[
          [0, 0, 0.03],
          endpoint,
        ]}
        color={color}
        lineWidth={3}
      />
      <DataOrb position={endpoint} color={color} radius={0.12} />
      <SceneLabel position={[0, radius + 0.55, 0]} tone="active">
        {label}
      </SceneLabel>
      <SceneLabel position={[0, -radius - 0.55, 0]} tone="muted">
        {phaseLabel}
      </SceneLabel>
    </group>
  );
}

function RopeScene({ model, selectedLayer, selectedToken, selectedHead }) {
  const layer = model.layers[selectedLayer];
  const kvHead = Math.floor(
    selectedHead / (model.config.queryHeads / model.config.kvHeads)
  );
  const keyPosition = Math.max(0, selectedToken - 2);
  const query = layer.query[selectedToken][selectedHead];
  const key = layer.key[keyPosition][kvHead];
  const relative = selectedToken - keyPosition;
  const match =
    query.reduce((sum, value, index) => sum + value * key[index], 0) /
    Math.sqrt(model.config.headDim);

  return (
    <group>
      <RopeClock
        position={[-2.65, 0.2, 0]}
        vector={query}
        color={COLORS.cyan}
        label={`Q · ${model.tokens[selectedToken]}`}
        phaseLabel={`position ${selectedToken}`}
      />
      <RopeClock
        position={[2.65, 0.2, 0]}
        vector={key}
        color={COLORS.violet}
        label={`K · ${model.tokens[keyPosition]}`}
        phaseLabel={`position ${keyPosition}`}
      />
      <Line
        points={[
          [-0.85, 0.2, 0],
          [0.85, 0.2, 0],
        ]}
        color={COLORS.gold}
        transparent
        opacity={0.75}
        lineWidth={2}
      />
      <SceneLabel position={[0, 0.65, 0]} tone="active">
        Δ position = {relative}
      </SceneLabel>
      <SceneLabel position={[0, -0.2, 0]} tone="muted">
        scaled match {match.toFixed(3)}
      </SceneLabel>
      <SceneLabel position={[0, -2.75, 0]} tone="normal">
        rotate Q and K, never V
      </SceneLabel>
    </group>
  );
}

function GQAScene({ variant, selectedHead, onHeadSelect }) {
  const queryHeads = 4;
  const kvHeads = variant === 'mha' ? 4 : variant === 'mqa' ? 1 : 2;
  const groupSize = queryHeads / kvHeads;
  const queryX = (index) => (index - (queryHeads - 1) / 2) * 1.65;
  const kvX = (index) => (index - (kvHeads - 1) / 2) * 2.35;

  return (
    <group>
      <SceneLabel position={[0, 3, 0]} tone="muted">
        four independent queries
      </SceneLabel>
      {Array.from({ length: queryHeads }, (_, head) => {
        const kvHead = Math.floor(head / groupSize);
        return (
          <group key={`query-${head}`}>
            <DataOrb
              position={[queryX(head), 2.15, 0]}
              color={HEAD_COLORS[head]}
              radius={head === selectedHead ? 0.28 : 0.19}
              active={head === selectedHead}
              onClick={() => onHeadSelect(head)}
            />
            <SceneLabel
              position={[queryX(head), 2.65, 0]}
              tone={head === selectedHead ? 'active' : 'muted'}
            >
              Q{head + 1}
            </SceneLabel>
            <Line
              points={[
                [queryX(head), 1.92, 0],
                [kvX(kvHead), -1.65, 0],
              ]}
              color={HEAD_COLORS[head]}
              transparent
              opacity={head === selectedHead ? 0.88 : 0.22}
              lineWidth={head === selectedHead ? 2.3 : 0.8}
            />
          </group>
        );
      })}

      {Array.from({ length: kvHeads }, (_, head) => (
        <group key={`kv-${head}`}>
          <GlassBox
            args={[1.22, 0.62, 1.05]}
            position={[kvX(head), -1.95, 0]}
            color={kvHeads === 1 ? COLORS.gold : COLORS.violet}
            active={Math.floor(selectedHead / groupSize) === head}
            opacity={0.32}
          />
          <SceneLabel position={[kvX(head), -1.9, 0]} tone="active">
            K{head + 1} / V{head + 1}
          </SceneLabel>
        </group>
      ))}

      <SceneLabel position={[0, -3, 0]} tone="muted">
        {variant === 'mha'
          ? 'MHA · one memory per query'
          : variant === 'mqa'
            ? 'MQA · one shared memory'
            : 'GQA · two shared memories'}
      </SceneLabel>
    </group>
  );
}

function CacheScene({ tokens, prefillCount, currentCount, reducedMotion }) {
  const layers = 4;
  const visibleTokens = tokens.slice(0, currentCount);
  const spacing = Math.min(0.78, 6.8 / Math.max(visibleTokens.length - 1, 1));
  const tokenX = (index) =>
    (index - (visibleTokens.length - 1) / 2) * spacing;
  const newest = visibleTokens.length - 1;
  const layerY = (index) => (index - (layers - 1) / 2) * 1.35;

  return (
    <group>
      {Array.from({ length: layers }, (_, layer) => (
        <group key={`cache-layer-${layer}`}>
          <Line
            points={[
              [-4.5, layerY(layer) - 0.42, -0.85],
              [4.5, layerY(layer) - 0.42, -0.85],
            ]}
            color={COLORS.dim}
            transparent
            opacity={0.5}
            lineWidth={1}
          />
          <SceneLabel position={[-4.85, layerY(layer), 0]} tone="muted">
            L{layer + 1}
          </SceneLabel>
          {visibleTokens.map((token, index) => {
            const generated = index >= prefillCount;
            const active = index === newest && generated;
            return (
              <group key={`${layer}-${token}-${index}`}>
                <GlassBox
                  args={[0.52, 0.42, 0.52]}
                  position={[tokenX(index), layerY(layer), -0.48]}
                  color={generated ? COLORS.gold : COLORS.violet}
                  active={active}
                  opacity={active ? 0.55 : generated ? 0.3 : 0.18}
                />
                <GlassBox
                  args={[0.52, 0.42, 0.52]}
                  position={[tokenX(index), layerY(layer), 0.48]}
                  color={generated ? COLORS.mint : COLORS.cyan}
                  active={active}
                  opacity={active ? 0.55 : generated ? 0.3 : 0.18}
                />
              </group>
            );
          })}
        </group>
      ))}

      {visibleTokens.map((token, index) => (
        <SceneLabel
          key={`cache-label-${token}-${index}`}
          position={[tokenX(index), layerY(0) - 1.05, 0]}
          tone={index >= prefillCount ? 'active' : 'muted'}
        >
          {token}
        </SceneLabel>
      ))}
      <SceneLabel position={[4.65, layerY(layers - 1) + 0.22, -0.48]} tone="muted">
        K
      </SceneLabel>
      <SceneLabel position={[4.65, layerY(layers - 1) + 0.22, 0.48]} tone="muted">
        V
      </SceneLabel>

      {newest >= prefillCount && (
        <Float
          speed={reducedMotion ? 0 : 1.4}
          floatIntensity={reducedMotion ? 0 : 0.35}
          rotationIntensity={0}
        >
          <DataOrb
            position={[tokenX(newest), layerY(layers - 1) + 1.05, 0]}
            color={COLORS.gold}
            radius={0.18}
          />
          <SceneLabel
            position={[tokenX(newest), layerY(layers - 1) + 1.48, 0]}
            tone="active"
          >
            new Q scans cached K
          </SceneLabel>
        </Float>
      )}
    </group>
  );
}

function FeatureBank({ values, position, color, label }) {
  const maximum = Math.max(...values.map((value) => Math.abs(value)), 0.001);
  const spacing = 0.26;
  const columns = Math.min(8, values.length);
  const rows = Math.ceil(values.length / columns);
  return (
    <group position={position}>
      {values.map((value, index) => {
        const height = 0.15 + (Math.abs(value) / maximum) * 1.2;
        const row = Math.floor(index / columns);
        const column = index % columns;
        const rowLength = Math.min(columns, values.length - row * columns);
        return (
          <mesh
            key={`${label}-${index}`}
            position={[
              (column - (rowLength - 1) / 2) * spacing,
              value >= 0 ? height / 2 : -height / 2,
              (row - (rows - 1) / 2) * 0.42,
            ]}
          >
            <boxGeometry args={[0.16, height, 0.32]} />
            <meshStandardMaterial
              color={value >= 0 ? color : COLORS.coral}
              emissive={value >= 0 ? color : COLORS.coral}
              emissiveIntensity={0.35 + Math.abs(value / maximum) * 0.65}
              transparent
              opacity={0.8}
            />
          </mesh>
        );
      })}
      <SceneLabel position={[0, -1.55, 0]} tone="muted">
        {label} · all {values.length}
      </SceneLabel>
    </group>
  );
}

function MLPScene({ model, selectedLayer, selectedToken }) {
  const layer = model.layers[selectedLayer];
  const gate = layer.gate[selectedToken];
  const activatedGate = gate.map(
    (value) => value / (1 + Math.exp(-value))
  );
  const up = layer.up[selectedToken];
  const swiglu = layer.swiglu[selectedToken];

  return (
    <group>
      <DataOrb position={[-4.25, 0, 0]} color={COLORS.white} radius={0.25} />
      <SceneLabel position={[-4.25, 0.55, 0]} tone="muted">
        RMSNorm(x)
      </SceneLabel>

      <Line
        points={[
          [-4, 0, 0],
          [-2.6, 1.45, 0],
        ]}
        color={COLORS.violet}
        transparent
        opacity={0.65}
        lineWidth={1.6}
      />
      <Line
        points={[
          [-4, 0, 0],
          [-2.6, -1.45, 0],
        ]}
        color={COLORS.cyan}
        transparent
        opacity={0.65}
        lineWidth={1.6}
      />
      <FeatureBank
        values={activatedGate}
        position={[-1.75, 1.45, 0]}
        color={COLORS.violet}
        label="SiLU(gate)"
      />
      <FeatureBank
        values={up}
        position={[-1.75, -1.45, 0]}
        color={COLORS.cyan}
        label="up"
      />

      <Line
        points={[
          [-0.55, 1.45, 0],
          [0.35, 0.3, 0],
        ]}
        color={COLORS.violet}
        transparent
        opacity={0.7}
        lineWidth={1.6}
      />
      <Line
        points={[
          [-0.55, -1.45, 0],
          [0.35, -0.3, 0],
        ]}
        color={COLORS.cyan}
        transparent
        opacity={0.7}
        lineWidth={1.6}
      />

      <GlassBox
        args={[0.72, 0.72, 0.72]}
        position={[0.6, 0, 0]}
        color={COLORS.gold}
        active
        opacity={0.4}
      />
      <SceneLabel position={[0.6, 0, 0]} tone="active">
        ×
      </SceneLabel>

      <Line
        points={[
          [0.98, 0, 0],
          [2.05, 0, 0],
        ]}
        color={COLORS.gold}
        transparent
        opacity={0.8}
        lineWidth={2}
      />
      <FeatureBank
        values={swiglu}
        position={[2.25, 0, 0]}
        color={COLORS.gold}
        label="signed gated features"
      />
      <Line
        points={[
          [3.45, 0, 0],
          [4.15, 0, 0],
        ]}
        color={COLORS.mint}
        transparent
        opacity={0.85}
        lineWidth={2}
      />
      <DataOrb position={[4.35, 0, 0]} color={COLORS.mint} radius={0.24} />
      <SceneLabel position={[4.35, 0.55, 0]} tone="muted">
        Wdown → + residual
      </SceneLabel>
    </group>
  );
}

function MatrixScene({ matrix, color, selectedCell, onCellSelect }) {
  const rows = Array.isArray(matrix?.[0]) ? matrix.length : 1;
  const columns = Array.isArray(matrix?.[0]) ? matrix[0].length : matrix?.length ?? 0;
  const values = rows === 1 ? [matrix ?? []] : matrix;
  const maximum = Math.max(
    ...values
      .flat()
      .filter(Number.isFinite)
      .map((value) => Math.abs(value)),
    0.001
  );
  const cell = Math.min(0.48, 6.8 / Math.max(rows, columns));
  const xFor = (column) => (column - (columns - 1) / 2) * (cell + 0.08);
  const zFor = (row) => (row - (rows - 1) / 2) * (cell + 0.08);

  return (
    <group>
      {values.map((row, rowIndex) =>
        row.map((value, columnIndex) => {
          const strength = Number.isFinite(value)
            ? Math.abs(value) / maximum
            : 1;
          const height = 0.12 + strength * 1.15;
          const active =
            selectedCell?.row === rowIndex &&
            selectedCell?.column === columnIndex;
          return (
            <group key={`${rowIndex}-${columnIndex}`}>
              <GlassBox
                args={[cell, height, cell]}
                position={[
                  xFor(columnIndex),
                  value >= 0 ? height / 2 : -height / 2,
                  zFor(rowIndex),
                ]}
                color={value >= 0 ? color : COLORS.coral}
                active={active}
                opacity={0.18 + strength * 0.35}
                onClick={() =>
                  onCellSelect?.({
                    row: rowIndex,
                    column: columnIndex,
                    value,
                  })
                }
              />
            </group>
          );
        })
      )}
      <Line
        points={[
          [xFor(0) - cell, 0, zFor(0) - cell],
          [xFor(columns - 1) + cell, 0, zFor(0) - cell],
          [xFor(columns - 1) + cell, 0, zFor(rows - 1) + cell],
          [xFor(0) - cell, 0, zFor(rows - 1) + cell],
          [xFor(0) - cell, 0, zFor(0) - cell],
        ]}
        color={COLORS.dim}
        transparent
        opacity={0.6}
        lineWidth={1}
      />
      <SceneLabel
        position={[0, 1.85, zFor(0) - 0.8]}
        tone="muted"
      >
        {rows} × {columns} · positive rises, negative drops
      </SceneLabel>
    </group>
  );
}

function World({
  mode,
  model,
  selectedLayer,
  selectedToken,
  selectedHead,
  exploded,
  variant,
  cacheTokens,
  prefillCount,
  cacheCount,
  matrix,
  matrixColor,
  selectedCell,
  onLayerSelect,
  onTokenSelect,
  onHeadSelect,
  onCellSelect,
  reducedMotion,
  resetKey,
}) {
  const controls = useRef();

  useEffect(() => () => setPointer(false), []);

  return (
    <>
      <color attach="background" args={['#070b16']} />
      <fog attach="fog" args={['#070b16', 12, 30]} />
      <ambientLight intensity={0.48} color="#b8c7ff" />
      <directionalLight
        position={[7, 10, 8]}
        intensity={1.2}
        color="#dce7ff"
      />
      <pointLight position={[-7, 2, 5]} intensity={16} color="#6248ff" />
      <pointLight position={[7, -2, 3]} intensity={13} color="#10bcd4" />
      <Sparkles
        count={reducedMotion ? 35 : 90}
        scale={[18, 12, 12]}
        size={1.1}
        speed={reducedMotion ? 0 : 0.12}
        opacity={0.28}
        color="#8da2ff"
      />

      <group>
        {mode === 'stack' && (
          <StackScene
            model={model}
            selectedLayer={selectedLayer}
            selectedToken={selectedToken}
            exploded={exploded}
            onLayerSelect={onLayerSelect}
            onTokenSelect={onTokenSelect}
            reducedMotion={reducedMotion}
          />
        )}
        {mode === 'attention' && (
          <AttentionScene
            model={model}
            selectedLayer={selectedLayer}
            selectedToken={selectedToken}
            selectedHead={selectedHead}
            onTokenSelect={onTokenSelect}
            onHeadSelect={onHeadSelect}
          />
        )}
        {mode === 'rope' && (
          <RopeScene
            model={model}
            selectedLayer={selectedLayer}
            selectedToken={selectedToken}
            selectedHead={selectedHead}
          />
        )}
        {mode === 'gqa' && (
          <GQAScene
            variant={variant}
            selectedHead={selectedHead}
            onHeadSelect={onHeadSelect}
          />
        )}
        {mode === 'cache' && (
          <CacheScene
            tokens={cacheTokens}
            prefillCount={prefillCount}
            currentCount={cacheCount}
            reducedMotion={reducedMotion}
          />
        )}
        {mode === 'mlp' && (
          <MLPScene
            model={model}
            selectedLayer={selectedLayer}
            selectedToken={selectedToken}
          />
        )}
        {mode === 'weights' && (
          <MatrixScene
            matrix={matrix}
            color={matrixColor}
            selectedCell={selectedCell}
            onCellSelect={onCellSelect}
          />
        )}
      </group>

      <CameraRig
        mode={mode}
        resetKey={resetKey}
        controls={controls}
        reducedMotion={reducedMotion}
      />
      <OrbitControls
        ref={controls}
        makeDefault
        enableDamping
        dampingFactor={0.075}
        minDistance={6.5}
        maxDistance={22}
        minPolarAngle={0.22}
        maxPolarAngle={Math.PI - 0.25}
        autoRotate={!reducedMotion && mode === 'stack' && !exploded}
        autoRotateSpeed={0.28}
      />
    </>
  );
}

export default function GPTScene(props) {
  return (
    <div
      className="gptx-canvas"
      role="img"
      aria-label={props.accessibleLabel}
    >
      <Canvas
        camera={{ position: CAMERA_PRESETS.stack.position, fov: 43, near: 0.1, far: 80 }}
        dpr={[1, 1.65]}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          toneMapping: THREE.ACESFilmicToneMapping,
        }}
        onPointerMissed={() => setPointer(false)}
      >
        <Suspense fallback={null}>
          <World {...props} />
        </Suspense>
      </Canvas>
    </div>
  );
}
