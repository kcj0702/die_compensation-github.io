'use client';

/**
 * 3D CAD/스캔 뷰어 — 외부 라이브러리 없이 WebGL2 로 직접 그린다.
 *
 * three.js 를 쓰려 했으나 이 저장소의 node_modules 가 pnpm 트리라
 * npm/pnpm 양쪽 다 설치가 깨졌다. 뷰어에 필요한 건 삼각망 하나를
 * 셰이딩해서 돌려보는 것뿐이라 의존성을 늘리는 대신 직접 그린다 —
 * "모든 처리는 이 PC 안에서" 원칙에도 이쪽이 맞다.
 *
 * 평면 셰이딩은 dFdx/dFdy 로 화면공간 미분을 써서 면 법선을 만든다.
 * 정점 법선을 CPU 에서 계산해 보낼 필요가 없어 전송량이 준다.
 */

import { useEffect, useRef, useState } from 'react';

export type CadHole = {
  kind: string; radius: number; diameter: number;
  center: [number, number, number]; axis: [number, number, number];
  height: number; area: number;
};

export type CadMesh = {
  summary: {
    name: string; source_format: string; units: string;
    bounds: { min: number[]; max: number[]; size: number[]; center: number[] };
    n_vertices: number; n_faces: number; n_faces_display: number; watertight: boolean;
  };
  positions: number[];
  indices: number[];
  holes: CadHole[];
  planes: { center: number[]; normal: number[]; area: number }[];
  counts: { cylinders: number; holes: number; planes: number };
  recentered: boolean;
  note?: string;
};

// ─── 행렬 (열 우선, WebGL 관례) ──────────────────────────────────
function perspective(fovY: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAt(eye: number[], target: number[], up: number[]) {
  const z = norm([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

const cross = (a: number[], b: number[]) => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
];
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function norm(v: number[]) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

const VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
uniform mat4 uProjection;
uniform mat4 uView;
out vec3 vViewPos;
void main() {
  vec4 viewPos = uView * vec4(aPosition, 1.0);
  vViewPos = viewPos.xyz;
  gl_Position = uProjection * viewPos;
}`;

// 면 법선을 화면공간 미분으로 만든다 — 정점 법선 전송이 필요 없다.
const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec3 vViewPos;
uniform vec3 uColor;
out vec4 outColor;
void main() {
  vec3 normal = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
  if (!gl_FrontFacing) normal = -normal;
  vec3 lightDir = normalize(vec3(0.4, 0.7, 1.0));
  float diffuse = max(dot(normal, lightDir), 0.0);
  float rim = pow(1.0 - max(dot(normal, vec3(0.0, 0.0, 1.0)), 0.0), 2.0);
  vec3 color = uColor * (0.34 + 0.66 * diffuse) + vec3(0.16) * rim;
  outColor = vec4(color, 1.0);
}`;

const LINE_VERTEX = `#version 300 es
in vec3 aPosition;
uniform mat4 uProjection;
uniform mat4 uView;
void main() { gl_Position = uProjection * uView * vec4(aPosition, 1.0); }`;

const LINE_FRAGMENT = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 outColor;
void main() { outColor = vec4(uColor, 1.0); }`;

function compile(gl: WebGL2RenderingContext, vsSource: string, fsSource: string) {
  const make = (type: number, source: string) => {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`셰이더 컴파일 실패: ${gl.getShaderInfoLog(shader)}`);
    }
    return shader;
  };
  const program = gl.createProgram()!;
  gl.attachShader(program, make(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(program, make(gl.FRAGMENT_SHADER, fsSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`프로그램 링크 실패: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

/** 홀 위치에 그릴 원형 마커의 정점을 만든다 (축에 수직인 원). */
function holeRings(holes: CadHole[], centre: number[], scale: number) {
  const points: number[] = [];
  const SEGMENTS = 32;
  for (const hole of holes) {
    const axis = norm(hole.axis as unknown as number[]);
    // 축에 수직인 임의의 두 벡터
    const helper = Math.abs(axis[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    const u = norm(cross(axis, helper));
    const v = cross(axis, u);
    const r = hole.radius * scale;
    const c = [
      (hole.center[0] - centre[0]),
      (hole.center[1] - centre[1]),
      (hole.center[2] - centre[2]),
    ];
    for (let i = 0; i < SEGMENTS; i += 1) {
      for (const t of [i, i + 1]) {
        const a = (t / SEGMENTS) * Math.PI * 2;
        points.push(
          c[0] + (u[0] * Math.cos(a) + v[0] * Math.sin(a)) * r,
          c[1] + (u[1] * Math.cos(a) + v[1] * Math.sin(a)) * r,
          c[2] + (u[2] * Math.cos(a) + v[2] * Math.sin(a)) * r,
        );
      }
    }
  }
  return new Float32Array(points);
}

export function CadViewer({ mesh, showHoles }: { mesh: CadMesh; showHoles: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);
  // 카메라 상태는 리렌더를 유발하면 안 되므로 ref 로 들고 있는다.
  const camera = useRef({ theta: 0.9, phi: 1.1, distance: 2.4, panX: 0, panY: 0 });
  const showHolesRef = useRef(showHoles);
  showHolesRef.current = showHoles;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) { setError('이 브라우저에서 WebGL2를 쓸 수 없습니다.'); return; }

    let program: WebGLProgram;
    let lineProgram: WebGLProgram;
    try {
      program = compile(gl, VERTEX_SHADER, FRAGMENT_SHADER);
      lineProgram = compile(gl, LINE_VERTEX, LINE_FRAGMENT);
    } catch (err) { setError(String(err)); return; }

    const positions = new Float32Array(mesh.positions);
    const indices = new Uint32Array(mesh.indices);
    // 부품이 크므로 화면 좌표로 정규화한다 (가장 긴 변이 1이 되도록)
    const size = mesh.summary.bounds.size;
    const longest = Math.max(size[0], size[1], size[2]) || 1;
    const scale = 1 / longest;
    for (let i = 0; i < positions.length; i += 1) positions[i] *= scale;

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    const positionLocation = gl.getAttribLocation(program, 'aPosition');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);

    // 홀 마커 — 백엔드가 준 좌표는 원본 위치라 메시와 같은 기준으로 옮긴다
    const ringData = holeRings(mesh.holes || [], mesh.summary.bounds.center, scale);
    const ringVao = gl.createVertexArray();
    gl.bindVertexArray(ringVao);
    const ringBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ringBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, ringData, gl.STATIC_DRAW);
    const ringLocation = gl.getAttribLocation(lineProgram, 'aPosition');
    gl.enableVertexAttribArray(ringLocation);
    gl.vertexAttribPointer(ringLocation, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.086, 0.114, 0.153, 1);

    let frame = 0;
    const render = () => {
      const width = canvas.clientWidth || 1;
      const height = canvas.clientHeight || 1;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const { theta, phi, distance, panX, panY } = camera.current;
      const eye = [
        distance * Math.sin(phi) * Math.cos(theta),
        distance * Math.cos(phi),
        distance * Math.sin(phi) * Math.sin(theta),
      ];
      const target = [panX, panY, 0];
      const view = lookAt([eye[0] + panX, eye[1] + panY, eye[2]], target, [0, 1, 0]);
      const projection = perspective(Math.PI / 4, width / height, 0.01, 100);

      gl.useProgram(program);
      gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uProjection'), false, projection);
      gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uView'), false, view);
      gl.uniform3f(gl.getUniformLocation(program, 'uColor'), 0.42, 0.62, 0.78);
      gl.bindVertexArray(vao);
      gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);

      if (showHolesRef.current && ringData.length > 0) {
        gl.useProgram(lineProgram);
        gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uView'), false, view);
        gl.uniform3f(gl.getUniformLocation(lineProgram, 'uColor'), 1.0, 0.35, 0.2);
        gl.bindVertexArray(ringVao);
        gl.disable(gl.DEPTH_TEST);
        gl.drawArrays(gl.LINES, 0, ringData.length / 3);
        gl.enable(gl.DEPTH_TEST);
      }
      gl.bindVertexArray(null);
      frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);

    // ─── 마우스 조작 ───
    let dragging: 'rotate' | 'pan' | null = null;
    let lastX = 0;
    let lastY = 0;
    const onDown = (e: PointerEvent) => {
      dragging = e.button === 2 || e.shiftKey ? 'pan' : 'rotate';
      lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      const cam = camera.current;
      if (dragging === 'rotate') {
        cam.theta -= dx * 0.008;
        cam.phi = Math.min(Math.PI - 0.05, Math.max(0.05, cam.phi - dy * 0.008));
      } else {
        cam.panX -= dx * 0.0022 * cam.distance;
        cam.panY += dy * 0.0022 * cam.distance;
      }
    };
    const onUp = (e: PointerEvent) => {
      dragging = null;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const cam = camera.current;
      cam.distance = Math.min(20, Math.max(0.25, cam.distance * (1 + Math.sign(e.deltaY) * 0.12)));
    };
    const onContext = (e: Event) => e.preventDefault();

    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('contextmenu', onContext);

    return () => {
      cancelAnimationFrame(frame);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('contextmenu', onContext);
      gl.deleteProgram(program);
      gl.deleteProgram(lineProgram);
    };
  }, [mesh]);

  if (error) return <div className="cad-viewer__error">{error}</div>;
  return <canvas ref={canvasRef} className="cad-viewer__canvas" />;
}
