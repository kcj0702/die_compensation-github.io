'use client';

/**
 * 3D CAD 뷰어 — three.js.
 *
 * 처음엔 WebGL2 로 직접 그렸다. 그때는 node_modules 트리가 깨져
 * three 설치가 안 됐고, 필요한 것도 삼각망 셰이딩 하나뿐이었다.
 * 지금은 실제 부품이 들어오면서 요구가 늘었다 —
 *
 *   001 REINF SIDE OTR.stp (CATIA V5 내보내기, 12.4MB)
 *     삼각형 45,224  정점 36,772
 *     크기 493.5 x 215.2 x 1062.4 mm
 *     원통 220개, 평면 30개
 *
 * 홀 220개를 축 방향에 맞춰 링으로 세우고, 평면 30개를 법선과 함께
 * 보여주고, 삼각망 자체를 눈으로 확인해야 한다. 직접 그리기로는
 * 감당이 안 돼 three 를 설치했다(로컬 node_modules, CDN 아님 —
 * "모든 처리는 이 PC 안에서" 원칙).
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';

export type CadHole = {
  kind: string; radius: number; diameter: number;
  center: [number, number, number]; axis: [number, number, number];
  height: number; area: number;
  wrap?: number; faces?: number;
};

/** /api/cad-overlay 가 돌려주는, CAD 표면 위로 옮겨진 스캔 결과. */
export type CadOverlay = {
  fit: {
    axis: number; sign: number; flip_u: boolean; flip_v: boolean;
    mm_per_px: number; iou: number; reliable: boolean;
  };
  zeroLines: { line_id: number | null; points: [number, number, number][] }[];
  /* 위치만 준다. 보정량은 최종 보정시트가 정하므로 화면이 넣는다. */
  points: { id: string; position: [number, number, number]; value: number }[];
  scanPart?: string | null;
  /* 화면용 정점 하나하나의 스캔 편차(mm). 부품 밖이면 null. */
  surfaceDeviation?: (number | null)[];
  deviationRange?: [number, number] | null;
  /* 컬러바 범위를 벗어나 제외한 판독. 실측 JD_67XX6 에서 +9.00 이
     5건 나왔는데 그 부품 컬러바는 +3.0~-3.0 이다. */
  rejected?: { id: string; value: number }[];
  colorbarLimit?: number | null;
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
  cadId?: string;
  note?: string;
};

export type CadDetail = 'solid' | 'edges' | 'wire';

/** 3D 주석 — 형상 위 한 점에 붙이는 메모. 시트 주석과 달리 3D 좌표라
 *  돌려봐도 그 자리에 남는다. */
export type CadNote = { id: string; at: [number, number, number]; text: string };

const SURFACE = 0x8fa3b4;
const HOLE_TINT = 0xff8b3d;
const PLANE_TINT = 0x35d68a;
const ZERO_TINT = 0xff3b30;      // 제로라인
// 최종 보정시트("보정 적용 내용")의 표기 — 노란 콜아웃, 빨간 점과 지시선.
const CALLOUT_FILL = '#ffef3a';
const CALLOUT_EDGE = '#3a3a3a';
const CALLOUT_TEXT = '#141414';
const MARK_TINT = 0xe01b1b;      // 보정 지점 빨간 점 · 지시선
const NO_DATA = new THREE.Color(0x55606b);   // 스캔 밖 — 회색으로 남긴다

/** 스캔 히트맵과 같은 색 순서: 파랑(-) → 청록 → 초록(0) → 노랑 → 빨강(+).
 *  검사 소프트웨어가 쓰는 배열이라 현업이 바로 알아본다. */
const RAMP: [number, [number, number, number]][] = [
  [0.00, [0.25, 0.13, 0.62]],
  [0.20, [0.13, 0.45, 0.85]],
  [0.38, [0.15, 0.78, 0.78]],
  [0.50, [0.20, 0.78, 0.35]],
  [0.62, [0.85, 0.90, 0.20]],
  [0.80, [0.95, 0.55, 0.12]],
  [1.00, [0.75, 0.10, 0.10]],
];

function rampColor(t: number): [number, number, number] {
  const x = Math.min(Math.max(t, 0), 1);
  for (let i = 1; i < RAMP.length; i += 1) {
    const [stop, colour] = RAMP[i];
    if (x <= stop) {
      const [prevStop, prev] = RAMP[i - 1];
      const k = (x - prevStop) / (stop - prevStop || 1);
      return [prev[0] + (colour[0] - prev[0]) * k,
              prev[1] + (colour[1] - prev[1]) * k,
              prev[2] + (colour[2] - prev[2]) * k];
    }
  }
  return RAMP[RAMP.length - 1][1];
}

/** 표준 뷰 — CATIA 의 정면/우측/평면/등각과 같은 자리. */
const VIEWS: { id: string; label: string; dir: [number, number, number] }[] = [
  { id: 'iso', label: '등각', dir: [1, 0.85, 1.25] },
  { id: 'front', label: '정면', dir: [0, -1, 0] },
  { id: 'right', label: '우측', dir: [1, 0, 0] },
  { id: 'top', label: '평면', dir: [0, 0, 1] },
];

/** 캔버스에 글자를 구워 스프라이트로 만든다.
 *  three 의 텍스트 지오메트리는 폰트 파일을 받아야 해서 쓰지 않는다
 *  (사내망 원칙 — 바깥에서 아무것도 안 받는다). */
function makeLabel(text: string, height: number): THREE.Sprite {
  /* 최종 보정시트("보정 적용 내용")의 표기를 그대로 옮긴다 —
     노란 박스에 검은 숫자, 얇은 검은 테두리. 현업이 시트에서 보던
     모양이라 설명이 필요 없다. */
  const pad = 14;
  const measure = document.createElement('canvas').getContext('2d');
  if (!measure) return new THREE.Sprite();
  const font = '700 52px ui-sans-serif, system-ui, sans-serif';
  measure.font = font;
  const textWidth = Math.ceil(measure.measureText(text).width);

  const canvas = document.createElement('canvas');
  canvas.width = textWidth + pad * 2;
  canvas.height = 76;
  const ctx = canvas.getContext('2d')!;
  const radius = 14;
  const w = canvas.width;
  const h = canvas.height;

  ctx.beginPath();
  ctx.moveTo(radius, 2);
  ctx.arcTo(w - 2, 2, w - 2, h - 2, radius);
  ctx.arcTo(w - 2, h - 2, 2, h - 2, radius);
  ctx.arcTo(2, h - 2, 2, 2, radius);
  ctx.arcTo(2, 2, w - 2, 2, radius);
  ctx.closePath();
  ctx.fillStyle = CALLOUT_FILL;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = CALLOUT_EDGE;
  ctx.stroke();

  ctx.font = font;
  ctx.fillStyle = CALLOUT_TEXT;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillText(text, w / 2, h / 2 + 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, depthTest: false, transparent: true,
  }));
  sprite.scale.set(height * w / h, height, 1);
  return sprite;
}

function makeNote(text: string, height: number): THREE.Sprite {
  /* 주석 상자 — 보정량 콜아웃과 구분되게 파란 테두리에 흰 바탕이다. */
  const pad = 16;
  const measure = document.createElement('canvas').getContext('2d');
  if (!measure) return new THREE.Sprite();
  const font = '600 40px ui-sans-serif, system-ui, sans-serif';
  measure.font = font;
  const body = text || '(빈 메모)';
  const canvas = document.createElement('canvas');
  canvas.width = Math.ceil(measure.measureText(body).width) + pad * 2;
  canvas.height = 62;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgba(248,251,253,.96)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 4;
  ctx.strokeStyle = '#3f7fb8';
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
  ctx.font = font;
  ctx.fillStyle = '#1c2530';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillText(body, canvas.width / 2, canvas.height / 2 + 1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, depthTest: false, transparent: true,
  }));
  sprite.scale.set(height * canvas.width / canvas.height, height, 1);
  return sprite;
}

export function CadViewer({ mesh, showHoles, overlay, sheetValues,
                           onCorrectionChange, notes, onNotesChange }: {
  mesh: CadMesh; showHoles: boolean; overlay?: CadOverlay | null;
  /* 포인트 아이디 -> 최종 보정량(mm). 시트에서 숨긴 포인트는 빠져 있다. */
  sheetValues?: Record<string, number> | null;
  /* 3D 에서 고친 값도 시트와 같은 저장소로 간다 — 양쪽이 어긋나면 안 된다. */
  onCorrectionChange?: (pointId: string, value: number | null) => void;
  notes?: CadNote[];
  onNotesChange?: (notes: CadNote[]) => void;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<CadDetail>('edges');
  const [showPlanes, setShowPlanes] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  /* 홀을 누르면 지름과 좌표를 띄운다 — 데이텀을 고를 때 필요하다. */
  const [picked, setPicked] = useState<CadHole | null>(null);
  /* 편차를 표면에 입힐지. 부품이 회색 덩어리로만 보이면 편차 프로젝트에서
     3D 가 할 일이 없다. */
  const [showHeat, setShowHeat] = useState(true);
  /* 단면 — 판금은 겹쳐진 면이 많아 겉에서만 보면 안쪽을 못 본다. */
  const [clip, setClip] = useState(0);          // 0 이면 끔, 아니면 자르는 위치
  const [clipRatio, setClipRatio] = useState(0);
  /* 측정 — 두 점을 찍으면 거리를 잰다. 금형에서 자주 쓴다. */
  const [measuring, setMeasuring] = useState(false);
  /* 보정시트는 편차 포인트를 전부 적지 않는다 — 손볼 자리만 골라 적는다.
     핵심 포인트 선별이 아직 개발 중이라, 그 전까지는 보정량 크기로 거른다. */
  const [threshold, setThreshold] = useState(0.5);
  /* 콜아웃을 눌러 값을 고친다. 화면 좌표를 들고 있어야 입력칸을 그 자리에
     띄울 수 있다. */
  const [editing, setEditing] = useState<
    { id: string; value: string; x: number; y: number } | null>(null);
  /* 주석 달기 — 켜면 형상을 누른 자리에 메모가 생긴다. */
  const [noting, setNoting] = useState(false);
  const [noteDraft, setNoteDraft] = useState<
    { at: [number, number, number]; text: string; x: number; y: number } | null>(null);
  const [measure, setMeasure] = useState<
    { from: [number, number, number]; to?: [number, number, number] } | null>(null);

  // 씬 안에서 켜고 끌 그룹들은 ref 로 들고 있어야 리렌더 없이 토글된다.
  const holeGroup = useRef<THREE.Group | null>(null);
  const planeGroup = useRef<THREE.Group | null>(null);
  const overlayGroup = useRef<THREE.Group | null>(null);
  const clipRef = useRef<THREE.Plane | null>(null);
  const noteRef = useRef<THREE.Group | null>(null);
  const notingRef = useRef(noting);
  notingRef.current = noting;
  const measureRef = useRef<THREE.Group | null>(null);
  const surfaceRef = useRef<THREE.Mesh | null>(null);
  const viewApi = useRef<{
    frame: (direction: THREE.Vector3) => void;
    snapshot: () => string;
    centre: THREE.Vector3; radius: number;
  } | null>(null);
  const edgeLines = useRef<THREE.LineSegments | null>(null);
  const solidMesh = useRef<THREE.Mesh | null>(null);

  // 쪼개진 원통면 합치기와 굽힘 R 걸러내기는 백엔드가 이미 했다
  // (step_reader._merge_cylinder_faces). 여기 오는 건 닫힌 원통뿐이다.
  const holes = useMemo(() => mesh.holes || [], [mesh.holes]);
  const holeLabel = useMemo(() => {
    const sizes = [...new Set(holes.map((h) => h.diameter.toFixed(2)))];
    if (!sizes.length) return '홀 없음';
    if (sizes.length === 1) return `홀 ${holes.length} · Ø${sizes[0]}`;
    return `홀 ${holes.length} · ${sizes.length}종`;
  }, [holes]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let renderer: THREE.WebGLRenderer;
    try {
      // preserveDrawingBuffer: 화면 저장을 하려면 버퍼가 남아 있어야 한다
      renderer = new THREE.WebGLRenderer({
        antialias: true, alpha: false, preserveDrawingBuffer: true });
    } catch {
      setError('이 브라우저에서 WebGL 을 쓸 수 없습니다.');
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.setClearColor(0x16202a);
    renderer.localClippingEnabled = true;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();

    // ── 형상 ─────────────────────────────────────────────────
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position', new THREE.Float32BufferAttribute(mesh.positions, 3));
    geometry.setIndex(mesh.indices);
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();

    const radius = geometry.boundingSphere?.radius ?? 1;
    const centre = geometry.boundingSphere?.center ?? new THREE.Vector3();

    // ── 편차를 표면 색으로 ───────────────────────────────────
    // 값의 절대 크기가 아니라 컬러바 범위에 맞춰 칠한다. 그래야 스캔
    // 히트맵과 같은 색이 같은 편차를 뜻한다.
    const deviations = overlay?.surfaceDeviation;
    const range = overlay?.deviationRange;
    let painted = false;
    if (showHeat && deviations && deviations.length === mesh.positions.length / 3) {
      const span = Math.max(
        Math.abs(range?.[0] ?? -1), Math.abs(range?.[1] ?? 1), 0.01);
      const colours = new Float32Array(deviations.length * 3);
      for (let i = 0; i < deviations.length; i += 1) {
        const value = deviations[i];
        if (value === null || value === undefined) {
          colours[i * 3] = NO_DATA.r;
          colours[i * 3 + 1] = NO_DATA.g;
          colours[i * 3 + 2] = NO_DATA.b;
          continue;
        }
        const [r, g, b] = rampColor((value / span + 1) / 2);
        colours[i * 3] = r; colours[i * 3 + 1] = g; colours[i * 3 + 2] = b;
      }
      geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3));
      painted = true;
    }

    // 단면 — 화면 기준이 아니라 부품 좌표 기준으로 자른다. 돌려봐도
    // 자른 자리가 그대로 있어야 단면을 읽을 수 있다.
    const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    clipRef.current = clipPlane;

    const surface = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
      color: painted ? 0xffffff : SURFACE,
      vertexColors: painted,
      metalness: painted ? 0.15 : 0.55,
      roughness: painted ? 0.75 : 0.42,
      side: THREE.DoubleSide, flatShading: false,
      clippingPlanes: [clipPlane],
    }));
    scene.add(surface);
    solidMesh.current = surface;
    surfaceRef.current = surface;

    const measureRoot = new THREE.Group();
    scene.add(measureRoot);
    measureRef.current = measureRoot;

    const noteRoot = new THREE.Group();
    scene.add(noteRoot);
    noteRef.current = noteRoot;

    // 삼각망을 눈으로 확인하는 층. 45,224개라 선이 촘촘하다 —
    // 그래서 기본값은 '모서리'(각진 곳만)로 두고 전체 와이어는 선택이다.
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, 24),
      new THREE.LineBasicMaterial({ color: 0x2f4256, transparent: true, opacity: 0.85 }),
    );
    scene.add(edges);
    edgeLines.current = edges;

    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(geometry),
      new THREE.LineBasicMaterial({ color: 0x33566f, transparent: true, opacity: 0.28 }),
    );
    wire.visible = false;
    wire.name = 'wire';
    scene.add(wire);

    // ── 홀 ───────────────────────────────────────────────────
    // 실측 부품은 Ø6mm 홀이 1062mm 짜리 형상에 박혀 있다. 실제 크기대로
    // 링만 그리면 점만 해서 안 보인다. 그래서 두 겹으로 그린다 —
    //   (1) 실제 지름 링: 위치와 크기를 정직하게 보여준다
    //   (2) 축 핀: 홀 축을 따라 부품 크기에 비례한 선을 뚫어 놓는다.
    //       어느 각도에서 봐도 홀이 어디 있는지 바로 찾을 수 있다.
    const holesRoot = new THREE.Group();
    const axisUp = new THREE.Vector3(0, 0, 1);
    const pinLength = radius * 0.16;
    const holeMaterial = new THREE.MeshBasicMaterial({ color: HOLE_TINT });
    for (const hole of holes) {
      const r = Math.max(hole.radius, radius * 0.0015);
      const tube = Math.max(r * 0.18, radius * 0.0022);

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(r, tube, 10, 32), holeMaterial);
      ring.position.set(...hole.center);
      const axis = new THREE.Vector3(...hole.axis).normalize();
      ring.quaternion.setFromUnitVectors(axisUp, axis);
      holesRoot.add(ring);

      const pin = new THREE.Mesh(
        new THREE.CylinderGeometry(tube * 0.75, tube * 0.75, pinLength, 6),
        holeMaterial);
      pin.position.set(...hole.center);
      // CylinderGeometry 는 Y 축을 따라 서 있다
      pin.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
      holesRoot.add(pin);

      // 클릭 판정용. 실제 링은 너무 얇아 못 누른다.
      const target = new THREE.Mesh(
        new THREE.SphereGeometry(Math.max(r * 1.6, radius * 0.006), 8, 6),
        new THREE.MeshBasicMaterial({ visible: false }));
      target.position.set(...hole.center);
      target.userData.hole = hole;
      holesRoot.add(target);
    }
    holesRoot.visible = showHoles;
    scene.add(holesRoot);
    holeGroup.current = holesRoot;

    // ── 평면(데이텀 후보) ────────────────────────────────────
    const planesRoot = new THREE.Group();
    for (const plane of mesh.planes || []) {
      const side = Math.sqrt(Math.max(plane.area, 1));
      const patch = new THREE.Mesh(
        new THREE.PlaneGeometry(side, side),
        new THREE.MeshBasicMaterial({
          color: PLANE_TINT, transparent: true, opacity: 0.22,
          side: THREE.DoubleSide, depthWrite: false,
        }),
      );
      patch.position.set(plane.center[0], plane.center[1], plane.center[2]);
      const normal = new THREE.Vector3(...plane.normal).normalize();
      patch.quaternion.setFromUnitVectors(axisUp, normal);
      planesRoot.add(patch);

      const arrow = new THREE.ArrowHelper(
        normal, new THREE.Vector3(...plane.center),
        Math.max(side * 0.6, radius * 0.05), PLANE_TINT, undefined, undefined);
      planesRoot.add(arrow);
    }
    planesRoot.visible = false;
    scene.add(planesRoot);
    planeGroup.current = planesRoot;

    // ── 스캔에서 옮겨온 것들 (제로라인·보정량) ───────────────
    const overlayRoot = new THREE.Group();
    const labelPicks: THREE.Sprite[] = [];
    if (overlay) {
      // 제로라인 — 표면에 살짝 띄워 z-파이팅을 피한다
      const lift = radius * 0.006;   // 형상에 묻히지 않게 굵게
      for (const line of overlay.zeroLines || []) {
        if (!line.points?.length) continue;
        const pts = line.points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
        const curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.0);
        const tube = new THREE.Mesh(
          new THREE.TubeGeometry(curve, Math.max(pts.length * 4, 32), lift, 6, false),
          new THREE.MeshBasicMaterial({ color: ZERO_TINT }),
        );
        overlayRoot.add(tube);
      }

      // 보정량 — 표면에서 화살표를 세우고 값을 붙인다.
      // 값은 최종 보정시트에서 온다. 시트에 없는 포인트(작업자가 숨긴 것)는
      // 3D 에도 안 나온다 — 두 화면이 항상 같은 것을 보여줘야 한다.
      const scale = radius * 0.05;
      const shown = overlay.points
        .map((p) => ({ point: p, correction: sheetValues?.[p.id] }))
        .filter((entry): entry is { point: typeof entry.point; correction: number } =>
          typeof entry.correction === 'number'
          && Math.abs(entry.correction) >= threshold);
      const maxCorrection = Math.max(
        ...shown.map((e) => Math.abs(e.correction)), 0.5);
      // 화살표는 표면 법선을 따라야 한다. 월드 축으로 세우면 곡면에서
      // 엉뚱한 쪽을 가리킨다 — 판넬은 전체가 곡면이다.
      const normalAt = (spot: THREE.Vector3) => {
        const normals = geometry.getAttribute('normal');
        const positions = geometry.getAttribute('position');
        let best = -1;
        let bestDistance = Infinity;
        // 가까운 정점의 법선을 쓴다. 정확한 면을 찾을 필요까진 없다.
        for (let i = 0; i < positions.count; i += 7) {
          const dx = positions.getX(i) - spot.x;
          const dy = positions.getY(i) - spot.y;
          const dz = positions.getZ(i) - spot.z;
          const distance = dx * dx + dy * dy + dz * dz;
          if (distance < bestDistance) { bestDistance = distance; best = i; }
        }
        if (best < 0) return new THREE.Vector3(0, 0, 1);
        return new THREE.Vector3(
          normals.getX(best), normals.getY(best), normals.getZ(best)).normalize();
      };

      // 시트와 같은 표기 — 보정 지점에 빨간 점, 거기서 빨간 점선을 뽑아
      // 끝에 노란 숫자 박스를 단다. 0 인 자리도 시트에는 적히므로 남긴다.
      //
      // [겹침을 어떻게 푸는가]
      // 앞서는 지시선을 표면 법선으로만 뽑았다. 그런데 판넬은 법선이
      // 대체로 한쪽을 향해서 라벨이 한 곳에 쌓여 값이 안 읽혔다.
      // 시트는 이 문제를 콜아웃을 **부품 바깥 테두리에 둘러** 푼다.
      // 같은 방법을 쓴다 — 스캔이 바라본 평면에서 각도로 정렬해 바깥
      // 링에 고르게 앉히면 순서가 유지되고 서로 겹치지 않는다.
      const markMaterial = new THREE.MeshBasicMaterial({ color: MARK_TINT });
      const leaderMaterial = new THREE.LineDashedMaterial({
        color: MARK_TINT, dashSize: radius * 0.012, gapSize: radius * 0.009,
        depthTest: false,
      });

      const viewAxis = overlay.fit?.axis ?? 2;
      const planeAxes = ([[1, 2], [0, 2], [0, 1]] as const)[viewAxis];
      const toArray = (v: THREE.Vector3) => [v.x, v.y, v.z];
      const flat = (v: THREE.Vector3) => {
        const a = toArray(v);
        return new THREE.Vector2(a[planeAxes[0]], a[planeAxes[1]]);
      };

      const spots = shown.map(({ point, correction }) => {
        const origin = new THREE.Vector3(...point.position);
        return { origin, correction, pointId: point.id, plane: flat(origin) };
      });
      const middle = spots.reduce(
        (sum, s) => sum.add(s.plane), new THREE.Vector2()).divideScalar(
          Math.max(spots.length, 1));
      const spread = Math.max(
        ...spots.map((s) => s.plane.distanceTo(middle)), radius * 0.2);
      // 라벨은 한 평면에 모아 둔다. 깊이가 제각각이면 다른 각도에서
      // 흩어져 보인다.
      const labelDepth = toArray(centre)[viewAxis]
        + (geometry.boundingSphere?.radius ?? radius) * 0.55;

      // [배치] 큰 원에 둘러 놓으니 지시선이 별처럼 퍼져 안 읽혔다.
      // 시트는 콜아웃을 **자기 점 바로 바깥**에 붙이고 겹칠 때만 조금씩
      // 밀어낸다. 같은 방법으로, 점에서 바깥쪽으로 짧게 빼고 겹치면
      // 한 칸씩 더 민다.
      const labelSize = new THREE.Vector2(spread * 0.20, spread * 0.075);
      const taken: THREE.Vector2[] = [];
      const seatFor = (plane: THREE.Vector2) => {
        const away = plane.clone().sub(middle);
        if (away.lengthSq() < 1e-9) away.set(1, 0);
        away.normalize();
        for (let step = 0; step < 14; step += 1) {
          const spot = plane.clone().add(
            away.clone().multiplyScalar(spread * (0.16 + step * 0.085)));
          const clash = taken.some((other) =>
            Math.abs(other.x - spot.x) < labelSize.x
            && Math.abs(other.y - spot.y) < labelSize.y);
          if (!clash) { taken.push(spot); return spot; }
        }
        const fallback = plane.clone().add(away.multiplyScalar(spread * 1.4));
        taken.push(fallback);
        return fallback;
      };

      // 바깥쪽부터 자리를 잡아야 안쪽 라벨이 멀리 밀려나지 않는다
      spots.sort((a, b) => b.plane.distanceTo(middle) - a.plane.distanceTo(middle));

      spots.forEach((spot) => {
        const flatSeat = seatFor(spot.plane);
        const seat = new THREE.Vector3();
        const coords = [0, 0, 0];
        coords[planeAxes[0]] = flatSeat.x;
        coords[planeAxes[1]] = flatSeat.y;
        coords[viewAxis] = labelDepth;
        seat.set(coords[0], coords[1], coords[2]);

        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(radius * 0.006, 10, 8), markMaterial);
        dot.position.copy(spot.origin);
        dot.renderOrder = 8;
        overlayRoot.add(dot);

        const leader = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([spot.origin, seat]),
          leaderMaterial);
        leader.computeLineDistances();     // 점선은 이걸 해야 보인다
        leader.renderOrder = 9;
        overlayRoot.add(leader);

        const label = makeLabel(
          `${spot.correction > 0 ? '+' : ''}${spot.correction.toFixed(1)}`,
          radius * 0.038);
        label.position.copy(seat);
        label.renderOrder = 10;
        label.userData.pointId = spot.pointId;
        label.userData.correction = spot.correction;
        labelPicks.push(label);
        overlayRoot.add(label);
      });
    }
    scene.add(overlayRoot);
    overlayGroup.current = overlayRoot;

    // ── 조명 ─────────────────────────────────────────────────
    scene.add(new THREE.HemisphereLight(0xdfeaf5, 0x1b2732, 1.15));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(1, 1.4, 1).multiplyScalar(radius * 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x9fc4e0, 0.7);
    fill.position.set(-1.2, -0.6, -0.9).multiplyScalar(radius * 3);
    scene.add(fill);

    // ── 카메라 ───────────────────────────────────────────────
    const camera = new THREE.PerspectiveCamera(
      42, mount.clientWidth / mount.clientHeight, radius * 0.01, radius * 60);
    // 스캔이 바라본 방향이 있으면 그쪽에 세운다. 안 그러면 얇은 쪽에서
    // 보게 되어 형상이 선처럼 보인다(실측: 판넬이 한 축으로 155mm 다).
    const start = new THREE.Vector3(radius * 1.5, radius * 1.1, radius * 1.9);
    if (overlay?.fit) {
      const along = [0, 0, 0];
      along[overlay.fit.axis] = overlay.fit.sign >= 0 ? 1 : -1;
      start.set(along[0], along[1], along[2]).multiplyScalar(radius * 2.4);
      // 완전히 정면이면 입체감이 없어 살짝 비껴 세운다
      start.x += radius * 0.12;
      start.y += radius * 0.1;
    }
    camera.position.copy(centre).add(start);

    // ── CATIA 식 마우스 ──────────────────────────────────────
    // OrbitControls 로는 CATIA 를 흉내낼 수 없어 직접 만들었다. 두 가지가
    // 걸렸다 —
    //   (1) CATIA 는 가운데를 먼저 누르고 오른쪽을 **나중에** 더한다.
    //       OrbitControls 는 누르는 순간 역할이 정해져 도중에 못 바꾼다.
    //   (2) 회전 방향이 반대다. CATIA 는 모델이 커서를 따라오는데
    //       OrbitControls 는 카메라가 도는 느낌이라 반대로 움직인다.
    //
    //   가운데 끌기              이동
    //   가운데 + 오른쪽 끌기      회전   (누르는 도중에 더해도 바뀐다)
    //   Ctrl + 가운데 끌기        확대·축소
    //   휠                       확대·축소
    //   왼쪽                     선택 (CATIA 와 같다. 회전 아님)
    const target = centre.clone();
    const spherical = new THREE.Spherical().setFromVector3(
      camera.position.clone().sub(target));
    let mode: 'none' | 'pan' | 'rotate' | 'zoom' = 'none';
    let last = { x: 0, y: 0 };

    const applyCamera = () => {
      spherical.phi = Math.max(0.001, Math.min(Math.PI - 0.001, spherical.phi));
      spherical.radius = Math.max(radius * 0.05,
        Math.min(radius * 40, spherical.radius));
      camera.position.copy(target).add(
        new THREE.Vector3().setFromSpherical(spherical));
      camera.lookAt(target);
    };
    applyCamera();

    const modeFor = (event: PointerEvent | MouseEvent) => {
      const middle = (event.buttons & 4) !== 0;
      if (!middle) return 'none';
      if (event.ctrlKey) return 'zoom';
      return (event.buttons & 2) !== 0 ? 'rotate' : 'pan';
    };

    const onMove = (event: PointerEvent) => {
      // 버튼 조합이 바뀌면 도중에라도 따라간다 — CATIA 는 가운데를 누른
      // 채로 오른쪽을 더해 회전으로 넘어간다.
      const next = modeFor(event);
      if (next !== mode) {
        mode = next as typeof mode;
        last = { x: event.clientX, y: event.clientY };
        return;
      }
      if (mode === 'none') return;
      const dx = event.clientX - last.x;
      const dy = event.clientY - last.y;
      last = { x: event.clientX, y: event.clientY };

      if (mode === 'rotate') {
        // 부호가 CATIA 기준이다 — 오른쪽으로 끌면 모델이 오른쪽으로 돈다.
        spherical.theta += dx * 0.005;
        spherical.phi += dy * 0.005;
      } else if (mode === 'pan') {
        const height = mount.clientHeight || 1;
        const perPixel = 2 * spherical.radius
          * Math.tan((camera.fov * Math.PI / 180) / 2) / height;
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
        target.add(right.multiplyScalar(-dx * perPixel));
        target.add(up.multiplyScalar(dy * perPixel));
      } else {
        spherical.radius *= Math.exp(dy * 0.006);
      }
      applyCamera();
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      spherical.radius *= Math.exp(Math.sign(event.deltaY) * 0.12);
      applyCamera();
    };
    const onButtonDown = (event: PointerEvent) => {
      if (event.button === 1) event.preventDefault();   // 가운데 자동스크롤 막기
      mode = modeFor(event) as typeof mode;
      last = { x: event.clientX, y: event.clientY };
      if (mode !== 'none') renderer.domElement.setPointerCapture(event.pointerId);
    };
    const onButtonUp = (event: PointerEvent) => {
      mode = modeFor(event) as typeof mode;
      if (mode === 'none' && renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
    };
    const blockMenu = (event: Event) => event.preventDefault();

    renderer.domElement.addEventListener('pointerdown', onButtonDown);
    renderer.domElement.addEventListener('pointermove', onMove);
    renderer.domElement.addEventListener('pointerup', onButtonUp);
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false });
    renderer.domElement.addEventListener('contextmenu', blockMenu);

    // 왼쪽 버튼을 끌지 않고 놓았을 때만 선택으로 본다.
    const raycaster = new THREE.Raycaster();
    let downAt: { x: number; y: number } | null = null;
    const onDown = (event: PointerEvent) => {
      if (event.button === 0) downAt = { x: event.clientX, y: event.clientY };
    };
    const onUp = (event: PointerEvent) => {
      if (event.button !== 0 || !downAt) return;
      const moved = Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y);
      downAt = null;
      if (moved > 4) return;
      const rect = renderer.domElement.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1), camera);
      // 콜아웃을 눌렀으면 값 고치기로 들어간다
      const onLabel = raycaster.intersectObjects(labelPicks, false)[0];
      if (onLabel?.object.userData?.pointId) {
        const sprite = onLabel.object as THREE.Sprite;
        const screen = sprite.position.clone().project(camera);
        setEditing({
          id: String(sprite.userData.pointId),
          value: String(sprite.userData.correction ?? 0),
          x: (screen.x * 0.5 + 0.5) * rect.width,
          y: (-screen.y * 0.5 + 0.5) * rect.height,
        });
        return;
      }
      setEditing(null);

      if (notingRef.current) {
        const spot = raycaster.intersectObject(surface, false)[0];
        if (spot) {
          setNoteDraft({
            at: [spot.point.x, spot.point.y, spot.point.z], text: '',
            x: event.clientX - rect.left, y: event.clientY - rect.top,
          });
        }
        return;
      }
      if (measuringRef.current) {
        const spot = raycaster.intersectObject(surface, false)[0];
        if (spot) {
          const point: [number, number, number] =
            [spot.point.x, spot.point.y, spot.point.z];
          setMeasure((current) => (current && !current.to)
            ? { ...current, to: point } : { from: point });
        }
        return;
      }
      const hit = raycaster.intersectObjects(holesRoot.children, false)
        .find((entry) => entry.object.userData?.hole);
      setPicked(hit ? (hit.object.userData.hole as CadHole) : null);
    };
    renderer.domElement.addEventListener('pointerdown', onDown);
    renderer.domElement.addEventListener('pointerup', onUp);

    // 표준 뷰와 전체 맞춤
    const frame = (direction: THREE.Vector3) => {
      target.copy(centre);
      spherical.setFromVector3(direction.clone().normalize()
        .multiplyScalar(radius * 2.6));
      applyCamera();
    };
    const snapshot = () => {
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL('image/png');
    };
    viewApi.current = { frame, snapshot, centre: centre.clone(), radius };

    // ── 루프 ─────────────────────────────────────────────────
    let loop = 0;
    const tick = () => {
      loop = requestAnimationFrame(tick);
      renderer.render(scene, camera);
    };
    tick();

    const resize = new ResizeObserver(() => {
      const w = mount.clientWidth, h = mount.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    resize.observe(mount);

    return () => {
      cancelAnimationFrame(loop);
      resize.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onButtonDown);
      renderer.domElement.removeEventListener('pointermove', onMove);
      renderer.domElement.removeEventListener('pointerup', onButtonUp);
      renderer.domElement.removeEventListener('wheel', onWheel);
      renderer.domElement.removeEventListener('contextmenu', blockMenu);
      renderer.domElement.removeEventListener('pointerdown', onDown);
      renderer.domElement.removeEventListener('pointerup', onUp);
      renderer.dispose();
      scene.traverse((node) => {
        const any = node as THREE.Mesh;
        any.geometry?.dispose?.();
        const material = any.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(material)) material.forEach((m) => m.dispose());
        else material?.dispose?.();
      });
      mount.removeChild(renderer.domElement);
    };
  }, [mesh, holes, showHoles, overlay, sheetValues, showHeat, threshold]);

  // 토글은 씬을 다시 만들지 않고 가시성만 바꾼다.
  useEffect(() => {
    if (holeGroup.current) holeGroup.current.visible = showHoles;
  }, [showHoles]);

  useEffect(() => {
    if (planeGroup.current) planeGroup.current.visible = showPlanes;
  }, [showPlanes]);

  useEffect(() => {
    if (overlayGroup.current) overlayGroup.current.visible = showOverlay;
  }, [showOverlay]);

  // 클릭 처리는 씬을 다시 만들지 않고 최신 상태를 봐야 한다
  const measuringRef = useRef(measuring);
  measuringRef.current = measuring;

  useEffect(() => {
    const plane = clipRef.current;
    const surface = surfaceRef.current;
    if (!plane || !surface) return;
    const material = surface.material as THREE.MeshStandardMaterial;
    material.clippingPlanes = clip ? [plane] : [];
    plane.constant = clip;
    material.needsUpdate = true;
  }, [clip]);

  useEffect(() => {
    const root = noteRef.current;
    if (!root) return;
    root.clear();
    const scale = (viewApi.current?.radius ?? 100);
    for (const note of notes ?? []) {
      const at = new THREE.Vector3(...note.at);
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(scale * 0.007, 10, 8),
        new THREE.MeshBasicMaterial({ color: 0x6fb4e8, depthTest: false }));
      dot.position.copy(at);
      dot.renderOrder = 12;
      root.add(dot);

      const sprite = makeNote(note.text, scale * 0.032);
      sprite.position.copy(at).add(new THREE.Vector3(0, 0, scale * 0.05));
      sprite.renderOrder = 13;
      root.add(sprite);

      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([at, sprite.position]),
        new THREE.LineBasicMaterial({ color: 0x6fb4e8, depthTest: false }));
      line.renderOrder = 12;
      root.add(line);
    }
  }, [notes]);

  useEffect(() => {
    const root = measureRef.current;
    if (!root) return;
    root.clear();
    if (!measure) return;
    const from = new THREE.Vector3(...measure.from);
    const scale = (viewApi.current?.radius ?? 100) * 0.008;
    const mark = (spot: THREE.Vector3) => {
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(scale, 10, 8),
        new THREE.MeshBasicMaterial({ color: 0x35d68a, depthTest: false }));
      dot.position.copy(spot);
      dot.renderOrder = 12;
      root.add(dot);
    };
    mark(from);
    if (measure.to) {
      const to = new THREE.Vector3(...measure.to);
      mark(to);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([from, to]),
        new THREE.LineBasicMaterial({ color: 0x35d68a, depthTest: false }));
      line.renderOrder = 12;
      root.add(line);
    }
  }, [measure]);

  const goToView = (dir: [number, number, number]) => {
    viewApi.current?.frame(new THREE.Vector3(...dir));
  };

  const saveImage = () => {
    const url = viewApi.current?.snapshot();
    if (!url) return;
    const link = document.createElement('a');
    link.href = url;
    link.download = `${mesh.summary.name || 'part'}.png`;
    link.click();
  };

  useEffect(() => {
    const solid = solidMesh.current;
    const edges = edgeLines.current;
    if (!solid || !edges) return;
    const wire = solid.parent?.getObjectByName('wire');
    solid.visible = detail !== 'wire';
    edges.visible = detail === 'edges';
    if (wire) wire.visible = detail === 'wire';
  }, [detail]);

  if (error) return <div className="cad-viewer__error">{error}</div>;

  const sheetCount = overlay?.points
    ?.filter((p) => typeof sheetValues?.[p.id] === 'number').length ?? 0;
  const overlayPoints = overlay?.points?.filter((p) => {
    const value = sheetValues?.[p.id];
    return typeof value === 'number' && Math.abs(value) >= threshold;
  }).length ?? 0;

  return <>
    <div ref={mountRef} className="cad-viewer__stage" />

    <div className="cad-viewer__section">
      <label htmlFor="cad-clip">단면</label>
      <input id="cad-clip" type="range" min={-1} max={1} step={0.01}
        value={clipRatio}
        onChange={(event) => {
          const ratio = Number(event.target.value);
          setClipRatio(ratio);
          const span = viewApi.current?.radius ?? 100;
          setClip(ratio === 0 ? 0 : ratio * span);
        }} />
      <span>{clip ? `${clip.toFixed(0)} mm` : '끔'}</span>
    </div>

    <div className="cad-viewer__views" role="group" aria-label="표준 뷰">
      {VIEWS.map((view) => (
        <button key={view.id} type="button" onClick={() => goToView(view.dir)}>
          {view.label}
        </button>
      ))}
      <button type="button" onClick={saveImage} title="보이는 그대로 PNG 로 저장">
        저장
      </button>
    </div>

    {overlay && showHeat && overlay.deviationRange && (
      <div className="cad-viewer__bar">
        <span>{overlay.deviationRange[1].toFixed(1)}</span>
        <i />
        <span>0</span>
        <span className="cad-viewer__bar-low">
          {overlay.deviationRange[0].toFixed(1)}
        </span>
        <small>편차 mm</small>
      </div>
    )}

    {overlay && showOverlay && overlay.points.length > 0 && (
      <div className="cad-viewer__legend">
        <span><i style={{ background: '#ffef3a' }} />보정량 (mm)</span>
        <span><i style={{ background: '#e01b1b' }} />보정 지점과 지시선</span>
        <span><i style={{ background: '#ff3b30' }} />제로라인</span>
        <small>보정시트와 같은 표기입니다</small>
      </div>
    )}

    <div className="cad-viewer__hud">
      <div className="cad-viewer__seg" role="group" aria-label="표시 방식">
        {([['solid', '표면'], ['edges', '모서리'], ['wire', '삼각망']] as const).map(
          ([value, label]) => (
            <button key={value} type="button"
              className={detail === value ? 'is-on' : ''}
              onClick={() => setDetail(value)}>{label}</button>
          ))}
      </div>
      <button type="button" className={showPlanes ? 'is-on' : ''}
        onClick={() => setShowPlanes((v) => !v)}>
        평면 {mesh.planes?.length ?? 0}
      </button>
      {overlay && <button type="button" className={showOverlay ? 'is-on' : ''}
        onClick={() => setShowOverlay((v) => !v)}>
        제로라인·보정량 {overlayPoints}
      </button>}
      {overlay && showOverlay && <span className="cad-viewer__pick-level">
        <label htmlFor="cad-threshold">보정량</label>
        <input id="cad-threshold" type="range" min={0} max={2} step={0.1}
          value={threshold}
          onChange={(event) => setThreshold(Number(event.target.value))} />
        <b>{threshold.toFixed(1)}mm 이상</b>
        {sheetCount > overlayPoints && (
          <em>{sheetCount - overlayPoints}개 숨김</em>
        )}
      </span>}
      {overlay?.surfaceDeviation?.length ? (
        <button type="button" className={showHeat ? 'is-on' : ''}
          onClick={() => setShowHeat((v) => !v)}>편차 색</button>
      ) : null}
      <button type="button" className={measuring ? 'is-on' : ''}
        onClick={() => { setMeasuring((v) => !v); setMeasure(null); setNoting(false); }}>
        측정
      </button>
      <button type="button" className={noting ? 'is-on' : ''}
        onClick={() => { setNoting((v) => !v); setNoteDraft(null); setMeasuring(false); }}>
        주석 {notes?.length ? notes.length : ''}
      </button>
      <span className="cad-viewer__stat">
        삼각형 {mesh.summary.n_faces.toLocaleString()} · {holeLabel}
        {mesh.counts?.cylinders
          ? ` · 굽힘 R ${mesh.counts.cylinders - holes.length}`
          : ''}
      </span>
      <span className="cad-viewer__stat cad-viewer__hint">
        {noting ? '형상 위를 눌러 메모를 답니다'
          : measuring ? '형상 위 두 곳을 눌러 거리를 잽니다'
          : '가운데 이동 · 가운데+오른쪽 회전 · Ctrl+가운데 확대 · 왼쪽 선택 · 콜아웃 눌러 수정'}
      </span>
    </div>

    {noteDraft && (
      <form className="cad-viewer__note"
        style={{ left: noteDraft.x, top: noteDraft.y }}
        onSubmit={(event) => {
          event.preventDefault();
          const text = noteDraft.text.trim();
          if (text) {
            onNotesChange?.([...(notes ?? []), {
              id: `N-${Date.now().toString(36)}`, at: noteDraft.at, text }]);
          }
          setNoteDraft(null);
        }}>
        <input autoFocus value={noteDraft.text} placeholder="메모"
          onChange={(event) => setNoteDraft((current) =>
            current ? { ...current, text: event.target.value } : current)}
          onKeyDown={(event) => { if (event.key === 'Escape') setNoteDraft(null); }} />
        <button type="submit">달기</button>
        <button type="button" onClick={() => setNoteDraft(null)}>취소</button>
      </form>
    )}

    {noting && notes && notes.length > 0 && (
      <div className="cad-viewer__notes">
        {notes.map((note) => (
          <span key={note.id}>
            {note.text}
            <button type="button" aria-label={`${note.text} 주석 삭제`}
              onClick={() => onNotesChange?.(
                notes.filter((other) => other.id !== note.id))}>×</button>
          </span>
        ))}
      </div>
    )}

    {editing && (
      <form className="cad-viewer__edit"
        style={{ left: editing.x, top: editing.y }}
        onSubmit={(event) => {
          event.preventDefault();
          const parsed = Number(editing.value);
          if (Number.isFinite(parsed)) onCorrectionChange?.(editing.id, parsed);
          setEditing(null);
        }}>
        <label htmlFor="cad-edit">{editing.id}</label>
        <input id="cad-edit" autoFocus type="number" step="0.1"
          value={editing.value}
          onChange={(event) =>
            setEditing((current) =>
              current ? { ...current, value: event.target.value } : current)}
          onKeyDown={(event) => { if (event.key === 'Escape') setEditing(null); }} />
        <span>mm</span>
        <button type="submit">확인</button>
        <button type="button" className="cad-viewer__edit-reset"
          onClick={() => { onCorrectionChange?.(editing.id, null); setEditing(null); }}>
          되돌리기
        </button>
      </form>
    )}

    {measure?.to && (
      <div className="cad-viewer__measure">
        <b>{new THREE.Vector3(...measure.from)
          .distanceTo(new THREE.Vector3(...measure.to)).toFixed(2)} mm</b>
        <span>두 점 사이 직선거리</span>
        <button type="button" onClick={() => setMeasure(null)}>지우기</button>
      </div>
    )}

    {picked && (
      <div className="cad-viewer__pick">
        <b>Ø{picked.diameter.toFixed(2)} mm</b>
        <span>깊이 {picked.height.toFixed(2)} mm</span>
        <span>중심 {picked.center.map((v) => v.toFixed(1)).join(', ')}</span>
        <span>축 {picked.axis.map((v) => v.toFixed(2)).join(', ')}</span>
        <button type="button" onClick={() => setPicked(null)}>닫기</button>
      </div>
    )}

    {overlay && !overlay.fit.reliable && (
      <p className="cad-viewer__warn">
        CAD 겉모양과 스캔이 {Math.round(overlay.fit.iou * 100)}% 만 겹칩니다.
        위치가 어긋날 수 있어 참고용으로만 보세요.
      </p>
    )}
  </>;
}
