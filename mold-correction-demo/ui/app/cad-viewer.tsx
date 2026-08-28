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
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

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
  points: { id: string; position: [number, number, number];
            value: number; correction: number }[];
  coefficient: number;
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

const SURFACE = 0x8fa3b4;
const HOLE_TINT = 0xff8b3d;
const PLANE_TINT = 0x35d68a;
const ZERO_TINT = 0xff3b30;      // 제로라인 — 보정시트의 빨간 선과 맞춘다
const PLUS_TINT = 0xe0483f;      // 살이 많다 (깎아낸다)
const MINUS_TINT = 0x2f7fe0;     // 살이 부족하다 (붙인다)

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
function makeLabel(text: string, color: string, height: number): THREE.Sprite {
  const pad = 8;
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return new THREE.Sprite();
  context.font = '600 44px ui-sans-serif, system-ui, sans-serif';
  const width = Math.ceil(context.measureText(text).width) + pad * 2;
  canvas.width = width;
  canvas.height = 64;

  const ctx = canvas.getContext('2d')!;
  ctx.font = '600 44px ui-sans-serif, system-ui, sans-serif';
  ctx.fillStyle = 'rgba(10,16,22,.82)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, pad, canvas.height / 2 + 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, depthTest: false, transparent: true,
  }));
  sprite.scale.set(height * canvas.width / canvas.height, height, 1);
  return sprite;
}

export function CadViewer({ mesh, showHoles, overlay }: {
  mesh: CadMesh; showHoles: boolean; overlay?: CadOverlay | null;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<CadDetail>('edges');
  const [showPlanes, setShowPlanes] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);

  // 씬 안에서 켜고 끌 그룹들은 ref 로 들고 있어야 리렌더 없이 토글된다.
  const holeGroup = useRef<THREE.Group | null>(null);
  const planeGroup = useRef<THREE.Group | null>(null);
  const overlayGroup = useRef<THREE.Group | null>(null);
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

    const surface = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
      color: SURFACE, metalness: 0.55, roughness: 0.42,
      side: THREE.DoubleSide, flatShading: false,
    }));
    scene.add(surface);
    solidMesh.current = surface;

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
    if (overlay) {
      // 제로라인 — 표면에 살짝 띄워 z-파이팅을 피한다
      const lift = radius * 0.004;
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
      // 길이는 보정량에 비례하되 최소 길이를 둬서 작은 값도 보이게 한다.
      const scale = radius * 0.05;
      const maxCorrection = Math.max(
        ...overlay.points.map((p) => Math.abs(p.correction)), 0.5);
      for (const point of overlay.points) {
        const magnitude = Math.abs(point.correction);
        if (magnitude < 0.05) continue;      // 손댈 필요 없는 자리
        const positive = point.correction > 0;
        const colour = positive ? PLUS_TINT : MINUS_TINT;
        const origin = new THREE.Vector3(...point.position);
        const length = scale * (0.35 + 0.65 * magnitude / maxCorrection);
        const up = new THREE.Vector3(0, 0, positive ? 1 : -1);

        const arrow = new THREE.ArrowHelper(
          up, origin, length, colour, length * 0.34, length * 0.2);
        overlayRoot.add(arrow);

        const label = makeLabel(
          `${point.correction > 0 ? '+' : ''}${point.correction.toFixed(1)}`,
          positive ? '#ffb4ad' : '#a8ccf5', radius * 0.028);
        label.position.copy(origin).add(up.clone().multiplyScalar(length * 1.25));
        overlayRoot.add(label);
      }
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
    camera.position.copy(centre).add(
      new THREE.Vector3(radius * 1.5, radius * 1.1, radius * 1.9));

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(centre);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    // ── CATIA 식 마우스 ──────────────────────────────────────
    //   가운데 버튼 끌기        = 이동  (CATIA 와 같다)
    //   가운데 + 오른쪽 끌기    = 회전  (CATIA 와 같다)
    //   Ctrl + 가운데 끌기      = 확대·축소
    //   왼쪽 끌기               = 회전  (웹에서 흔한 방식이라 같이 둔다)
    //   휠                      = 확대·축소
    // OrbitControls 는 버튼 조합을 모르니, 누르는 순간 buttons 비트를
    // 보고 가운데 버튼의 역할을 바꿔 준다.
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.PAN,
    };
    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 1) return;
      const rightHeld = (event.buttons & 2) !== 0;
      controls.mouseButtons.MIDDLE = (rightHeld || event.ctrlKey)
        ? (event.ctrlKey ? THREE.MOUSE.DOLLY : THREE.MOUSE.ROTATE)
        : THREE.MOUSE.PAN;
    };
    const blockMenu = (event: Event) => event.preventDefault();
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('contextmenu', blockMenu);

    // 표준 뷰와 전체 맞춤을 밖에서 부를 수 있게 걸어 둔다
    const frame = (direction: THREE.Vector3) => {
      const distance = radius * 2.6;
      camera.position.copy(centre).add(direction.clone().normalize()
        .multiplyScalar(distance));
      controls.target.copy(centre);
      controls.update();
    };
    const snapshot = () => {
      // preserveDrawingBuffer 를 켜지 않았으므로 저장 직전에 한 번 더 그린다
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL('image/png');
    };
    viewApi.current = { frame, snapshot, centre: centre.clone(), radius };

    // ── 루프 ─────────────────────────────────────────────────
    let loop = 0;
    const tick = () => {
      loop = requestAnimationFrame(tick);
      controls.update();
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
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('contextmenu', blockMenu);
      controls.dispose();
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
  }, [mesh, holes, showHoles, overlay]);

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

  const overlayPoints = overlay?.points?.length ?? 0;

  return <>
    <div ref={mountRef} className="cad-viewer__stage" />

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

    {overlay && showOverlay && overlay.points.length > 0 && (
      <div className="cad-viewer__legend">
        <span><i style={{ background: '#e0483f' }} />살이 많다 · 깎아낸다</span>
        <span><i style={{ background: '#2f7fe0' }} />살이 부족하다 · 붙인다</span>
        <span><i style={{ background: '#ff3b30' }} />제로라인</span>
        <small>화살표 길이는 보정량에 비례합니다</small>
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
      <span className="cad-viewer__stat">
        삼각형 {mesh.summary.n_faces.toLocaleString()} · {holeLabel}
        {mesh.counts?.cylinders
          ? ` · 굽힘 R ${mesh.counts.cylinders - holes.length}`
          : ''}
      </span>
      <span className="cad-viewer__stat cad-viewer__hint">
        가운데 버튼 이동 · 가운데+오른쪽 회전 · 휠 확대
      </span>
    </div>

    {overlay && !overlay.fit.reliable && (
      <p className="cad-viewer__warn">
        CAD 겉모양과 스캔이 {Math.round(overlay.fit.iou * 100)}% 만 겹칩니다.
        위치가 어긋날 수 있어 참고용으로만 보세요.
      </p>
    )}
  </>;
}
