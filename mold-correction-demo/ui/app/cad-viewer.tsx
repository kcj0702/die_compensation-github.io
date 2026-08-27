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

export type CadDetail = 'solid' | 'edges' | 'wire';

const SURFACE = 0x8fa3b4;
const HOLE_TINT = 0xff8b3d;
const PLANE_TINT = 0x35d68a;

export function CadViewer({ mesh, showHoles }: { mesh: CadMesh; showHoles: boolean }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<CadDetail>('edges');
  const [showPlanes, setShowPlanes] = useState(false);

  // 씬 안에서 켜고 끌 그룹들은 ref 로 들고 있어야 리렌더 없이 토글된다.
  const holeGroup = useRef<THREE.Group | null>(null);
  const planeGroup = useRef<THREE.Group | null>(null);
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
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
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

    // ── 루프 ─────────────────────────────────────────────────
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
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
      cancelAnimationFrame(frame);
      resize.disconnect();
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
  }, [mesh, holes, showHoles]);

  // 토글은 씬을 다시 만들지 않고 가시성만 바꾼다.
  useEffect(() => {
    if (holeGroup.current) holeGroup.current.visible = showHoles;
  }, [showHoles]);

  useEffect(() => {
    if (planeGroup.current) planeGroup.current.visible = showPlanes;
  }, [showPlanes]);

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

  return <>
    <div ref={mountRef} className="cad-viewer__stage" />
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
      <span className="cad-viewer__stat">
        삼각형 {mesh.summary.n_faces.toLocaleString()} · {holeLabel}
        {mesh.counts?.cylinders
          ? ` · 굽힘 R ${mesh.counts.cylinders - holes.length}`
          : ''}
      </span>
    </div>
  </>;
}
