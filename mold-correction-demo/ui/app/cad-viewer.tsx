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
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

export type CadHole = {
  kind: string; radius: number; diameter: number;
  center: [number, number, number]; axis: [number, number, number];
  height: number; area: number;
  wrap?: number; faces?: number;
};

/** /api/cad-sections 가 돌려주는, 시트 단면 표기로 계산한 제로라인.
 *  다른 제로라인과 달리 **추정이 없다** — 시트가 준 좌표로 CAD 를 자른 것이다. */
export type CadSection = {
  label: string; axis: number; value: number;
  polylines: [number, number, number][][]; point_count: number;
};

/** /api/cad-overlay 가 돌려주는, CAD 표면 위로 옮겨진 스캔 결과. */
export type CadOverlay = {
  fit: {
    axis: number; sign: number; flip_u: boolean; flip_v: boolean;
    mm_per_px: number; iou: number; reliable: boolean;
    /* 스캔 위의 점이 형상에 얹히는 비율. 겹침 넓이보다 이게 실제 기준이다 —
       껍질 겹침은 후하고(실측 64XX2 96.9%) 실루엣은 박하다(42.2%). */
    hit_rate?: number; detail_iou?: number;
  };
  zeroLines: {
    line_id: number | null;
    points: [number, number, number][];
    /* 표면에 얹힌 구간(실선)과 빈 공간을 지나는 구간(점선).
       구멍 위를 실선으로 그리면 없는 자리에 선이 있는 것처럼 보인다. */
    runs?: [number, number, number][][];
    gaps?: [number, number, number][][];
  }[];
  /* 정점마다 0 = 아님 · 1 = 제로라인(띠) · 2 = 제로 영역.
     선을 공간에 띄우는 대신 **표면을 칠한다** — 곡면을 그대로 따라간다. */
  zeroSurface?: number[];
  /* 제로 영역의 네모 테두리를 표면 위로 옮긴 것. 칠하기만으로는
     경계가 삼각망을 따라 들쭉날쭉해 네모로 안 보인다. */
  zeroAreas?: { runs: [number, number, number][][];
                gaps: [number, number, number][][] }[];
  zeroKind?: string;
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

/** 공정 구역 — 시트의 분홍 영역과 "① : 하형 용접" 표기에 해당한다.
 *  어느 금형(상형/하형)을 어떻게(용접/가공/심고음) 손볼지 적는다. */
export type CadRegion = {
  id: string;
  /* 붓으로 칠한 자국들. 끌면 여러 개가 쌓여 작업자가 원하는 모양이 된다.
     예전에는 at + radius 하나뿐이라 클릭한 자리 둘레의 **동그라미밖에**
     못 만들었다 — "구역이 랜덤으로 잡힌다" 는 게 그 얘기였다. */
  stamps?: { at: [number, number, number]; radius: number }[];
  /* 네모·동그라미로 한 번에 잡은 구역. ADC 보정시트가 영역을 이렇게
     표기하므로 붓질보다 이쪽이 시트와 그대로 맞는다.
     끌기 시작할 때의 화면 가로(u)·세로(v) 방향을 함께 적어 둔다 —
     돌려봐도 같은 자리를 덮으려면 방향이 부품에 붙어 있어야 한다. */
  shape?: {
    kind: 'rect' | 'circle';
    center: [number, number, number];
    u: [number, number, number];
    v: [number, number, number];
    hu: number; hv: number;
  };
  die: '상형' | '하형';
  work: '용접' | '가공' | '심고음';
  /* 예전 형식. 저장해 둔 작업을 계속 읽으려고 남겨 둔다. */
  at?: [number, number, number];
  radius?: number;
};

/** 예전 형식(at+radius)과 새 형식(stamps)을 한 가지로 본다. */
export function stampsOf(region: CadRegion) {
  if (region.stamps?.length) return region.stamps;
  if (region.at && region.radius) return [{ at: region.at, radius: region.radius }];
  return [];
}

export const DIE_CHOICES: CadRegion['die'][] = ['상형', '하형'];
export const WORK_CHOICES: CadRegion['work'][] = ['용접', '가공', '심고음'];
/** 시트가 쓰는 원문자. 구역이 열 개를 넘을 일은 없다. */
/** 보정 후 형상 — 원본과 견줘 보려고 만든다. */
export type CadMorph = {
  positions: number[];
  shift: number[];
  stats: { moved: number; max_shift: number; mean_shift: number; reach_mm: number };
  points: number;
};

export const CIRCLED = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩'];

const SURFACE = 0x8fa3b4;
const HOLE_TINT = 0xff8b3d;
const PLANE_TINT = 0x35d68a;
const ZERO_TINT = 0xff3b30;      // 제로라인 (스캔에서 추정한 것)
const SECTION_TINT = 0x21c07a;  // 시트 단면으로 계산한 제로라인 (오차 없음)
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

function makeZoneLabel(text: string, height: number): THREE.Sprite {
  /* 공정 표기 — 시트의 "① : 하형 용접" 과 같게 분홍 글씨로 낸다. */
  const pad = 16;
  const measure = document.createElement('canvas').getContext('2d');
  if (!measure) return new THREE.Sprite();
  const font = '700 44px ui-sans-serif, system-ui, sans-serif';
  measure.font = font;
  const canvas = document.createElement('canvas');
  canvas.width = Math.ceil(measure.measureText(text).width) + pad * 2;
  canvas.height = 66;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgba(255,246,250,.96)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 4;
  ctx.strokeStyle = '#d61f77';
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
  ctx.font = font;
  ctx.fillStyle = '#b31563';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2 + 1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, depthTest: false, transparent: true,
  }));
  sprite.scale.set(height * canvas.width / canvas.height, height, 1);
  return sprite;
}

export function CadViewer({ active = true, sections, mesh, showHoles, overlay, sheetValues,
                           onCorrectionChange, notes, onNotesChange,
                           onCapture, regions, onRegionsChange,
                           morph, morphMode = 'off' }: {
  /* 감춰져 있으면 그리지 않는다. 3D 화면을 안 보고 있어도
     컴포넌트는 살아 있어서(읽어 둔 CAD 를 지키려고) 그냥 두면
     보이지도 않는 화면을 계속 GPU 로 그린다. */
  active?: boolean;
  /* 시트 단면 표기로 계산한 제로라인. 오차 없는 값이라 굵게 그린다. */
  sections?: CadSection[] | null;
  mesh: CadMesh; showHoles: boolean; overlay?: CadOverlay | null;
  /* 포인트 아이디 -> 최종 보정량(mm). 시트에서 숨긴 포인트는 빠져 있다. */
  sheetValues?: Record<string, number> | null;
  /* 3D 에서 고친 값도 시트와 같은 저장소로 간다 — 양쪽이 어긋나면 안 된다. */
  onCorrectionChange?: (pointId: string, value: number | null) => void;
  notes?: CadNote[];
  onNotesChange?: (notes: CadNote[]) => void;
  /* 지금 보이는 화면을 PNG 로 넘겨준다 — 보정시트에 넣을 그림이다. */
  onCapture?: (dataUrl: string) => void;
  regions?: CadRegion[];
  onRegionsChange?: (regions: CadRegion[]) => void;
  /* 보정 후 형상. 있으면 원본과 겹쳐 보거나 갈아 끼울 수 있다. */
  morph?: CadMorph | null;
  morphMode?: 'off' | 'after' | 'both';
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail] = useState<CadDetail>('solid');   // 표면만 쓴다
  const [showPlanes, setShowPlanes] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  /* 홀을 누르면 지름과 좌표를 띄운다 — 데이텀을 고를 때 필요하다. */
  const [picked, setPicked] = useState<CadHole | null>(null);
  /* 편차를 표면에 입힐지. 부품이 회색 덩어리로만 보이면 편차 프로젝트에서
     3D 가 할 일이 없다. */
  const [showHeat] = useState(false);   // 편차 색은 화면에서 뺐다
  /* 단면 — 판금은 겹쳐진 면이 많아 겉에서만 보면 안쪽을 못 본다. */
  /* 단면 — 판금은 겹쳐진 면이 많아 겉에서만 보면 안쪽을 못 본다.
     자르는 위치는 **부품의 실제 Z 범위** 안에서 고른다. 100% 면 끔. */
  const [depth, setDepth] = useState<{ min: number; max: number } | null>(null);
  const [clipPct, setClipPct] = useState(100);
  const clip = depth && clipPct < 100
    ? depth.min + (depth.max - depth.min) * (clipPct / 100) : null;
  /* 씬을 다시 만들 때 지금 값을 알아야 한다. 상태로 읽으면 그 시점의
     값이 아니라 이펙트가 묶인 시점의 값이 온다. */
  const clipRefValue = useRef<number | null>(clip);
  clipRefValue.current = clip;
  /* 측정 — 두 점을 찍으면 거리를 잰다. 금형에서 자주 쓴다. */
  /* 화면 돌리기(roll). 길쭉한 부품이 세로로 서서 나오면 화면을 반도
     못 쓴다 — 실측 64XX1 은 220 x 1492 x 555mm 라 세로로 선다.
     CATIA 식 조작(가운데+오른쪽 끌기)으로는 화면 안에서 눕히는 회전을
     못 하고, 마우스 가운데 버튼이 없는 사람도 있다. 버튼으로 준다. */
  const rollRef = useRef(0);
  const [roll, setRoll] = useState(0);
  const [measuring, setMeasuring] = useState(false);
  /* 보정시트는 편차 포인트를 전부 적지 않는다 — 손볼 자리만 골라 적는다.
     핵심 포인트 선별이 아직 개발 중이라, 그 전까지는 보정량 크기로 거른다. */
  /* 보정량 범위. 아래(이상)만 있었는데 위(이하)도 필요하다 —
     "0.5 이상" 만으로는 큰 값에 묻혀 작은 자리를 못 본다. */
  const [threshold, setThreshold] = useState(0.5);
  const [ceiling, setCeiling] = useState(9);
  /* 콜아웃을 눌러 값을 고친다. 화면 좌표를 들고 있어야 입력칸을 그 자리에
     띄울 수 있다. */
  const [editing, setEditing] = useState<
    { id: string; value: string; x: number; y: number } | null>(null);
  /* 주석 달기 — 켜면 형상을 누른 자리에 메모가 생긴다. */
  const [noting, setNoting] = useState(false);
  const [noteDraft, setNoteDraft] = useState<
    { at: [number, number, number]; text: string; x: number; y: number } | null>(null);
  /* 공정 구역 — 시트의 분홍 영역. 찍은 자리 둘레를 칠하고 번호를 붙인다. */
  /* 보정 후 형상을 몇 배로 부풀려 볼지. 1 이면 실제 그대로(=안 보인다).
     메인 이펙트의 의존성에 들어가므로 반드시 그보다 **앞에서** 선언해야
     한다 — 뒤에 두면 의존성 배열이 평가될 때 아직 초기화 전이다. */
  const [exaggeration, setExaggeration] = useState(30);
  /* 홀을 지름으로 거른다. 실측 67XX6 은 홀이 152 개라 전부 켜 두면
     핀이 빽빽해 데이텀으로 쓸 큰 홀을 못 고른다. */
  const [holeFloor, setHoleFloor] = useState(0);
  const [zoning, setZoning] = useState(false);
  /* 구역을 어떻게 잡을지. 시트처럼 네모를 기본으로 둔다 — 붓질은
     "정확한 구역 표시가 안 된다" 는 지적이 있었다. */
  const [zoneTool, setZoneTool] = useState<'rect' | 'circle' | 'brush'>('rect');
  const [zoneRadius, setZoneRadius] = useState(0.12);   // 부품 크기 대비
  const [measure, setMeasure] = useState<
    { from: [number, number, number]; to?: [number, number, number] } | null>(null);

  // 씬 안에서 켜고 끌 그룹들은 ref 로 들고 있어야 리렌더 없이 토글된다.
  const holeGroup = useRef<THREE.Group | null>(null);
  const planeGroup = useRef<THREE.Group | null>(null);
  const overlayGroup = useRef<THREE.Group | null>(null);
  const clipRef = useRef<THREE.Plane | null>(null);
  const noteRef = useRef<THREE.Group | null>(null);
  /* 단면에서 보정 전후 윤곽을 견주는 층. */
  const sliceRef = useRef<THREE.Group | null>(null);
  const [sliceGap, setSliceGap] = useState<number | null>(null);
  const regionRef = useRef<THREE.Group | null>(null);
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);
  const zoningRef = useRef(zoning);
  zoningRef.current = zoning;
  const zoneRadiusRef = useRef(zoneRadius);
  zoneRadiusRef.current = zoneRadius;
  const zoneToolRef = useRef(zoneTool);
  zoneToolRef.current = zoneTool;
  const notingRef = useRef(noting);
  notingRef.current = noting;
  /* 구역 목록을 ref 로도 들고 있는다. 클릭 핸들러는 WebGL 이펙트 안에
     있는데 그 이펙트의 의존성에 regions 가 없다. 그래서 핸들러가 처음
     만들어질 때의 빈 목록을 계속 붙들고 있었고, 구역을 새로 찍을 때마다
     **직전 것이 교체**됐다. 구역이 추가가 안 되던 이유다. */
  const regionsRef = useRef(regions);
  regionsRef.current = regions;
  /* 지금 칠하고 있는 구역. 고르면 거기에 덧칠하고, 없으면 새로 만든다. */
  const [activeZone, setActiveZone] = useState<string | null>(null);
  const activeZoneRef = useRef(activeZone);
  activeZoneRef.current = activeZone;
  const measureRef = useRef<THREE.Group | null>(null);
  const surfaceRef = useRef<THREE.Mesh | null>(null);
  const viewApi = useRef<{
    frame: (direction: THREE.Vector3) => void;
    snapshot: () => string;
    /* 카메라를 지금 값으로 다시 세운다(화면 돌리기 등). */
    refresh: () => void;
    centre: THREE.Vector3; radius: number;
    /* 단면 슬라이더가 쓸 실제 Z 범위. 구 반지름으로 갈음하면
       원점이 부품 밖에 있는 CAD 에서 최대로 밀어도 잘린다. */
    zMin: number; zMax: number;
  } | null>(null);
  const edgeLines = useRef<THREE.LineSegments | null>(null);
  const solidMesh = useRef<THREE.Mesh | null>(null);

  // 쪼개진 원통면 합치기, 굽힘 R 걸러내기, 더 큰 홀 안의 턱 빼기는
  // 백엔드가 이미 했다(step_reader). 여기 오는 건 관통 홀뿐이다.
  const holes = useMemo(() => mesh.holes || [], [mesh.holes]);
  /* 큰 것부터 몇 종류나 되는지 — 데이텀은 대개 큰 홀이라 고르기 쉽게. */
  const holeSizes = useMemo(() => [...new Set(
    holes.map((h) => Math.round(h.diameter * 10) / 10))]
    .sort((a, b) => b - a), [holes]);
  const holeLabel = useMemo(() => {
    const sizes = [...new Set(holes.map((h) => h.diameter.toFixed(2)))];
    if (!sizes.length) return '홀 없음';
    if (sizes.length === 1) return `홀 ${holes.length} · Ø${sizes[0]}`;
    return `홀 ${holes.length} · ${sizes.length}종`;
  }, [holes]);

  /* 보이는지 여부는 ref 로 넘긴다. 의존성에 넣으면 탭을 옮길 때마다
     장면을 통째로 다시 만들어 카메라 위치가 초기화된다. */
  const activeRef = useRef(active);
  activeRef.current = active;
  /* 표시 모드(표면/모서리/삼각망)를 ref 로도 들고 있는다. 장면은 오버레이가
     바뀔 때마다 통째로 다시 만들어지는데, 그때 새 객체는 전부 기본 표시
     상태다. detail 이펙트는 detail 이 바뀔 때만 도니 다시 적용되지 않아
     **모드가 저절로 풀렸다** — "시간 지나면 이렇게 됨" 이 이것이다. */
  const detailRef = useRef<'solid' | 'edges' | 'wire'>('solid');

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
    // 금속은 밝은 곳을 반사해야 형태가 읽힌다. 톤매핑 없이 두면
    // 반사 하이라이트가 흰색으로 다 타버린다.
    // 그림자는 끈다. team-15 뷰어에서 가져왔다가 되돌렸다 — 그쪽은
    // 로봇·용접건처럼 **서로 떨어진 물체 여러 개**라 그림자가 공간을
    // 설명해 준다. 우리는 얇은 판금 껍데기 **하나**뿐이라 자기그림자밖에
    // 안 생기고, 두께가 없다시피 해서 앞뒤 면이 서로를 가려 얼룩진다
    // (shadow acne). 형상이 깨져 보이던 원인이다.
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.8;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();

    // ── 환경 반사 ────────────────────────────────────────────
    // 이게 없어서 형상이 시커멓게 나왔다. MeshStandardMaterial 은 PBR 이라
    // metalness 를 올리면 확산광(diffuse)이 그만큼 사라지고 대신 **주변을
    // 반사**해서 형태를 보여준다. 그런데 환경맵이 없으면 반사할 게 없어
    // 방향광의 좁은 하이라이트만 남고 나머지는 검게 깔린다. 판넬처럼
    // 완만한 곡면은 하이라이트가 거의 안 걸려서 통째로 실루엣이 된다.
    //
    // RoomEnvironment 는 three 가 들고 있는 절차적 실내 장면이라 파일을
    // 받아올 필요가 없다 — 로컬 node_modules 안에서 끝난다.
    const pmrem = new THREE.PMREMGenerator(renderer);
    const environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = environment;
    pmrem.dispose();

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
    // 정합이 나쁘면(겹침 75% 미만) 칠하지 않는다. 51% 정합으로 편차색을
    // 입혔더니 엉뚱한 자리에 빨간 조각이 흩어져 "형상이 깨졌다" 는 인상을
    // 줬다 — 틀린 그림을 그럴듯하게 보여주는 것이 제일 나쁘다.
    if (showHeat && overlay?.fit.reliable && deviations
        && deviations.length === mesh.positions.length / 3) {
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

    // 보정 후 형상 — 원본 위에 겹치거나 원본을 대신한다.
    if (morph && morphMode !== 'off'
        && morph.positions.length === mesh.positions.length) {
      const after = new THREE.BufferGeometry();
      // 변형을 **과장해서** 보여준다.
      //
      // 실제 보정량은 부품 크기의 0.13~0.16% 다 —
      //     64XX2  1492mm 에 2.0mm    화면에서 0.90px
      //     67XX6  2043mm 에 3.0mm    화면에서 0.99px
      //     71XX2  1230mm 에 2.0mm    화면에서 1.09px
      // 전부 1픽셀 안팎이라 눈으로는 원리적으로 구분할 수 없다. 겹쳐
      // 놓아도 색만 다르고 형상은 똑같아 보인다. 해석 소프트웨어가
      // 변형을 수십 배 부풀려 보여주는 이유가 이것이다.
      //
      // 부풀린 형상은 **보는 용도**다. STL 로 내보내는 값은 손대지 않는다.
      const puffed = new Float32Array(morph.positions.length);
      for (let i = 0; i < morph.positions.length; i += 1) {
        puffed[i] = mesh.positions[i]
          + (morph.positions[i] - mesh.positions[i]) * exaggeration;
      }
      after.setAttribute('position',
        new THREE.Float32BufferAttribute(puffed, 3));
      after.setIndex(mesh.indices);
      after.computeVertexNormals();
      // 얼마나 밀렸는지 색으로 — 살을 붙인 쪽이 분홍, 깎은 쪽이 하늘색
      const tint = new Float32Array(morph.shift.length * 3);
      const peak = Math.max(morph.stats.max_shift, 0.01);
      for (let i = 0; i < morph.shift.length; i += 1) {
        const ratio = Math.min(Math.abs(morph.shift[i]) / peak, 1);
        const warm = morph.shift[i] > 0;
        tint[i * 3] = warm ? 0.55 + ratio * 0.45 : 0.55 - ratio * 0.3;
        tint[i * 3 + 1] = 0.62 - ratio * 0.25;
        tint[i * 3 + 2] = warm ? 0.72 - ratio * 0.3 : 0.72 + ratio * 0.28;
      }
      after.setAttribute('color', new THREE.BufferAttribute(tint, 3));
      const skin = new THREE.Mesh(after, new THREE.MeshStandardMaterial({
        vertexColors: true, metalness: 0.05, roughness: 0.75,
        envMapIntensity: 0.35, side: THREE.DoubleSide,
        transparent: morphMode === 'both', opacity: morphMode === 'both' ? 0.85 : 1,
      }));
      skin.renderOrder = 2;
      scene.add(skin);
    }

    // metalness 0.55 로 두었더니 환경맵이 없던 시절 형상이 새까맣게 나왔다.
    // 환경맵을 넣은 지금도 판금은 완전한 거울이 아니므로 0.25 정도가
    // 실제 강판에 가깝고 곡면 음영이 훨씬 잘 읽힌다.
    const surface = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
      color: painted ? 0xffffff : SURFACE,
      vertexColors: painted,
      metalness: painted ? 0.05 : 0.15,
      roughness: painted ? 0.85 : 0.62,
      // 환경맵을 세게 주면 판넬이 하얗게 번진다. 형태는 주광이 만든다.
      envMapIntensity: painted ? 0.25 : 0.45,
      side: THREE.DoubleSide, flatShading: false,
      // 자르지 않을 때는 평면을 **달지 않는다.**
      //
      // 예전에는 늘 달아 놓고 [clip] 이펙트가 떼도록 했다. 그런데 그
      // 이펙트는 clip 값이 바뀔 때만 돈다. 씬을 다시 만들면(오버레이·
      // 보정량 범위·과장 변경 등) 새 표면이 평면을 단 채로 나오고,
      // 그때 상수는 0 이라 부품이 z=0 에서 반쪽만 보인다. 단면을 한 번
      // 건드리기 전까지 그대로다 — "형상이 한번씩 짤린다" 가 이것이다.
      clippingPlanes: clipRefValue.current === null ? [] : [clipPlane],
    }));
    clipPlane.constant = clipRefValue.current ?? 0;
    // 겹쳐 볼 때는 원본을 반투명 뼈대로 남긴다
    if (morph && morphMode === 'after') surface.visible = false;
    else if (morph && morphMode === 'both') {
      const material = surface.material as THREE.MeshStandardMaterial;
      material.transparent = true;
      material.opacity = 0.28;
    }
    scene.add(surface);
    solidMesh.current = surface;
    surfaceRef.current = surface;

    const measureRoot = new THREE.Group();
    scene.add(measureRoot);
    measureRef.current = measureRoot;

    const noteRoot = new THREE.Group();
    scene.add(noteRoot);
    noteRef.current = noteRoot;

    const sliceRoot = new THREE.Group();
    sliceRoot.renderOrder = 18;
    scene.add(sliceRoot);
    sliceRef.current = sliceRoot;

    const regionRoot = new THREE.Group();
    scene.add(regionRoot);
    regionRef.current = regionRoot;
    geometryRef.current = geometry;

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

    // 장면을 새로 만들었으니 지금 고른 표시 모드를 곧바로 입힌다
    const shownAs = detailRef.current;
    surface.visible = shownAs !== 'wire';
    edges.visible = shownAs === 'edges';
    wire.visible = shownAs === 'wire';

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
      ring.userData.diameter = hole.diameter;
      ring.position.set(...hole.center);
      const axis = new THREE.Vector3(...hole.axis).normalize();
      ring.quaternion.setFromUnitVectors(axisUp, axis);
      holesRoot.add(ring);

      const pin = new THREE.Mesh(
        new THREE.CylinderGeometry(tube * 0.75, tube * 0.75, pinLength, 6),
        holeMaterial);
      pin.userData.diameter = hole.diameter;
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
      target.userData.diameter = hole.diameter;
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
    if (overlay?.fit.reliable) {
      // 제로라인.
      //
      // 선은 **선으로** 그린다. 표면 삼각형을 칠해 봤더니 리브와 구멍이
      // 많은 면에서 조각조각 갈라져 "물감 칠한 느낌" 이 났다. 대신
      // 서버가 선 위를 촘촘히(4px 간격) 쏴서 표면에 얹어 주므로, 그
      // 점들을 이으면 곡면을 그대로 따라간다.
      const lift = radius * 0.004;
      // 영역으로 답한 부품은 영역만 그린다 — 우리 자체 검출선을 같이
      // 그리면 부품을 가로질러 밖으로 뻗어 나간다.
      const zeroLines = (overlay.zeroKind === 'areas'
        && (overlay.zeroAreas?.length ?? 0)) ? [] : (overlay.zeroLines || []);
      for (const line of zeroLines) {
        // 빈 공간을 지나는 구간은 점선으로. 부품이 없는 자리라 실선으로
        // 그리면 거짓이 된다 — 받은 파이프라인은 구멍을 지나갈 수 있게
        // 돼 있어서 링 부품(선루프)에서 실제로 빈 데를 가로지른다.
        for (const gap of line.gaps ?? []) {
          if (gap.length < 2) continue;
          const dashed = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(
              gap.map(([x, y, z]) => new THREE.Vector3(x, y, z))),
            new THREE.LineDashedMaterial({
              color: ZERO_TINT, dashSize: radius * 0.02,
              gapSize: radius * 0.014, transparent: true, opacity: 0.75,
            }));
          dashed.computeLineDistances();   // 이걸 해야 점선이 보인다
          dashed.renderOrder = 9;
          overlayRoot.add(dashed);
        }

        // 표면에 얹힌 구간만 실선(관)으로 그린다
        const runs = line.runs?.length
          ? line.runs : (line.points?.length ? [line.points] : []);
        for (const run of runs) {
        const pts = run.map(([x, y, z]) => new THREE.Vector3(x, y, z));
        if (pts.length < 2) continue;
        const curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.0);
        // depthTest 를 끄면 형상에 가려야 할 뒷면 부분까지 앞에 그려져
        // **면 위에 얹은 스티커**처럼 보인다. 깊이 검사는 켜 두고
        // polygonOffset 으로 살짝만 띄워 z-파이팅만 피한다 — 그래야
        // 굴곡을 따라 파묻히고 돌아 나오는 게 보인다.
        // 흰 테두리를 깔고 그 위에 색선을 얹는다.
        //
        // 밝은 회색 판금 위에 얇은 빨간 선만 그리면 잘 안 읽힌다.
        // 보정시트도 같은 방법을 쓴다(흰 선 위에 주황 선).
        const casing = new THREE.Mesh(
          new THREE.TubeGeometry(
            curve, Math.max(pts.length, 32), lift * 2.1, 8, false),
          new THREE.MeshBasicMaterial({
            color: 0xffffff,
            polygonOffset: true, polygonOffsetFactor: -3,
          }));
        casing.renderOrder = 8;
        overlayRoot.add(casing);

        const tube = new THREE.Mesh(
          new THREE.TubeGeometry(
            curve, Math.max(pts.length, 32), lift * 1.25, 8, false),
          new THREE.MeshBasicMaterial({
            color: ZERO_TINT,
            polygonOffset: true, polygonOffsetFactor: -4,
          }));
        tube.renderOrder = 9;
        overlayRoot.add(tube);
        }
      }

      // 제로 **영역**(67XX6)은 테두리로만 그린다.
      //
      // 예전에는 표면을 칠했다. 칠하기는 정점 단위라, 세 꼭짓점이 모두
      // 영역 안에 든 삼각형만 남는다 — 리브와 구멍이 많은 면에서
      // 조각조각 갈라져 **네모로 만들어 놓고도 물감 칠한 것처럼**
      // 보였다. 세 번을 고쳐도 그대로였던 게 이 칠하기 때문이다.
      //
      // 테두리는 서버가 네모의 네 변을 촘촘히 쏴서 표면에 얹어 준다.
      // 곡면을 타면서도 경계가 반듯하다.
      if (overlay.zeroKind === 'areas' && (overlay.zeroAreas?.length ?? 0)) {
        const seat = new THREE.Vector3();
        let seen = 0;
        for (const area of overlay.zeroAreas ?? []) {
          for (const run of area.runs ?? []) {
            for (const [x, y, z] of run) {
              seat.add(new THREE.Vector3(x, y, z));
              seen += 1;
            }
          }
        }
        if (seen) {
          seat.divideScalar(seen);
          const tag = makeZoneLabel('제로라인 (영역)', radius * 0.04);
          tag.position.copy(seat).add(new THREE.Vector3(0, 0, radius * 0.06));
          tag.renderOrder = 15;
          overlayRoot.add(tag);
        }
      }

      // 영역 테두리 — 네모의 네 변을 표면에 얹은 것. 칠한 면은 경계가
      // 삼각형을 따라 들쭉날쭉하므로, 그 위에 반듯한 테두리를 덧그려야
      // 어디까지가 그 영역인지 읽힌다. 시트도 영역을 네모로 표기한다.
      for (const area of overlay.zeroAreas ?? []) {
        for (const run of area.runs ?? []) {
          const pts = run.map(([x, y, z]) => new THREE.Vector3(x, y, z));
          if (pts.length < 2) continue;
          const curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.0);
          const casing = new THREE.Mesh(
            new THREE.TubeGeometry(
              curve, Math.max(pts.length, 32), lift * 2.0, 8, false),
            new THREE.MeshBasicMaterial({
              color: 0xffffff,
              polygonOffset: true, polygonOffsetFactor: -5,
            }));
          casing.renderOrder = 9;
          overlayRoot.add(casing);

          const edge = new THREE.Mesh(
            new THREE.TubeGeometry(
              curve, Math.max(pts.length, 32), lift * 1.2, 8, false),
            new THREE.MeshBasicMaterial({
              color: ZERO_TINT,
              polygonOffset: true, polygonOffsetFactor: -6,
            }));
          edge.renderOrder = 10;
          overlayRoot.add(edge);
        }
        // 테두리가 부품 밖(구멍·개구부)을 지나는 구간은 점선이다 —
        // 제로라인과 같은 규칙으로 사실대로 보인다.
        for (const gap of area.gaps ?? []) {
          if (gap.length < 2) continue;
          const dashed = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(
              gap.map(([x, y, z]) => new THREE.Vector3(x, y, z))),
            new THREE.LineDashedMaterial({
              color: ZERO_TINT, dashSize: radius * 0.02,
              gapSize: radius * 0.014, transparent: true, opacity: 0.75,
            }));
          dashed.computeLineDistances();
          dashed.renderOrder = 10;
          overlayRoot.add(dashed);
        }
      }

      // 보정량 — 표면에서 화살표를 세우고 값을 붙인다.
      // 값은 최종 보정시트에서 온다. 시트에 없는 포인트(작업자가 숨긴 것)는
      // 3D 에도 안 나온다 — 두 화면이 항상 같은 것을 보여줘야 한다.
      const scale = radius * 0.05;
      const shown = overlay.points
        .map((p) => ({ point: p, correction: sheetValues?.[p.id] }))
        .filter((entry): entry is { point: typeof entry.point; correction: number } =>
          typeof entry.correction === 'number'
          && Math.abs(entry.correction) >= threshold
          && Math.abs(entry.correction) <= ceiling);
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
        + (geometry.boundingSphere?.radius ?? radius) * 0.18;

      // [배치] 큰 원에 둘러 놓으니 지시선이 별처럼 퍼져 안 읽혔다.
      // 시트는 콜아웃을 **자기 점 바로 바깥**에 붙이고 겹칠 때만 조금씩
      // 밀어낸다. 같은 방법으로, 점에서 바깥쪽으로 짧게 빼고 겹치면
      // 한 칸씩 더 민다.
      // 라벨 자리는 **글자 크기** 기준으로 잡는다. 예전에는 부품 반지름
      // (spread)에 비례해 0.16 배씩 밀어냈는데, 그러면 라벨이 형상 한참
      // 바깥에 놓이고 지시선만 길어진다. 시트는 콜아웃을 자기 점 바로
      // 옆에 붙이고, 겹칠 때만 조금씩 비킨다.
      const tagHeight = radius * 0.038;              // makeLabel 과 같은 크기
      const labelSize = new THREE.Vector2(tagHeight * 2.6, tagHeight * 1.25);
      const taken: THREE.Vector2[] = [];
      const seatFor = (plane: THREE.Vector2) => {
        const away = plane.clone().sub(middle);
        if (away.lengthSq() < 1e-9) away.set(1, 0);
        away.normalize();
        for (let step = 0; step < 16; step += 1) {
          const spot = plane.clone().add(
            away.clone().multiplyScalar(labelSize.x * (0.7 + step * 0.55)));
          const clash = taken.some((other) =>
            Math.abs(other.x - spot.x) < labelSize.x
            && Math.abs(other.y - spot.y) < labelSize.y);
          if (!clash) { taken.push(spot); return spot; }
        }
        const fallback = plane.clone().add(
          away.multiplyScalar(labelSize.x * 9));
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

    // ── 시트 단면으로 계산한 제로라인 ────────────────────────
    // 색을 읽거나 실루엣을 맞춘 게 아니라 시트가 준 좌표로 CAD 를 자른
    // 것이라 오차가 없다. 그래서 추정한 제로라인과 색을 구분해 그린다.
    if (sections?.length) {
      const sectionRoot = new THREE.Group();
      const tint = new THREE.LineBasicMaterial({
        color: SECTION_TINT, depthTest: false });
      for (const section of sections) {
        for (const poly of section.polylines) {
          if (poly.length < 2) continue;
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(
              poly.map(([x, y, z]) => new THREE.Vector3(x, y, z))),
            tint);
          line.renderOrder = 11;
          sectionRoot.add(line);
        }
      }
      scene.add(sectionRoot);
    }

    // ── 조명 ─────────────────────────────────────────────────
    // 환경맵을 넣기 전에는 이 세 개가 장면을 통째로 밝히고 있었다
    // (반구 1.15 · 주광 1.5 · 보조 0.7). 환경맵이 그 일을 대신하게 됐는데
    // 값을 그대로 두는 바람에 빛이 두 번 더해져 판넬이 **하얗게 날아갔다**.
    // 어두워서 안 보이던 게 이번엔 밝아서 안 보였다.
    //
    // 환경맵은 고루 퍼진 빛이라 형태를 못 만든다. 형태는 주광 하나가
    // 만든다 — 그래서 반구와 보조는 색만 얹는 정도로 낮추고 주광을 남긴다.
    // team-15 뷰어(frontend/threejs/viewer.js)의 조명 구성을 가져왔다.
    // 그쪽이 깔끔해 보이는 건 밝기가 아니라 **그림자와 림 라이트**다 —
    // 그림자가 면과 면을 가르고, 뒤에서 치는 림이 윤곽을 세워 준다.
    // 밝기 자체는 우리 화면(어두운 배경 + 환경맵)에 맞게 낮췄다.
    scene.add(new THREE.HemisphereLight(0xdbeafe, 0x1e293b, 0.3));
    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(1, 1.4, 1).multiplyScalar(radius * 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x93c5fd, 0.3);
    fill.position.set(-1.2, -0.6, -0.9).multiplyScalar(radius * 3);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xf8fafc, 0.4);
    rim.position.set(-0.4, 0.3, -1.4).multiplyScalar(radius * 3);
    scene.add(rim);

    // ── 카메라 ───────────────────────────────────────────────
    // 감춰진 채로 만들어지면 mount 크기가 0 이라 0/0 = NaN 이 되고
    // 카메라 위치가 통째로 NaN 이 된다. 크기를 얻을 때까지는 임시 비율을
    // 쓰고, ResizeObserver 가 보이는 순간 제대로 맞춘다.
    const camera = new THREE.PerspectiveCamera(
      42, (mount.clientWidth / mount.clientHeight) || 16 / 9,
      radius * 0.01, radius * 60);
    // 스캔이 바라본 방향이 있으면 그쪽에 세운다. 안 그러면 얇은 쪽에서
    // 보게 되어 형상이 선처럼 보인다(실측: 판넬이 한 축으로 155mm 다).
    // 바운딩 **구**가 아니라 **상자**로 맞춘다.
    //
    // 예전에는 방향에 상관없이 구 반지름 x2.6 에 카메라를 세웠다. 부품이
    // 길쭉하면 이게 방향마다 크게 어긋난다 — 실측 64XX1(220 x 1492.5 x
    // 555.5mm, 구 반지름 804mm, 뷰어 2.42:1)에서 부품이 화면 세로를
    // 차지하는 비율을 재보면:
    //
    //     방향   예전 거리   차지     새 거리   차지
    //     등각     2090mm    102%     2212mm    96%   <- 예전엔 잘렸다
    //     정면     2090mm     54%     1499mm    96%
    //     우측     2090mm     98%     2132mm    96%
    //     평면     2090mm    107%     2300mm    96%   <- 예전엔 잘렸다
    //
    // 정면은 절반만 쓰고 있었고, 등각·평면은 되레 부품 모서리가 화면
    // 밖으로 잘려 나가고 있었다. 상자 꼭짓점으로 맞추면 어느 방향에서든
    // 96% 로 일정하다.
    geometry.computeBoundingBox();
    const box = geometry.boundingBox ?? new THREE.Box3();
    const corners: THREE.Vector3[] = [];
    for (const x of [box.min.x, box.max.x])
      for (const y of [box.min.y, box.max.y])
        for (const z of [box.min.z, box.max.z])
          corners.push(new THREE.Vector3(x, y, z));

    const fitDistance = (direction: THREE.Vector3) => {
      // three 의 lookAt 과 같은 축을 써야 화면 크기가 맞는다:
      //   z = normalize(eye - target),  x = up X z,  y = z X x
      const back = direction.clone().normalize();
      const worldUp = new THREE.Vector3(0, 1, 0);
      // 시선이 up 과 나란하면 축이 무너진다(정면 뷰가 정확히 그렇다)
      if (Math.abs(back.dot(worldUp)) > 0.999) worldUp.set(0, 0, 1);
      const right = new THREE.Vector3().crossVectors(worldUp, back).normalize();
      const up = new THREE.Vector3().crossVectors(back, right).normalize();

      const vFov = (camera.fov * Math.PI) / 180;
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
      // 꼭짓점마다 "이 점이 화면에 들어오려면 얼마나 물러나야 하나" 를
      // 따로 구해 최댓값을 쓴다. 원근이라 앞으로 튀어나온 점일수록 크게
      // 보이므로 그 점의 깊이를 그 점에서만 더해야 한다. 가장 큰 반경과
      // 가장 큰 깊이를 따로 구해 합치면(그렇게 짜봤다) 필요 이상으로
      // 물러나 부품이 작아진다.
      const margin = 1.04;
      let need = 0;
      for (const corner of corners) {
        const v = corner.clone().sub(centre);
        const depth = v.dot(back);
        need = Math.max(need,
          (Math.abs(v.dot(up)) * margin) / Math.tan(vFov / 2) + depth,
          (Math.abs(v.dot(right)) * margin) / Math.tan(hFov / 2) + depth);
      }
      return need;
    };

    const startDir = new THREE.Vector3(0.62, 0.46, 0.79).normalize();
    if (overlay?.fit) {
      const along = [0, 0, 0];
      along[overlay.fit.axis] = overlay.fit.sign >= 0 ? 1 : -1;
      startDir.set(along[0], along[1], along[2]).normalize();
      // 완전히 정면이면 입체감이 없어 살짝 비껴 세운다
      startDir.x += 0.12;
      startDir.y += 0.1;
      startDir.normalize();
    }
    camera.position.copy(centre).add(
      startDir.clone().multiplyScalar(fitDistance(startDir)));

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
      // 시선 축을 중심으로 위쪽 방향을 돌린다 — 화면 안에서만 도는
      // 회전이라 부품을 눕혀 볼 수 있다.
      const look = camera.position.clone().sub(target).normalize();
      camera.up.set(0, 1, 0).applyAxisAngle(look, rollRef.current);
      if (Math.abs(camera.up.dot(look)) > 0.999) camera.up.set(0, 0, 1);
      camera.lookAt(target);

      // 근평면을 **지금 거리에 맞춰** 다시 잡는다.
      //
      // 예전에는 radius*0.01 로 못 박아 뒀는데, 확대 하한이 radius*0.05
      // 라 카메라가 부품 안까지 들어간다. 그러면 근평면이 표면을 베어
      // 형상이 뭉텅뭉텅 사라진다 — "한번씩 짤려서 보이는" 게 이것이다.
      // 멀리 있을 때는 근평면을 밀어야 깊이 정밀도도 산다.
      const away = camera.position.distanceTo(centre);
      camera.near = Math.max(radius * 0.001, (away - radius) * 0.5);
      camera.far = away + radius * 4;
      camera.updateProjectionMatrix();
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
        // 형상 위면 그 자리에, **빈 공간이면 부품 중심을 지나는 평면
        // 위**에 찍는다. 예전에는 형상에 맞아야만 찍혀서 여백에 메모를
        // 달 수가 없었다 — 시트는 여백에 지시문을 적는데 3D 는 못 했다.
        const spot = raycaster.intersectObject(surface, false)[0];
        let at: [number, number, number];
        if (spot) {
          at = [spot.point.x, spot.point.y, spot.point.z];
        } else {
          const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
            camera.getWorldDirection(new THREE.Vector3()), centre);
          const hit = new THREE.Vector3();
          if (!raycaster.ray.intersectPlane(plane, hit)) return;
          at = [hit.x, hit.y, hit.z];
        }
        setNoteDraft({
          at, text: '',
          x: event.clientX - rect.left, y: event.clientY - rect.top,
        });
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
    // ── 공정 구역 붓 ─────────────────────────────────────────
    // 누른 채 끌면 지나간 자리마다 자국이 찍힌다. 클릭 한 번으로 끝내면
    // 클릭 지점 둘레의 동그라미밖에 안 나와서, 작업자가 원하는 모양을
    // 만들 수 없었다.
    let painting: string | null = null;
    const stampAt = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1), camera);
      const spot = raycaster.intersectObject(surface, false)[0];
      if (!spot) return;

      const brush = zoneRadiusRef.current * radius;
      const at: [number, number, number] =
        [spot.point.x, spot.point.y, spot.point.z];
      const current = regionsRef.current ?? [];
      const target = painting
        ?? activeZoneRef.current
        ?? null;
      const existing = current.find((r) => r.id === target);

      if (existing) {
        const stamps = stampsOf(existing);
        const last = stamps[stamps.length - 1];
        // 너무 촘촘하면 자국을 더 찍지 않는다 — 그리는 결과는 같은데
        // 개수만 늘어 덮는 면을 다시 계산할 때 느려진다.
        if (last) {
          const dx = last.at[0] - at[0], dy = last.at[1] - at[1],
                dz = last.at[2] - at[2];
          if (Math.hypot(dx, dy, dz) < brush * 0.35) return;
        }
        painting = existing.id;
        onRegionsChange?.(current.map((r) => r.id === existing.id
          ? { ...r, stamps: [...stamps, { at, radius: brush }] } : r));
        return;
      }

      const id = `Z-${Date.now().toString(36)}`;
      painting = id;
      setActiveZone(id);
      onRegionsChange?.([...current, {
        id, stamps: [{ at, radius: brush }], die: '하형', work: '용접',
      }]);
    };
    // ── 네모·동그라미 구역 ───────────────────────────────────
    // 시트가 영역을 네모/동그라미로 표기하므로 같은 방법을 준다.
    // 끌기 시작점과 끝점을 **그때의 화면 가로·세로 방향**으로 재서
    // 부품 좌표에 박아 둔다. 그래야 돌려봐도 같은 자리를 덮는다.
    const preview = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xff5fa8, depthTest: false }));
    preview.visible = false;
    preview.renderOrder = 20;
    scene.add(preview);

    let dragFrom: THREE.Vector3 | null = null;
    let dragU = new THREE.Vector3();
    let dragV = new THREE.Vector3();

    const surfaceHit = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1), camera);
      return raycaster.intersectObject(surface, false)[0]?.point ?? null;
    };

    /** 두 점으로 만든 네모/동그라미의 테두리 점들. */
    const outline = (kind: 'rect' | 'circle', middle: THREE.Vector3,
                     u: THREE.Vector3, v: THREE.Vector3,
                     hu: number, hv: number) => {
      const at = (du: number, dv: number) => middle.clone()
        .add(u.clone().multiplyScalar(du)).add(v.clone().multiplyScalar(dv));
      if (kind === 'rect') {
        return [at(-hu, -hv), at(hu, -hv), at(hu, hv), at(-hu, hv), at(-hu, -hv)];
      }
      const ring: THREE.Vector3[] = [];
      for (let i = 0; i <= 48; i += 1) {
        const t = (i / 48) * Math.PI * 2;
        ring.push(at(Math.cos(t) * hu, Math.sin(t) * hv));
      }
      return ring;
    };

    const onPaintDown = (event: PointerEvent) => {
      if (!zoningRef.current || event.button !== 0) return;
      event.preventDefault();
      if (zoneToolRef.current === 'brush') {
        painting = activeZoneRef.current;
        stampAt(event);
        return;
      }
      const spot = surfaceHit(event);
      if (!spot) return;
      dragFrom = spot;
      // 끌기 시작 순간의 화면 가로·세로 축을 그대로 쓴다
      dragU.setFromMatrixColumn(camera.matrixWorld, 0).normalize();
      dragV.setFromMatrixColumn(camera.matrixWorld, 1).normalize();
    };

    const onPaintMove = (event: PointerEvent) => {
      if (!zoningRef.current || (event.buttons & 1) === 0) return;
      if (zoneToolRef.current === 'brush') {
        if (painting) stampAt(event);
        return;
      }
      if (!dragFrom) return;
      const spot = surfaceHit(event);
      if (!spot) return;
      const gap = spot.clone().sub(dragFrom);
      const hu = Math.abs(gap.dot(dragU)) / 2;
      const hv = Math.abs(gap.dot(dragV)) / 2;
      const middle = dragFrom.clone()
        .add(dragU.clone().multiplyScalar(gap.dot(dragU) / 2))
        .add(dragV.clone().multiplyScalar(gap.dot(dragV) / 2));
      preview.geometry.dispose();
      preview.geometry = new THREE.BufferGeometry().setFromPoints(
        outline(zoneToolRef.current, middle, dragU, dragV, hu, hv));
      preview.visible = true;
    };

    const onPaintUp = (event: PointerEvent) => {
      painting = null;
      if (!dragFrom || zoneToolRef.current === 'brush') { dragFrom = null; return; }
      preview.visible = false;
      const spot = surfaceHit(event);
      const start = dragFrom;
      dragFrom = null;
      if (!spot) return;
      const gap = spot.clone().sub(start);
      const hu = Math.abs(gap.dot(dragU)) / 2;
      const hv = Math.abs(gap.dot(dragV)) / 2;
      // 손이 떨려 생기는 점만 한 구역은 버린다
      if (hu < radius * 0.004 || hv < radius * 0.004) return;
      const middle = start.clone()
        .add(dragU.clone().multiplyScalar(gap.dot(dragU) / 2))
        .add(dragV.clone().multiplyScalar(gap.dot(dragV) / 2));
      const triple = (v: THREE.Vector3): [number, number, number] =>
        [v.x, v.y, v.z];
      const current = regionsRef.current ?? [];
      const id = `Z-${Date.now().toString(36)}`;
      setActiveZone(id);
      onRegionsChange?.([...current, {
        id, die: '하형', work: '용접',
        shape: {
          kind: zoneToolRef.current, center: triple(middle),
          u: triple(dragU), v: triple(dragV), hu, hv,
        },
      }]);
    };

    renderer.domElement.addEventListener('pointerdown', onPaintDown);
    renderer.domElement.addEventListener('pointermove', onPaintMove);
    window.addEventListener('pointerup', onPaintUp);

    renderer.domElement.addEventListener('pointerdown', onDown);
    renderer.domElement.addEventListener('pointerup', onUp);

    // 표준 뷰와 전체 맞춤
    const frame = (direction: THREE.Vector3) => {
      target.copy(centre);
      const dir = direction.clone().normalize();
      spherical.setFromVector3(dir.multiplyScalar(fitDistance(dir)));
      applyCamera();
    };
    const snapshot = () => {
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL('image/png');
    };
    viewApi.current = { frame, snapshot, refresh: applyCamera,
                        centre: centre.clone(), radius,
                        zMin: box.min.z, zMax: box.max.z };
    setDepth({ min: box.min.z, max: box.max.z });

    // ── 루프 ─────────────────────────────────────────────────
    let loop = 0;
    const tick = () => {
      loop = requestAnimationFrame(tick);
      if (activeRef.current) renderer.render(scene, camera);
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
      renderer.domElement.removeEventListener('pointerdown', onPaintDown);
      renderer.domElement.removeEventListener('pointermove', onPaintMove);
      window.removeEventListener('pointerup', onPaintUp);
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
  }, [mesh, holes, showHoles, overlay, sheetValues, showHeat, threshold,
      morph, morphMode, sections, exaggeration, ceiling]);

  // 토글은 씬을 다시 만들지 않고 가시성만 바꾼다.
  useEffect(() => {
    if (holeGroup.current) holeGroup.current.visible = showHoles;
  }, [showHoles]);

  /* 씬을 다시 만들면 CAD 를 다시 올리는 셈이라, 문턱은 이미 만들어 둔
     표시물을 켜고 끄는 것으로 처리한다. */
  useEffect(() => {
    const root = holeGroup.current;
    if (!root) return;
    for (const child of root.children) {
      const size = child.userData?.diameter;
      child.visible = typeof size !== 'number' || size >= holeFloor;
    }
  }, [holeFloor, holes]);

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
    material.clippingPlanes = clip === null ? [] : [plane];
    plane.constant = clip ?? 0;
    material.needsUpdate = true;
  }, [clip]);

  /* 단면에서 보정 **전후 윤곽**을 견준다.
   *
   * 금형 기술자가 실제로 보는 그림이다. 겉모양만 겹쳐 놓으면 보정량이
   * 부품 크기의 0.13~0.16% 라 1픽셀 안팎이고(위 과장 주석 참고), 색만
   * 달라 보인다. 자른 자리의 **선 두 개**를 나란히 놓으면 어디가 얼마나
   * 밀렸는지 바로 읽힌다.
   *
   * 자를 때마다 삼각형을 한 번 훑는다(40만 개 기준 수십 ms). 매 프레임이
   * 아니라 자르는 자리가 바뀔 때만 도므로 화면이 끊기지 않는다. */
  useEffect(() => {
    const root = sliceRef.current;
    if (!root) return;
    root.clear();
    setSliceGap(null);
    if (clip === null || !morph || morphMode === 'off'
        || morph.positions.length !== mesh.positions.length) return;

    const cut = (points: ArrayLike<number>) => {
      const found: number[] = [];
      const index = mesh.indices;
      for (let i = 0; i < index.length; i += 3) {
        const a = index[i] * 3, b = index[i + 1] * 3, c = index[i + 2] * 3;
        const zs = [points[a + 2], points[b + 2], points[c + 2]];
        const above = zs.map((z) => z > clip);
        if (above[0] === above[1] && above[1] === above[2]) continue;
        // 평면을 걸친 삼각형 — 두 변이 잘린다
        const corners = [a, b, c];
        const hit: number[] = [];
        for (let k = 0; k < 3; k += 1) {
          const m = (k + 1) % 3;
          if (above[k] === above[m]) continue;
          const t = (clip - zs[k]) / (zs[m] - zs[k]);
          hit.push(
            points[corners[k]] + (points[corners[m]] - points[corners[k]]) * t,
            points[corners[k] + 1]
              + (points[corners[m] + 1] - points[corners[k] + 1]) * t,
            clip);
        }
        if (hit.length === 6) found.push(...hit);
      }
      return found;
    };

    const puffed = new Float32Array(morph.positions.length);
    for (let i = 0; i < morph.positions.length; i += 1) {
      puffed[i] = mesh.positions[i]
        + (morph.positions[i] - mesh.positions[i]) * exaggeration;
    }

    const before = cut(mesh.positions);
    const after = cut(puffed);
    if (!before.length && !after.length) return;

    const draw = (points: number[], colour: number, width: number) => {
      if (!points.length) return;
      const line = new THREE.LineSegments(
        new THREE.BufferGeometry().setAttribute(
          'position', new THREE.Float32BufferAttribute(points, 3)),
        new THREE.LineBasicMaterial({
          color: colour, depthTest: false, linewidth: width,
        }));
      line.renderOrder = 19;
      root.add(line);
    };
    draw(before, 0xdfe8f0, 1);        // 보정 전 — 옅은 회색
    draw(after, 0xff8a2b, 2);         // 보정 후 — 주황

    // 이 단면에서 가장 많이 밀린 양(실제 값 — 과장 전)
    let worst = 0;
    for (let i = 0; i < mesh.indices.length; i += 1) {
      const v = mesh.indices[i] * 3;
      if (Math.abs(mesh.positions[v + 2] - clip) > (viewApi.current?.radius ?? 1) * 0.004) continue;
      worst = Math.max(worst, Math.abs(morph.shift[mesh.indices[i]] ?? 0));
    }
    setSliceGap(worst);
  }, [clip, morph, morphMode, exaggeration, mesh]);

  // 공정 구역 — 찍은 자리 둘레의 면만 뽑아 분홍으로 덮는다.
  // 시트가 영역을 분홍으로 칠하고 번호를 붙이는 것과 같은 표기다.
  useEffect(() => {
    const root = regionRef.current;
    const geometry = geometryRef.current;
    if (!root || !geometry) return;
    root.clear();
    const scale = viewApi.current?.radius ?? 100;
    const position = geometry.getAttribute('position');
    const index = geometry.getIndex();
    if (!position || !index) return;

    (regions ?? []).forEach((region, order) => {
      // 구역은 두 가지다 — 시트처럼 네모/동그라미로 한 번에 잡은 것과,
      // 붓으로 칠한 자국들. 둘 다 "이 점이 구역 안인가" 하나로 줄인다.
      const stamps = stampsOf(region);
      const shape = region.shape;
      if (!shape && !stamps.length) return;

      const hull = new THREE.Vector3();
      let reach = 0;
      let inside: (p: THREE.Vector3) => boolean;

      if (shape) {
        const u = new THREE.Vector3(...shape.u);
        const v = new THREE.Vector3(...shape.v);
        const middle = new THREE.Vector3(...shape.center);
        // 판금은 앞뒤 껍질이 겹쳐 있다. 두께 방향으로 막지 않으면
        // 뒷면까지 같이 칠해진다 — 구역 크기의 1/4 만 본다.
        const deep = Math.max(shape.hu, shape.hv) * 0.25;
        const normal = new THREE.Vector3().crossVectors(u, v).normalize();
        hull.copy(middle);
        reach = Math.hypot(shape.hu, shape.hv);
        inside = (p) => {
          const d = p.clone().sub(middle);
          if (Math.abs(d.dot(normal)) > deep) return false;
          const du = d.dot(u), dv = d.dot(v);
          if (shape.kind === 'rect') {
            return Math.abs(du) <= shape.hu && Math.abs(dv) <= shape.hv;
          }
          const nu = du / (shape.hu || 1), nv = dv / (shape.hv || 1);
          return nu * nu + nv * nv <= 1;
        };
      } else {
        const centres = stamps.map((s) => new THREE.Vector3(...s.at));
        const limits = stamps.map((s) => s.radius * s.radius);
        for (const c of centres) hull.add(c);
        hull.divideScalar(centres.length);
        for (let k = 0; k < centres.length; k += 1) {
          reach = Math.max(reach, hull.distanceTo(centres[k]) + stamps[k].radius);
        }
        inside = (p) => centres.some(
          (c, k) => p.distanceToSquared(c) <= limits[k]);
      }

      // 구역을 감싸는 공. 이 밖의 삼각형은 하나씩 재볼 것도 없이
      // 건너뛴다 — 삼각형이 11만 개라 이게 없으면 느리다.
      const reachSq = reach * reach;
      const keep: number[] = [];
      const a = new THREE.Vector3();
      for (let i = 0; i < index.count; i += 3) {
        const i0 = index.getX(i);
        a.set(position.getX(i0), position.getY(i0), position.getZ(i0));
        if (a.distanceToSquared(hull) > reachSq) continue;
        if (inside(a)) keep.push(i0, index.getX(i + 1), index.getX(i + 2));
      }
      if (keep.length) {
        const patch = geometry.clone();
        patch.setIndex(keep);
        const skin = new THREE.Mesh(patch, new THREE.MeshBasicMaterial({
          color: 0xff5fa8, transparent: true, opacity: 0.42,
          side: THREE.DoubleSide, depthWrite: false,
        }));
        skin.renderOrder = 6;
        root.add(skin);
      }

      const tag = makeZoneLabel(
        `${CIRCLED[order] ?? order + 1} ${region.die} ${region.work}`,
        scale * 0.042);
      // 이름표는 칠한 자리 한가운데 위에 띄운다
      tag.position.copy(hull).add(new THREE.Vector3(0, 0, reach * 1.1));
      tag.renderOrder = 14;
      root.add(tag);
    });
  }, [regions]);

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

  /* 전체화면. 브라우저 기본 기능이라 따로 만들 게 없다 — 나갈 때는
     Esc 다. 화면 크기가 바뀌면 ResizeObserver 가 카메라를 다시 맞춘다. */
  const [full, setFull] = useState(false);
  useEffect(() => {
    const onChange = () => setFull(document.fullscreenElement != null);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);
  const toggleFull = () => {
    const box = mountRef.current?.parentElement;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void box?.requestFullscreen?.();
  };

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

  useEffect(() => { detailRef.current = detail; }, [detail]);

  if (error) return <div className="cad-viewer__error">{error}</div>;

  const sheetCount = overlay?.points
    ?.filter((p) => typeof sheetValues?.[p.id] === 'number').length ?? 0;
  const overlayPoints = overlay?.points?.filter((p) => {
    const value = sheetValues?.[p.id];
    return typeof value === 'number'
      && Math.abs(value) >= threshold && Math.abs(value) <= ceiling;
  }).length ?? 0;

  return <>
    <div ref={mountRef} className="cad-viewer__stage" />

    {/* 아래쪽 조작부는 한 덩어리로 쌓는다. 예전에는 단면 슬라이더를
        bottom:52px 로 못 박아 뒀는데, 아래 버튼 줄이 두 줄로 접히면
        그 위로 겹쳐 올라와 둘 다 누르기 어려웠다. */}
    <div className="cad-viewer__controls">
    {morph && morphMode !== 'off' && (
      <div className="cad-viewer__morph-key">
        <span><i style={{ background: '#ff6fa8' }} />살 붙임 (+)</span>
        <span><i style={{ background: '#5fb4e8' }} />깎음 (−)</span>
        <b>최대 {morph.stats.max_shift.toFixed(2)}mm</b>
        {clip === null
          ? <em>단면을 자르면 보정 전후 윤곽을 나란히 볼 수 있습니다</em>
          : <>
              <span><i style={{ background: '#dfe8f0' }} />보정 전</span>
              <span><i style={{ background: '#ff8a2b' }} />보정 후</span>
              {sliceGap !== null && <b>이 단면 최대 {sliceGap.toFixed(2)}mm</b>}
            </>}
      </div>
    )}
    {morph && morphMode !== 'off' && (
      <div className="cad-viewer__section cad-viewer__puff">
        <label htmlFor="cad-puff">변형 과장</label>
        <input id="cad-puff" type="range" min={1} max={100} step={1}
          value={exaggeration}
          onChange={(event) => setExaggeration(Number(event.target.value))} />
        <span>{exaggeration}배</span>
        <em>실제 보정량은 부품 크기의 0.1%대라 1배로는 안 보입니다 ·
          내보내는 STL 은 실제 값입니다</em>
      </div>
    )}
    <div className="cad-viewer__section">
      <label htmlFor="cad-clip">단면</label>
      <input id="cad-clip" type="range" min={0} max={100} step={0.5}
        value={clipPct}
        onChange={(event) => setClipPct(Number(event.target.value))} />
      <span>{clip === null ? '끔' : `${clip.toFixed(0)} mm`}</span>
      {clipPct < 100 && (
        <button type="button" onClick={() => setClipPct(100)}>끄기</button>
      )}
    </div>

    <div className="cad-viewer__views" role="group" aria-label="표준 뷰">
      {VIEWS.map((view) => (
        <button key={view.id} type="button" onClick={() => goToView(view.dir)}>
          {view.label}
        </button>
      ))}
      <button type="button" title="화면을 90도 돌립니다 (길쭉한 부품 눕히기)"
        onClick={() => {
          const next = (roll + 90) % 360;
          setRoll(next);
          rollRef.current = (next * Math.PI) / 180;
          viewApi.current?.refresh();
        }}>
        {roll ? `${roll}°` : '눕히기'}
      </button>
      <button type="button" onClick={toggleFull}
        title="3D 화면을 전체화면으로 봅니다 (Esc 로 나감)">
        {full ? '축소' : '확대'}
      </button>
      <button type="button" onClick={saveImage} title="보이는 그대로 PNG 로 저장">
        저장
      </button>
      {onCapture && <button type="button" title="이 화면을 보정시트에 담습니다"
        onClick={() => {
          const url = viewApi.current?.snapshot();
          if (url) onCapture(url);
        }}>시트에 담기</button>}
    </div>

    {/* 우측 보조 표시들. 예전엔 컬러바(top:52)와 범례(bottom:12)가 각자
        absolute 라 화면이 낮으면 서로 겹치고 밖으로 삐져나왔다. 한 열로
        쌓아 겹칠 수 없게 한다. */}
    <div className="cad-viewer__side">
    {overlay && showHeat && overlay.fit.reliable && overlay.deviationRange && (
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

    {overlay && showOverlay && overlay.fit.reliable && overlay.points.length > 0 && (
      <div className="cad-viewer__legend">
        <span><i style={{ background: '#ffef3a' }} />보정량 (mm)</span>
        <span><i style={{ background: '#e01b1b' }} />보정 지점과 지시선</span>
        <span><i style={{ background: '#ff3b30' }} />제로라인</span>
        <span><i style={{ background: 'repeating-linear-gradient(90deg,#ff3b30 0 4px,transparent 4px 7px)' }} />빈 공간을 지나는 구간</span>
        <small>보정시트와 같은 표기입니다</small>
      </div>
    )}
    </div>

    <div className="cad-viewer__hud">
      {/* 표면만 쓴다. 모서리·삼각망은 형상 확인용으로 넣었는데 실제
          작업에는 안 쓰이고 버튼 줄만 길어졌다. 편차 색도 뺀다 —
          정합이 조금만 어긋나도 엉뚱한 자리가 물들어 오해를 부르고,
          제대로 맞아도 색이 옅어 잘 읽히지 않았다. */}
      <button type="button" className={showPlanes ? 'is-on' : ''}
        onClick={() => setShowPlanes((v) => !v)}>
        평면 {mesh.planes?.length ?? 0}
      </button>
      {showHoles && holes.length > 0 && (
        <span className="cad-viewer__pick-level">
          <label htmlFor="cad-hole-floor">홀 Ø</label>
          <input id="cad-hole-floor" type="range" min={0}
            max={holeSizes[0] ?? 0} step={0.1} value={holeFloor}
            onChange={(event) => setHoleFloor(Number(event.target.value))} />
          <input className="cad-viewer__num" type="number" min={0} step={0.5}
            aria-label="홀 지름 하한" value={holeFloor}
            onChange={(event) => {
              const low = Number(event.target.value);
              if (Number.isFinite(low)) setHoleFloor(low);
            }} />
          <b>mm 이상</b>
          <em>{holes.filter((h) => h.diameter >= holeFloor).length}
            {' / '}{holes.length}개 · {holeSizes.length}종</em>
        </span>
      )}
      {overlay && <button type="button" className={showOverlay ? 'is-on' : ''}
        onClick={() => setShowOverlay((v) => !v)}>
        제로라인·보정량 {overlayPoints}
      </button>}
      {overlay && showOverlay && <span className="cad-viewer__pick-level">
        <label htmlFor="cad-threshold">보정량</label>
        <input id="cad-threshold" type="range" min={0} max={9} step={0.1}
          value={threshold}
          onChange={(event) => {
            const low = Number(event.target.value);
            setThreshold(low);
            if (low > ceiling) setCeiling(low);
          }} />
        {/* 슬라이더만 있으면 0.35 같은 값을 정확히 못 맞춘다.
            숫자로도 넣을 수 있게 둘을 같은 값에 묶는다. */}
        <input className="cad-viewer__num" type="number" min={0} step={0.1}
          aria-label="보정량 하한" value={threshold}
          onChange={(event) => {
            const low = Number(event.target.value);
            if (!Number.isFinite(low)) return;
            setThreshold(low);
            if (low > ceiling) setCeiling(low);
          }} />
        <label htmlFor="cad-ceiling">~</label>
        <input id="cad-ceiling" type="range" min={0} max={9} step={0.1}
          value={ceiling}
          onChange={(event) => {
            const high = Number(event.target.value);
            setCeiling(high);
            if (high < threshold) setThreshold(high);
          }} />
        <input className="cad-viewer__num" type="number" min={0} step={0.1}
          aria-label="보정량 상한" value={ceiling}
          onChange={(event) => {
            const high = Number(event.target.value);
            if (!Number.isFinite(high)) return;
            setCeiling(high);
            if (high < threshold) setThreshold(high);
          }} />
        <b>mm</b>
        {sheetCount > overlayPoints && (
          <em>{sheetCount - overlayPoints}개 숨김</em>
        )}
      </span>}
      <button type="button" className={measuring ? 'is-on' : ''}
        onClick={() => { setMeasuring((v) => !v); setMeasure(null); setNoting(false); }}>
        측정
      </button>
      <button type="button" className={noting ? 'is-on' : ''}
        onClick={() => { setNoting((v) => !v); setNoteDraft(null);
                         setMeasuring(false); setZoning(false); }}>
        주석 {notes?.length ? notes.length : ''}
      </button>
      {onRegionsChange && <button type="button" className={zoning ? 'is-on' : ''}
        onClick={() => { setZoning((v) => !v); setMeasuring(false); setNoting(false); }}>
        공정 구역 {regions?.length ? regions.length : ''}
      </button>}
      <span className="cad-viewer__stat">
        삼각형 {mesh.summary.n_faces.toLocaleString()} · {holeLabel}
        {mesh.counts?.cylinders
          ? ` · 굽힘 R ${mesh.counts.cylinders - holes.length}`
          : ''}
      </span>
      <span className="cad-viewer__stat cad-viewer__hint">
        {zoning ? (zoneTool === 'brush'
          ? '형상 위를 눌러 공정 구역을 칠합니다'
          : '형상 위에서 끌어 공정 구역을 잡습니다')
          : noting ? '형상 위를 눌러 메모를 답니다'
          : measuring ? '형상 위 두 곳을 눌러 거리를 잽니다'
          : '가운데 이동 · 가운데+오른쪽 회전 · Ctrl+가운데 확대 · 왼쪽 선택 · 콜아웃 눌러 수정'}
      </span>
    </div>
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

    {zoning && (
      <div className="cad-viewer__zones">
        <div className="cad-viewer__zones-size">
          <span className="cad-viewer__zone-tools" role="group" aria-label="구역 도구">
            {([['rect', '네모'], ['circle', '동그라미'], ['brush', '붓']] as const)
              .map(([kind, name]) => (
                <button key={kind} type="button"
                  className={zoneTool === kind ? 'is-on' : ''}
                  onClick={() => setZoneTool(kind)}>{name}</button>
              ))}
          </span>
          {zoneTool === 'brush' ? <>
            <label htmlFor="cad-zone-size">붓 크기</label>
            <input id="cad-zone-size" type="range" min={0.04} max={0.4} step={0.01}
              value={zoneRadius}
              onChange={(event) => setZoneRadius(Number(event.target.value))} />
          </> : <em>형상 위에서 끌면 그만큼이 구역이 됩니다</em>}
        </div>
        {(regions ?? []).map((region, order) => (
          <div key={region.id}
            className={`cad-viewer__zone${region.id === activeZone ? ' is-active' : ''}`}>
            {/* 번호를 누르면 그 구역에 덧칠한다. 안 고르면 새 구역이 된다. */}
            <button type="button" className="cad-viewer__zone-pick"
              title={region.id === activeZone ? '덧칠 중' : '이 구역에 덧칠'}
              onClick={() => setActiveZone(
                region.id === activeZone ? null : region.id)}>
              {CIRCLED[order] ?? order + 1}
            </button>
            <select value={region.die} aria-label="금형"
              onChange={(event) => onRegionsChange?.((regions ?? []).map((other) =>
                other.id === region.id
                  ? { ...other, die: event.target.value as CadRegion['die'] }
                  : other))}>
              {DIE_CHOICES.map((name) => <option key={name}>{name}</option>)}
            </select>
            <select value={region.work} aria-label="공정"
              onChange={(event) => onRegionsChange?.((regions ?? []).map((other) =>
                other.id === region.id
                  ? { ...other, work: event.target.value as CadRegion['work'] }
                  : other))}>
              {WORK_CHOICES.map((name) => <option key={name}>{name}</option>)}
            </select>
            <button type="button" aria-label={`구역 ${order + 1} 삭제`}
              onClick={() => {
                if (region.id === activeZone) setActiveZone(null);
                onRegionsChange?.(
                  (regions ?? []).filter((other) => other.id !== region.id));
              }}>×</button>
          </div>
        ))}
        {(regions ?? []).length > 0 && (
          <button type="button" className="cad-viewer__zone-new"
            onClick={() => setActiveZone(null)}
            disabled={activeZone === null}>
            + 새 구역으로 칠하기
          </button>
        )}
        <p>{activeZone
          ? '고른 구역에 덧칠합니다 — 번호를 다시 누르면 해제됩니다'
          : '형상 위를 누른 채 끌어서 칠하세요. 놓았다 다시 끌면 새 구역입니다'}</p>
      </div>
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
        스캔 위의 점 중 {Math.round((overlay.fit.hit_rate ?? 0) * 100)}% 만
        형상에 얹혔습니다 (기준 60%). 제로라인·보정량을 그리지
        않았습니다 — 틀린 자리에 그리는 것보다 안 그리는 쪽을 택했습니다.
        {/* 단면 표기는 스캔 정합을 고치는 방법이 아니다. 스캔을 아예
            쓰지 않고 CAD 를 시트가 적어 준 숫자로 잘라 제로라인을
            얻는, **따로 가는 길**이다. 그래서 얹힘 비율과 무관하게
            정확하다 — 여기서 그 점을 분명히 말해 둔다. */}
        {sections?.length
          ? <> 아래 <b>시트 단면 표기</b>로 그린 제로라인
              {' '}{sections.length}개는 스캔 정합과 무관하게 맞습니다 —
              시트가 적어 준 값으로 CAD 를 직접 자른 것이라 추정이 없습니다.</>
          : <> 이 부품은 <b>시트 단면 표기</b>(H·T 값)로 제로라인을 계산하세요.
              스캔을 쓰지 않고 CAD 를 그 값으로 직접 자르는 별개의 방법이라,
              얹힘 비율이 낮아도 결과는 정확합니다.</>}
      </p>
    )}
  </>;
}
