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
  points: { id: string; position: [number, number, number]; value: number;
             /* CAD 원래 좌표(mm). CATIA 작업자가 그 자리에 값을 넣는다. */
             cad?: [number, number, number] }[];
  scanPart?: string | null;
  /* 화면용 정점 하나하나의 스캔 편차(mm). 부품 밖이면 null. */
  surfaceDeviation?: (number | null)[];
  deviationRange?: [number, number] | null;
  /* 컬러바 범위를 벗어나 제외한 판독. 실측 JD_67XX6 에서 +9.00 이
     5건 나왔는데 그 부품 컬러바는 +3.0~-3.0 이다. */
  rejected?: { id: string; value: number }[];
  /* 소수점이 날아간 판독을 되살린 것. 실측 64XX2 에서 6 · 80 · -10 이
     각각 0.6 · 0.8 · -1.0 이었다. 조용히 고치면 안 되므로 화면에 알린다. */
  mended?: { id: string; was: number; value: number }[];
  /* 어디서 온 포인트인가. 'workspace' 면 검사 원본(PolyWorks)에서 부품
     좌표로 그대로 가져온 것이라 정합을 하지 않았다 — 얹힘 대신
     surfaceGap(표면까지 실제 거리)을 봐야 한다. */
  source?: 'scan' | 'workspace';
  sourceName?: string;
  surfaceGap?: { median: number; max: number } | null;
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
  /* CATIA 가 STEP 에 넣어 둔 부품 색(넓이 기준 대표색). 없으면 기본색을
     쓴다 — 실측 64XX1 은 #00FF00, 71XX1 은 #FFFFFF, 67XX6 은 여섯 색이다. */
  colour?: string | null;
  palette?: string[];
  /* 색이 같은 삼각형 구간 [색, 시작삼각형, 개수]. CATIA 는 한 부품을
     여러 색으로 칠한다 — 실측 71XX1 은 회색 몸통에 아랫부분만 분홍이다. */
  colourGroups?: [string, number, number, boolean?][];
  note?: string;
  symmetricPair?: {
    axis: number; middle: number; side: number; part: number;
    matchedToScan: boolean;
  };
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
  /* **부품 좌표(mm)로 미리 등록해 둔 구역.**
     
     보정시트는 "① 하형 용접 · ② 상형 심고음" 처럼 구역을 고정된 자리에
     표기한다. 같은 부품이면 그 자리가 매번 같으므로, 손으로 다시 그릴
     것이 아니라 좌표로 한 번 등록해 두면 그 부품을 열 때마다 저절로
     뜬다. 좌표는 CAD 원래 좌표계다(화면은 원점을 옮겨 놓으므로 그릴 때
     되돌린다). */
  box?: { min: [number, number, number]; max: [number, number, number] };
  die: '상형' | '하형';
  work: '용접' | '가공' | '심고음';
  /* 시트의 "상형 인서트 스틸 이음매(도면 확인)" 같은 메모. */
  note?: string;
  /* 좌표로 등록해 둔 표준 구역에서 왔는가 — 손으로 그린 것과 구분한다. */
  standard?: boolean;
  /* 제로라인 기준으로 저절로 잡은 구역인가.
     id 앞글자로 가리려다 손으로 그린 구역도 `Z-` 로 시작한다는 걸
     놓쳤다 — 그대로 뒀으면 자동 구역을 지울 때 손으로 칠한 것까지
     같이 지워졌다. 지우고 다시 잡는 대상을 이 표시로만 고른다. */
  auto?: boolean;
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

/* 가린 것이 없을 때 늘 같은 배열을 준다. 렌더마다 새 배열을 만들면
   그걸 보고 도는 이펙트가 매번 다시 돈다. */
const EMPTY_HIDES: CadRegion['shape'][] = [];

export const DIE_CHOICES: CadRegion['die'][] = ['상형', '하형'];
export const WORK_CHOICES: CadRegion['work'][] = ['용접', '가공', '심고음'];
/** 시트가 쓰는 원문자. 구역이 열 개를 넘을 일은 없다. */
/** 보정 후 형상 — 원본과 견줘 보려고 만든다. */
export type CadMorph = {
  positions: number[];
  shift: number[];
  stats: { moved: number; max_shift: number; mean_shift: number; reach_mm: number };
  points: number;
  /* 공정별 물량. 보정량의 부호가 곧 공정이다 —
     + 살을 붙인다(용접), - 살을 깎는다(CNC 가공). */
  work?: { kind: 'weld' | 'cut'; area_mm2: number; volume_mm3: number;
           max_mm: number; mean_mm: number; faces: number }[];
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

/** 표준 뷰 — CATIA 도면의 정면도·측면도·평면도와 같은 자리.
 *
 * [축을 어떻게 아는가 — 추측이 아니다]
 * 이 CAD 는 차량 좌표계로 들어온다. 71XX1-DR000_HDCT0458.stp 실측 범위가
 *
 *     X 1445.0 ~ 1990.0   (0 에서 멀다 = 차량 앞뒤로 떨어진 자리)
 *     Y -795.5 ~  795.5   (0 을 가운데 두고 대칭 = 차량 중심선 기준 좌우)
 *     Z    30.5 ~ 1260.2  (바닥에서 올라간 높이)
 *
 * 이고, 보정시트의 "H : 300 · T : 1700" 표기에서 H(높이)가 Z 범위에만,
 * T(전후)가 X 범위에만 들어간다. 그래서 X=전후 · Y=좌우 · Z=높이다.
 * section_zero.py 의 AXIS_OF 도 같은 근거로 정해 뒀다.
 *
 * [그래서 도면의 뷰는 이렇게 된다]
 * 정면도는 차를 앞에서 본 그림이므로 **X 축을 따라** 본다. 예전에는
 * 정면이 [0,-1,0](Y 축을 따라 봄)이라 실제로는 측면도였고, 우측이
 * [1,0,0] 이라 그게 정면도였다 — 이름과 그림이 서로 바뀌어 있었다.
 *
 * dir 은 부품에서 카메라로 가는 방향이다. */
const VIEWS: { id: string; label: string; dir: [number, number, number] }[] = [
  // CATIA 의 기본 등각과 같은 팔분면(앞·좌·위)에서 본다.
  { id: 'iso', label: '등각', dir: [1, -0.85, 0.75] },
  { id: 'front', label: '정면', dir: [-1, 0, 0] },
  { id: 'rear', label: '배면', dir: [1, 0, 0] },
  { id: 'left', label: '좌측', dir: [0, -1, 0] },
  { id: 'right', label: '우측', dir: [0, 1, 0] },
  { id: 'top', label: '평면', dir: [0, 0, 1] },
  { id: 'bottom', label: '저면', dir: [0, 0, -1] },
];

/** 좌표축 이름 — 차량 좌표계에서 무엇을 뜻하는지 같이 적는다. */
const AXIS_MEANING: [string, string, string] = ['X 전후', 'Y 좌우', 'Z 높이'];

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
  // 시트의 "상형 인서트 스틸 이음매(도면 확인)" 같은 메모를 아랫줄에
  // 붙일 수 있어야 한다. 한 줄만 그리면 그 말이 통째로 사라진다.
  const rows = text.split(String.fromCharCode(10));
  const big = '700 44px ui-sans-serif, system-ui, sans-serif';
  const small = '500 34px ui-sans-serif, system-ui, sans-serif';
  const line = 58;
  let width = 0;
  rows.forEach((row, i) => {
    measure.font = i ? small : big;
    width = Math.max(width, Math.ceil(measure.measureText(row).width));
  });
  const canvas = document.createElement('canvas');
  canvas.width = width + pad * 2;
  canvas.height = 12 + line * rows.length;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgba(255,246,250,.96)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 4;
  ctx.strokeStyle = '#d61f77';
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  rows.forEach((row, i) => {
    ctx.font = i ? small : big;
    ctx.fillStyle = i ? '#8a3a68' : '#b31563';
    ctx.fillText(row, canvas.width / 2, 6 + line * (i + 0.5));
  });

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
  onCapture?: (dataUrl: string, viewName: string) => void;
  regions?: CadRegion[];
  onRegionsChange?: (regions: CadRegion[]) => void;
  /* 보정 후 형상. 있으면 원본과 겹쳐 보거나 갈아 끼울 수 있다. */
  morph?: CadMorph | null;
  morphMode?: 'off' | 'after' | 'both';
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [rendererRetry, setRendererRetry] = useState(0);
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
  /* 지금 보고 있는 시점의 이름. 시트에 담을 때 어느 방향에서 본
     그림인지 같이 적는다 — 현업 시트도 전체도와 상세도를 따로 싣고,
     나중에 보는 사람은 그림만으로는 방향을 못 가린다. */
  const [viewName, setViewName] = useState('등각');
  /* 흰 바탕. 시트에 실을 그림이라 이쪽이 기본이다 — 어두운 네모가
     엑셀에 통째로 박히면 인쇄에서 튄다. 화면에서 히트맵을 볼 때는
     어두운 배경이 색을 잘 읽어 주므로 버튼으로 바꾼다.
     씬을 다시 만들 때 최신 값을 봐야 하므로 ref 도 같이 둔다. */
  const [light, setLight] = useState(true);
  const lightRef = useRef(true);
  lightRef.current = light;
  /* 부품 색을 손으로 정한 값. STEP 이 색을 안 들고 있거나(면 대부분이
     색 없음) CATIA 에서 보던 것과 다를 때 쓴다 — 실측 67XX6 은 파랑
     #0080FF 이 팔레트에 등록만 돼 있고 어느 면에도 안 칠해져 있었다. */
  /* 손으로 정한 색은 **부품마다 따로** 기억한다.
     뷰어 컴포넌트는 CAD 를 바꿔도 그대로 살아 있어서, 값 하나로 두면
     한 부품 색을 바꿨을 때 세 부품이 다 바뀐다. 열쇠는 파일 이름이다 —
     cadId 는 열 때마다 새로 생겨 새로고침을 못 넘긴다(주석·구역과 같은
     방식이다). */
  const [paintNote, setPaintNote] = useState('');
  /* 보고 싶지 않은 자리를 가려 둔다.
     시트에 실을 그림에서 반대편 살이나 옆 부품이 겹쳐 보이는 일이 잦다.
     네모·동그라미로 훑으면 그 안의 삼각형을 아예 안 그린다 — 형상을
     지우는 것이 아니라 이 화면에서만 감추는 것이라 되돌릴 수 있다.
     색과 같이 **부품마다** 따로 기억한다. */
  const [hiding, setHiding] = useState(false);
  const [hideByCad, setHideByCad] = useState<Record<string, CadRegion['shape'][]>>({});
  const [tintByCad, setTintByCad] = useState<Record<string, string>>({});
  const tintKey = mesh.summary.name;
  const partTint = tintByCad[tintKey] ?? null;
  const partTintRef = useRef<string | null>(null);
  partTintRef.current = partTint;
  const hides = hideByCad[tintKey] ?? EMPTY_HIDES;
  /* 씬을 다시 만들지 말지 가리는 열쇠.
   *
   * 이 배열을 그대로 의존성에 넣었더니 `?? []` 가 렌더마다 새 배열을
   * 만들어, 화면이 다시 그려질 때마다 씬을 통째로 새로 지었다. 카메라가
   * 매번 처음 자리로 돌아가니 확대도 표준 뷰도 먹히지 않고, WebGL
   * 컨텍스트가 계속 새로 열려 오류까지 났다. 내용이 바뀔 때만 달라지는
   * 글자로 바꿔 건다 — 도형 몇 개뿐이라 값이 싸다. */
  const hideKey = hides.length ? JSON.stringify(hides) : '';
  const hidingRef = useRef(false);
  hidingRef.current = hiding;
  const hidesRef = useRef<CadRegion['shape'][]>([]);
  hidesRef.current = hides;
  const addHide = (shape: CadRegion['shape']) => setHideByCad((current) => ({
    ...current, [tintKey]: [...(current[tintKey] ?? []), shape],
  }));
  const setPartTint = (tone: string | null) => setTintByCad((current) => {
    if (tone === null) {
      const next = { ...current };
      delete next[tintKey];
      return next;
    }
    return { ...current, [tintKey]: tone };
  });
  /* 화면에 그릴 좌표축 방향(카메라 기준). 카메라가 움직일 때마다 갱신한다. */
  const [axisView, setAxisView] = useState<[number, number, number][]>(
    [[1, 0, 0], [0, -1, 0], [0, 0, 1]]);
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
    snapshot: (scale?: number) => string;
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
      // 캡처 함수가 저장 직전에 직접 render() 하므로 프레임 버퍼를 계속
      // 붙들고 있을 필요가 없다. 끄면 GPU 메모리와 화면 합성 비용이 준다.
      renderer = new THREE.WebGLRenderer({
        antialias: true, alpha: false, preserveDrawingBuffer: false });
    } catch (primaryError) {
      // 브라우저의 WebGL 컨텍스트/멀티샘플 자원이 잠시 부족한 경우에는
      // 안티앨리어싱을 끈 가벼운 설정으로 한 번 더 시도한다.
      try {
        renderer = new THREE.WebGLRenderer({
          antialias: false,
          alpha: false,
          preserveDrawingBuffer: false,
          powerPreference: 'default',
        });
      } catch (fallbackError) {
        console.error('[cad-viewer] WebGL renderer initialization failed', {
          primaryError,
          fallbackError,
        });
        setError('3D 화면을 시작하지 못했습니다. 다른 CAD 탭을 닫거나 페이지를 새로고침해 주세요.');
        return;
      }
    }
    // Fast Refresh 뒤 이전 오류 상태가 남아 있더라도 재시도가 성공하면 지운다.
    setError(null);
    const fullPixelRatio = Math.min(window.devicePixelRatio, 2);
    renderer.setPixelRatio(fullPixelRatio);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    // 배경. 시트에 실을 그림은 흰 바탕이라야 엑셀에서 잘려 보이지
    // 않는다 — 어두운 네모가 통째로 박히면 인쇄에서도 튄다. 화면에서
    // 히트맵을 볼 때는 어두운 쪽이 색이 잘 읽혀 버튼으로 바꿀 수 있게 뒀다.
    renderer.setClearColor(lightRef.current ? 0xffffff : 0x16202a);
    renderer.localClippingEnabled = true;
    // 금속은 밝은 곳을 반사해야 형태가 읽힌다. 톤매핑 없이 두면
    // 반사 하이라이트가 흰색으로 다 타버린다.
    // 그림자는 끈다. team-15 뷰어에서 가져왔다가 되돌렸다 — 그쪽은
    // 로봇·용접건처럼 **서로 떨어진 물체 여러 개**라 그림자가 공간을
    // 설명해 준다. 우리는 얇은 판금 껍데기 **하나**뿐이라 자기그림자밖에
    // 안 생기고, 두께가 없다시피 해서 앞뒤 면이 서로를 가려 얼룩진다
    // (shadow acne). 형상이 깨져 보이던 원인이다.
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    /* 흰 바탕에서는 노출을 낮춰야 색이 산다.
     *
     * 어두운 배경에 맞춘 0.8 을 흰 바탕에 그대로 쓰면, CATIA 색이 밝은
     * 회색(#C1C4C0, 0.76)이라 주광·반구광·환경 반사가 겹쳐 1 을 넘고
     * 하얗게 타 버린다 — 부품이 배경과 구분이 안 되고, 옆에 칠해진
     * 분홍·초록도 흰색으로 날아간다. 배경이 밝은 만큼 노출을 내린다. */
    renderer.toneMappingExposure = lightRef.current ? 0.58 : 0.8;
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

    /* 가려 둔 자리는 아예 그리지 않는다.
     *
     * 삼각형을 빼면 색 구간(colourGroups)의 번호도 어긋나므로 함께 다시
     * 센다. 원래 순서를 지키며 걸러 내므로 구간은 여전히 이어진 토막이다.
     * 형상 자체는 손대지 않는다 — 목록을 비우면 그대로 돌아온다. */
    const drop = hidesRef.current.filter(Boolean) as NonNullable<CadRegion['shape']>[];
    let keptIndices = mesh.indices;
    let keptGroups = mesh.colourGroups ?? [];
    if (drop.length) {
      const cover = drop.map((shape) => {
        const u = new THREE.Vector3(...shape.u);
        const v = new THREE.Vector3(...shape.v);
        const middle = new THREE.Vector3(...shape.center);
        const normal = new THREE.Vector3().crossVectors(u, v).normalize();
        // 판금은 앞뒤 껍질이 붙어 있다. 두께 방향으로 넉넉히 잡아야
        // 가린 자리에서 반대쪽 살이 비쳐 보이지 않는다.
        const deep = Math.max(shape.hu, shape.hv);
        return (p: THREE.Vector3) => {
          const d = p.clone().sub(middle);
          if (Math.abs(d.dot(normal)) > deep) return false;
          const du = d.dot(u), dv = d.dot(v);
          if (shape.kind === 'rect') {
            return Math.abs(du) <= shape.hu && Math.abs(dv) <= shape.hv;
          }
          const nu = du / (shape.hu || 1), nv = dv / (shape.hv || 1);
          return nu * nu + nv * nv <= 1;
        };
      });
      const spot = new THREE.Vector3();
      const middleOf = (a: number, b: number, c: number) => spot.set(
        (mesh.positions[a * 3] + mesh.positions[b * 3] + mesh.positions[c * 3]) / 3,
        (mesh.positions[a * 3 + 1] + mesh.positions[b * 3 + 1] + mesh.positions[c * 3 + 1]) / 3,
        (mesh.positions[a * 3 + 2] + mesh.positions[b * 3 + 2] + mesh.positions[c * 3 + 2]) / 3);
      // 삼각형마다 어느 색 구간에 속했는지 미리 적어 둔다.
      const bandOf = new Int32Array(mesh.indices.length / 3).fill(-1);
      keptGroups.forEach(([, start, count], slot) => {
        for (let t = start; t < start + count; t += 1) bandOf[t] = slot;
      });
      const keep: number[] = [];
      const bandRun: number[] = [];
      for (let t = 0; t < mesh.indices.length / 3; t += 1) {
        const a = mesh.indices[t * 3], b = mesh.indices[t * 3 + 1], c = mesh.indices[t * 3 + 2];
        if (cover.some((inside) => inside(middleOf(a, b, c)))) continue;
        keep.push(a, b, c);
        bandRun.push(bandOf[t]);
      }
      keptIndices = keep;
      const rebuilt: [string, number, number, boolean?][] = [];
      let at = 0;
      while (at < bandRun.length) {
        const slot = bandRun[at];
        let run = 1;
        while (at + run < bandRun.length && bandRun[at + run] === slot) run += 1;
        if (slot >= 0) {
          const [tone, , , direct] = keptGroups[slot];
          rebuilt.push([tone, at, run, direct]);
        }
        at += run;
      }
      keptGroups = rebuilt;
    }
    geometry.setIndex(keptIndices);
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
    // 색 순서: 히트맵을 칠할 때는 정점색 -> 아니면 CATIA 색 -> 기본색.
    // 손으로 정한 색이 먼저, 없으면 CATIA 가 STEP 에 넣어 둔 색.
    const chosen = partTintRef.current ?? mesh.colour;
    const catia = (!painted && chosen) ? new THREE.Color(chosen) : null;

    /* CATIA 는 한 부품을 여러 색으로 칠한다 — 실측 71XX1 은 회색 몸통
     * (삼각형 74,594)에 아랫부분만 분홍(11,581)이다. 색이 같은 삼각형을
     * 구간으로 받아 재질을 여러 개 붙인다. 정점마다 색을 실으면 30만 개가
     * 넘어 무겁고, 히트맵이 쓰는 정점색과도 부딪힌다.
     *
     * 히트맵을 칠할 때나 손으로 색을 정했을 때는 구간을 쓰지 않는다 —
     * 그때는 온 부품이 한 색이어야 뜻이 맞는다. */
    const groups = keptGroups;
    // 구간이 삼각형을 모두 덮는지 본다. 빠진 삼각형이 있으면 그 자리는
    // 재질이 없어 **아예 안 그려진다** — 색 하나로 칠하는 편이 낫다.
    const covered = groups.reduce((sum, [, , count]) => sum + count, 0);
    const whole = keptIndices.length / 3;
    const bands = (!painted && !partTintRef.current
                   && groups.length > 1 && whole > 0 && covered === whole)
      ? groups : null;
    const surface = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
      color: painted ? 0xffffff : (catia ?? SURFACE),
      vertexColors: painted,
      metalness: painted ? 0.05 : 0.15,
      roughness: painted ? 0.85 : 0.62,
      // 환경맵을 세게 주면 판넬이 하얗게 번진다. 형태는 주광이 만든다.
      // 환경 반사를 세게 주면 밝은 CATIA 색이 흰 바탕에서 날아간다.
      envMapIntensity: painted ? 0.25 : (lightRef.current ? 0.22 : 0.45),
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

    if (bands) {
      // 구간마다 같은 설정에 색만 바꾼 재질을 붙인다.
      const base = surface.material as THREE.MeshStandardMaterial;
      /* 겹친 껍질을 어떻게 가르나.
       *
       * 실측 71XX1 은 부품 전체를 덮는 회색 껍질 위에 색칠한 껍질이
       * **0.1mm 안으로 겹쳐** 있다(분홍 삼각형 중심 3,861개 중 98%).
       * 그냥 그리면 깊이 싸움이 나고 회색이 이겨 색이 하나도 안 보인다 —
       * 화면이 온통 회색으로 나오던 것이 이것이다.
       *
       * 직접 칠한 껍질을 카메라 쪽으로 살짝 당겨 늘 이기게 한다. 실제
       * 자리를 옮기는 것이 아니라 깊이만 비켜 주는 것이라 형상은 그대로다. */
      const coats = bands.map(([tone, , , direct]) => {
        const coat = base.clone();
        coat.color = new THREE.Color(tone);
        if (direct) {
          coat.polygonOffset = true;
          coat.polygonOffsetFactor = -2;
          coat.polygonOffsetUnits = -2;
        }
        return coat;
      });
      geometry.clearGroups();
      bands.forEach(([, start, count], slot) => {
        // three 는 삼각형이 아니라 정점 번호로 센다.
        geometry.addGroup(start * 3, count * 3, slot);
      });
      base.dispose();
      // Mesh 는 재질 하나로 만들어 두었다. 여러 개를 다는 것은 three 가
      // 허락하지만 타입 정의가 좁아 여기서만 넓혀 준다.
      (surface as unknown as { material: THREE.Material[] }).material = coats;
    }
    /* 색이 왜 이렇게 나오는지 화면에서 바로 알 수 있게 적어 둔다.
       "5가지" 라고 떠도 실제로 안 칠해지는 경우가 있어서, 재질이 몇 개
       붙었는지와 안 붙었으면 그 까닭을 함께 남긴다. */
    setPaintNote(bands
      ? `색 ${bands.length}개 적용`
      : painted ? '편차색 표시 중'
      : partTintRef.current ? '내가 정한 색'
      : groups.length <= 1 ? `구간 ${groups.length}개`
      : `구간이 면을 다 못 덮음 ${covered}/${whole}`);

    // 겹쳐 볼 때는 원본을 반투명 뼈대로 남긴다
    if (morph && morphMode === 'after') surface.visible = false;
    else if (morph && morphMode === 'both') {
      const coats = Array.isArray(surface.material)
        ? surface.material : [surface.material];
      for (const coat of coats as THREE.MeshStandardMaterial[]) {
        coat.transparent = true;
        coat.opacity = 0.28;
      }
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

    // 화면에서는 표면 모드만 제공한다. 숨겨진 모서리·와이어 형상을 매번
    // 만들면 40만 면 CAD를 처음 열 때 CPU와 메모리만 크게 사용한다.
    edgeLines.current = null;
    const shownAs = detailRef.current;
    surface.visible = shownAs !== 'wire';

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
    // 정합 신뢰도는 경고 여부만 결정한다. 서버가 표면 투영에 성공해 돌려준
    // 포인트와 제로라인까지 숨기면 사용자는 수동 보정조차 할 수 없다.
    if (overlay) {
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
    // 흰 바탕에서는 아래쪽 반사광을 밝게 준다. 어두운 배경에 맞춰
    // 놓은 짙은 바닥색(0x1e293b)을 그대로 쓰면 판 아랫면이 새까맣게
    // 죽어 흰 바탕에서 얼룩처럼 보인다.
    /* 흰 바탕이라고 전체를 밝히면 안 된다.
     *
     * 처음에 흰 배경으로 바꾸면서 반구광을 0.3 에서 0.55 로 올리고 주광을
     * 0.85 에서 0.7 로 낮췄더니 부품이 납작해 보였다 — 사방에서 고르게
     * 비추면 면과 면을 가르는 그늘이 사라진다. 흰 바탕에서 형태를 세우는
     * 것은 밝기가 아니라 **밝은 면과 그늘의 차이**다. 그래서 반구광은
     * 어두운 배경 때와 같게 두고 주광을 오히려 더 세게 준다. */
    const pale = lightRef.current;
    scene.add(new THREE.HemisphereLight(
      0xdbeafe, pale ? 0xa8b4c0 : 0x1e293b, 0.3));
    const key = new THREE.DirectionalLight(0xffffff, pale ? 0.9 : 0.85);
    key.position.set(1, 1.4, 1).multiplyScalar(radius * 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x93c5fd, pale ? 0.22 : 0.3);
    fill.position.set(-1.2, -0.6, -0.9).multiplyScalar(radius * 3);
    scene.add(fill);
    // 흰 바탕에서는 뒤에서 치는 림이 윤곽을 오히려 지운다(흰 위에 흰).
    // 대신 아래에서 살짝 받쳐 바닥면이 새까맣게 죽지 않게만 한다.
    const rim = new THREE.DirectionalLight(
      pale ? 0xd8e2ec : 0xf8fafc, pale ? 0.2 : 0.4);
    rim.position.set(-0.4, pale ? -0.9 : 0.3, -1.4).multiplyScalar(radius * 3);
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

    // 기본 화면은 가장 얇은 축에서 바라봐 판금의 넓은 면이 먼저 보이게 한다.
    // 등각 고정 시 긴 패널이 옆면만 보이는 문제가 있었다.
    const dimensions = box.getSize(new THREE.Vector3());
    const thinAxis = dimensions.x <= dimensions.y && dimensions.x <= dimensions.z
      ? 0 : dimensions.y <= dimensions.z ? 1 : 2;
    const startDir = new THREE.Vector3();
    startDir.setComponent(thinAxis, 1);
    startDir.setComponent((thinAxis + 1) % 3, 0.12);
    startDir.setComponent((thinAxis + 2) % 3, 0.08);
    startDir.normalize();
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
    const spherical = new THREE.Spherical();
    /* 자리에서 각도를 되짚는다 — **Z 가 위**인 기준으로.
     *
     * three 의 Spherical.setFromVector3 는 Y 가 위인 각도를 준다. 자리를
     * 잡는 쪽(applyCamera)은 Z 가 위인데 여기서만 Y 기준으로 잡아 두면
     * 시작 각도부터 어긋나, 처음 끌자마자 화면이 튀고 그 뒤로도 계속
     * 엉뚱한 축으로 돈다. 두 곳이 같은 식을 써야 한다. */
    const setAngles = (offset: THREE.Vector3) => {
      const away = offset.length() || 1;
      spherical.radius = away;
      spherical.phi = Math.acos(Math.max(-1, Math.min(1, offset.z / away)));
      spherical.theta = Math.atan2(offset.y, offset.x);
    };
    setAngles(camera.position.clone().sub(target));
    let mode: 'none' | 'pan' | 'rotate' | 'zoom' = 'none';
    let last = { x: 0, y: 0 };
    let qualityTimer: ReturnType<typeof setTimeout> | null = null;
    let interactiveQuality = false;

    // 고해상도 화면은 devicePixelRatio=2라 픽셀 수가 네 배다. 드래그와 휠
    // 조작 중에만 1배 해상도로 낮추고 손을 놓으면 원래 품질로 복구한다.
    // 형상과 계산 결과는 건드리지 않아 저사양 PC에서도 안전하다.
    const setInteractionQuality = (interacting: boolean) => {
      if (qualityTimer !== null) {
        clearTimeout(qualityTimer);
        qualityTimer = null;
      }
      if (interacting) {
        if (!interactiveQuality && fullPixelRatio > 1) {
          interactiveQuality = true;
          renderer.setPixelRatio(1);
          renderer.setSize(mount.clientWidth, mount.clientHeight, false);
        }
        return;
      }
      qualityTimer = setTimeout(() => {
        if (interactiveQuality) {
          interactiveQuality = false;
          renderer.setPixelRatio(fullPixelRatio);
          renderer.setSize(mount.clientWidth, mount.clientHeight, false);
        }
        qualityTimer = null;
      }, 140);
    };

    /* 카메라를 Z 가 위인 채로 돌린다.
     *
     * [왜 어색했나]
     * three 의 Spherical 은 **Y 가 위**인 좌표를 준다. 그런데 이 부품은
     * 차량 좌표계라 **Z 가 높이**다(축 표시의 X 전후 · Y 좌우 · Z 높이).
     * 그래서 가로로 끌면 부품이 제자리에서 도는 게 아니라 옆으로 굴렀고,
     * 화면의 위쪽도 부품의 위가 아니라 차량 좌우였다. 세워 보려고 눕히기
     * 버튼을 따로 눌러야 했던 것도 이 때문이다.
     *
     * 반지름·각도는 그대로 두고 **자리를 잡는 식만** Z 위로 바꾼다 —
     *     x = r sinφ cosθ · y = r sinφ sinθ · z = r cosφ
     * 이러면 가로 끌기는 높이축(Z) 둘레를 돌고, 세로 끌기는 위아래로
     * 넘긴다. 사람이 부품을 손에 들고 돌리는 것과 같은 느낌이 된다. */
    const applyCamera = () => {
      spherical.phi = Math.max(0.001, Math.min(Math.PI - 0.001, spherical.phi));
      spherical.radius = Math.max(radius * 0.05,
        Math.min(radius * 40, spherical.radius));
      const { radius: away0, phi, theta } = spherical;
      camera.position.copy(target).add(new THREE.Vector3(
        away0 * Math.sin(phi) * Math.cos(theta),
        away0 * Math.sin(phi) * Math.sin(theta),
        away0 * Math.cos(phi)));
      // 시선 축을 중심으로 위쪽 방향을 돌린다 — 화면 안에서만 도는
      // 회전이라 부품을 눕혀 볼 수 있다.
      const look = camera.position.clone().sub(target).normalize();
      camera.up.set(0, 0, 1).applyAxisAngle(look, rollRef.current);
      // 바로 위나 아래에서 내려다보면 위쪽 축이 시선과 겹쳐 무너진다.
      if (Math.abs(camera.up.dot(look)) > 0.999) {
        camera.up.set(0, 1, 0).applyAxisAngle(look, rollRef.current);
      }
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

      /* 좌표축 표시를 카메라와 함께 돌린다.
       *
       * 3D 는 돌려 보는 물건이라 "지금 어느 쪽에서 보고 있는지" 를
       * 이름표만으로는 못 가린다. 축을 화면에 그려 두면 정면·우측이
       * 무엇을 기준으로 한 말인지 볼 때마다 확인된다.
       * 카메라 기준으로 축의 방향만 필요하므로 회전 성분만 쓴다 —
       * 화면 좌표는 x 오른쪽, y 위쪽이라 y 는 부호를 뒤집는다. */
      const spin = new THREE.Matrix4().extractRotation(camera.matrixWorldInverse);
      setAxisView(([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
      ] as [number, number, number][]).map((unit) => {
        const seen = new THREE.Vector3(...unit).applyMatrix4(spin);
        return [seen.x, -seen.y, seen.z] as [number, number, number];
      }));
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
      setInteractionQuality(true);
      const dx = event.clientX - last.x;
      const dy = event.clientY - last.y;
      last = { x: event.clientX, y: event.clientY };

      if (mode === 'rotate') {
        /* 화면을 세로로 한 번 훑으면 꼭 반 바퀴(180도) 돌게 맞춘다.
         * 예전에는 픽셀당 0.005 라디안으로 못 박아 둬서, 창이 크면 한
         * 바퀴 돌리는 데 한참 끌어야 하고 작으면 홱 돌아갔다.
         * 부호는 CATIA 와 같다 — 오른쪽으로 끌면 모델이 오른쪽으로 돈다. */
        /* 부호는 three 의 OrbitControls 와 같게 둔다 — 웹에서 3D 를
         * 돌려 본 사람은 다 그 감각에 익어 있다. 오른쪽으로 끌면 부품이
         * 왼쪽으로 돌아 오른쪽 옆면이 보이고, 아래로 끌면 위에서 내려다본다. */
        const perPixel = Math.PI / Math.max(mount.clientHeight || 1, 1);
        spherical.theta -= dx * perPixel;
        spherical.phi -= dy * perPixel;
        // 손으로 돌린 순간부터는 표준 뷰가 아니다.
        setViewName((current) => (current === '자유 시점' ? current : '자유 시점'));
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
      setInteractionQuality(true);
      spherical.radius *= Math.exp(Math.sign(event.deltaY) * 0.12);
      applyCamera();
      setInteractionQuality(false);
    };
    const onButtonDown = (event: PointerEvent) => {
      if (event.button === 1) event.preventDefault();   // 가운데 자동스크롤 막기
      mode = modeFor(event) as typeof mode;
      last = { x: event.clientX, y: event.clientY };
      if (mode !== 'none') {
        setInteractionQuality(true);
        renderer.domElement.setPointerCapture(event.pointerId);
      }
    };
    const onButtonUp = (event: PointerEvent) => {
      mode = modeFor(event) as typeof mode;
      if (mode === 'none' && renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
      if (mode === 'none') setInteractionQuality(false);
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
        .find((entry: any) => entry.object.userData?.hole);
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
      if ((!zoningRef.current && !hidingRef.current) || event.button !== 0) return;
      event.preventDefault();
      if (zoneToolRef.current === 'brush' && !hidingRef.current) {
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
      if ((!zoningRef.current && !hidingRef.current) || (event.buttons & 1) === 0) return;
      if (zoneToolRef.current === 'brush' && !hidingRef.current) {
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
        outline(zoneToolRef.current === 'brush' ? 'rect' : zoneToolRef.current,
                middle, dragU, dragV, hu, hv));
      preview.visible = true;
    };

    const onPaintUp = (event: PointerEvent) => {
      painting = null;
      if (!dragFrom || (zoneToolRef.current === 'brush' && !hidingRef.current)) {
        dragFrom = null; return;
      }
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
      const drawn = {
        kind: (hidingRef.current && zoneToolRef.current === 'brush'
               ? 'rect' : zoneToolRef.current) as 'rect' | 'circle',
        center: triple(middle), u: triple(dragU), v: triple(dragV), hu, hv,
      };
      if (hidingRef.current) {
        // 형상을 지우는 것이 아니라 이 화면에서만 감춘다.
        addHide(drawn);
        return;
      }
      const current = regionsRef.current ?? [];
      const id = `Z-${Date.now().toString(36)}`;
      setActiveZone(id);
      onRegionsChange?.([...current, {
        id, die: '하형', work: '용접', shape: drawn,
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
      setAngles(dir.clone().multiplyScalar(fitDistance(dir)));
      applyCamera();
    };
    /* 화면을 그림으로 굽는다.
     *
     * [왜 배율이 필요한가]
     * 보정시트는 A4 가로로 인쇄된다. 양식의 그림 자리가 597pt 인데
     * 화면 캔버스는 CSS 1000 x 470 이라, 그대로 실으면 인쇄에서 글자와
     * 지시선이 뭉개진다. 시트에 담을 때만 버퍼를 키워 다시 그린다.
     * setSize 의 셋째 인자를 false 로 둬야 CSS 크기는 그대로고 그리는
     * 버퍼만 커진다 — 화면이 출렁이지 않는다. */
    const snapshot = (scale = 1) => {
      const canvas = renderer.domElement;
      if (scale <= 1) {
        renderer.render(scene, camera);
        return canvas.toDataURL('image/png');
      }
      const wide = canvas.clientWidth || canvas.width;
      const high = canvas.clientHeight || canvas.height;
      const ratio = renderer.getPixelRatio();
      renderer.setPixelRatio(Math.min(ratio * scale, 4));
      renderer.setSize(wide, high, false);
      renderer.render(scene, camera);
      const url = canvas.toDataURL('image/png');
      renderer.setPixelRatio(ratio);
      renderer.setSize(wide, high, false);
      renderer.render(scene, camera);
      return url;
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

    let resizeFrame = 0;
    const resize = new ResizeObserver(() => {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => {
      const w = mount.clientWidth, h = mount.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      // CSS 크기는 건드리지 않아 ResizeObserver 자기 호출을 막는다.
      renderer.setSize(w, h, false);
      });
    });
    resize.observe(mount);

    return () => {
      cancelAnimationFrame(loop);
      cancelAnimationFrame(resizeFrame);
      if (qualityTimer !== null) clearTimeout(qualityTimer);
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
      scene.traverse((node: any) => {
        const any = node as THREE.Mesh;
        any.geometry?.dispose?.();
        const material = any.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(material)) material.forEach((m) => m.dispose());
        else material?.dispose?.();
      });
      scene.environment = null;
      environment.dispose();
      renderer.dispose();
      // 효과 의존성 변경과 개발 중 Fast Refresh가 반복되어도 브라우저의
      // 제한된 WebGL 컨텍스트가 누적되지 않도록 즉시 반환한다.
      renderer.forceContextLoss();
      surfaceRef.current = null;
      overlayGroup.current = null;
      holeGroup.current = null;
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [mesh, holes, showHoles, overlay, sheetValues, showHeat, threshold,
      morph, morphMode, sections, exaggeration, ceiling, light, partTint,
      hideKey, rendererRetry]);

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
    // 부품이 여러 색이면 재질도 여러 개다. 하나로 가정하면 단면이
    // 안 먹는다 — 배열에 clippingPlanes 를 꽂아도 아무 일도 안 일어난다.
    const coats = (Array.isArray(surface.material)
      ? surface.material : [surface.material]) as THREE.MeshStandardMaterial[];
    for (const coat of coats) {
      coat.clippingPlanes = clip === null ? [] : [plane];
      coat.needsUpdate = true;
    }
    plane.constant = clip ?? 0;
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
      const box = region.box;
      if (!shape && !box && !stamps.length) return;

      const hull = new THREE.Vector3();
      let reach = 0;
      let inside: (p: THREE.Vector3) => boolean;

      if (box) {
        // 부품 좌표 -> 화면 좌표. 화면은 원점을 옮겨 놓았다.
        const back = mesh.recentered
          ? new THREE.Vector3(...mesh.summary.bounds.center)
          : new THREE.Vector3();
        const low = new THREE.Vector3(...box.min).sub(back);
        const high = new THREE.Vector3(...box.max).sub(back);
        const span = new THREE.Box3(
          new THREE.Vector3(Math.min(low.x, high.x), Math.min(low.y, high.y),
                            Math.min(low.z, high.z)),
          new THREE.Vector3(Math.max(low.x, high.x), Math.max(low.y, high.y),
                            Math.max(low.z, high.z)));
        span.getCenter(hull);
        reach = span.getSize(new THREE.Vector3()).length() / 2;
        inside = (p) => span.containsPoint(p);
      } else if (shape) {
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
          // 칠은 옅게 깐다. 진하게 칠하면 그 아래 형상과 보정량이 묻힌다.
          color: 0xff5fa8, transparent: true, opacity: 0.26,
          side: THREE.DoubleSide, depthWrite: false,
          // 테두리와 같은 자리를 다투지 않게 살짝 뒤로 민다.
          polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
        }));
        skin.renderOrder = 6;
        root.add(skin);

        /* 테두리 — 구역을 "깔끔하게" 보이게 하는 것은 칠이 아니라 이 선이다.
         *
         * 칠만 하면 삼각망을 따라 가장자리가 들쭉날쭉해 색칠한 자국처럼
         * 보인다(실측 67XX6 에서 그랬다). 고른 삼각형 집합에서 **한 번만
         * 나오는 모서리**가 곧 그 구역의 바깥선이다 — 안쪽 모서리는 이웃한
         * 삼각형 둘이 공유하므로 두 번 나온다. 그 선만 그리면 곡면을 그대로
         * 따라가는 닫힌 윤곽이 남는다. */
        /* 먼저 같은 자리 정점을 하나로 본다(용접).
         *
         * STEP 삼각망은 같은 좌표에 정점을 여러 개 둔다 — 면마다 따로
         * 쪼개 내보내기 때문이다. 실측 64XX1 은 정점 302,340개 중 자리가
         * 서로 다른 것이 92,515개뿐이었고, 그대로 세면 안쪽 모서리가
         * 이웃과 짝을 못 이뤄 셋 중 하나가 바깥선으로 잡힌다. 그러면
         * 테두리가 아니라 그물이 그려진다. */
        const canon = new Map<string, number>();
        const welded = (vertex: number) => {
          const key = `${position.getX(vertex).toFixed(3)}`
            + `_${position.getY(vertex).toFixed(3)}`
            + `_${position.getZ(vertex).toFixed(3)}`;
          const found = canon.get(key);
          if (found !== undefined) return found;
          canon.set(key, vertex);
          return vertex;
        };

        const rim = new Map<string, [number, number]>();
        for (let i = 0; i < keep.length; i += 3) {
          const corner = [keep[i], keep[i + 1], keep[i + 2]];
          for (let k = 0; k < 3; k += 1) {
            const from = welded(corner[k]);
            const to = welded(corner[(k + 1) % 3]);
            if (from === to) continue;          // 눌린 삼각형
            const key = from < to ? `${from}_${to}` : `${to}_${from}`;
            if (rim.has(key)) rim.delete(key);   // 안쪽 모서리
            else rim.set(key, [from, to]);
          }
        }
        if (rim.size) {
          const line: number[] = [];
          rim.forEach(([from, to]) => {
            line.push(position.getX(from), position.getY(from), position.getZ(from));
            line.push(position.getX(to), position.getY(to), position.getZ(to));
          });
          const edge = new THREE.LineSegments(
            new THREE.BufferGeometry().setAttribute(
              'position', new THREE.Float32BufferAttribute(line, 3)),
            new THREE.LineBasicMaterial({ color: 0xd6146e, depthWrite: false }));
          edge.renderOrder = 7;
          root.add(edge);
        }
      }

      // 시트 표기와 같은 말로 적는다 — "① 하형 용접".
      // 메모가 있으면 아랫줄에 붙인다("상형 인서트 스틸 이음매" 같은 것).
      const title = `${CIRCLED[order] ?? order + 1} ${region.die} ${region.work}`;
      const tag = makeZoneLabel(
        region.note ? [title, region.note].join(String.fromCharCode(10))
                    : title, scale * 0.042);
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

  const goToView = (dir: [number, number, number], label?: string) => {
    viewApi.current?.frame(new THREE.Vector3(...dir));
    if (label) setViewName(label);
  };

  const saveImage = () => {
    const url = viewApi.current?.snapshot(2);
    if (!url) return;
    const link = document.createElement('a');
    link.href = url;
    link.download = `${mesh.summary.name || 'part'}_${viewName}.png`;
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

  if (error) return <div className="cad-viewer__error">
    <span>{error}</span>
    <button type="button" onClick={() => {
      setError(null);
      setRendererRetry((value) => value + 1);
    }}>3D 화면 다시 시작</button>
  </div>;

  /* 제로라인 기준으로 공정 구역을 저절로 잡는다.
   *
   * [왜 이게 되는가]
   * 제로라인은 보정량의 **부호가 뒤집히는 자리**다. 그 선을 넘어가면
   * 할 일이 바뀐다 — 안쪽(+)은 살을 붙이고(용접), 바깥쪽(-)은 깎는다
   * (CNC 가공). 그러니 부호가 같은 포인트끼리 묶으면 그 덩어리가 곧
   * 한 공정 구역이고, 구역의 경계는 제로라인이 된다. 손으로 칠하는
   * 것과 달리 근거가 보정값 자체라 사람마다 달라지지 않는다.
   *
   * [묶는 규칙]
   * 부호가 같고 서로 가까운 것끼리 잇는다(단일 연결). 거리 기준은
   * 부품 크기의 12% — 이보다 좁히면 포인트마다 구역이 하나씩 생기고,
   * 넓히면 제로라인 건너편까지 한 구역으로 삼킨다. 부호가 다른 것은
   * 애초에 서로 안 묶이므로 제로라인을 넘지 않는다. */
  const zonesFromZeroLine = () => {
    const points = (overlay?.points ?? [])
      .map((p) => ({ at: p.position, value: sheetValues?.[p.id] }))
      .filter((p): p is { at: [number, number, number]; value: number } =>
        typeof p.value === 'number' && Math.abs(p.value) >= 0.05);
    if (!points.length || !onRegionsChange) return;

    // 부품 크기를 못 읽었으면 잡지 않는다. 100mm 로 갈음하면 큰 부품에서
    // 포인트마다 구역이 하나씩 생겨 화면이 못 쓰게 된다.
    const scale = viewApi.current?.radius ?? 0;
    if (scale <= 0) return;
    const near = scale * 0.12;
    const nearSq = near * near;
    const gap = (a: number[], b: number[]) =>
      (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;

    const made: CadRegion[] = [];
    for (const sign of [1, -1]) {
      const mine = points.filter((p) => Math.sign(p.value) === sign);
      const taken = new Array(mine.length).fill(false);
      for (let seed = 0; seed < mine.length; seed += 1) {
        if (taken[seed]) continue;
        // 씨앗에서 이웃을 타고 번져 나간다 — 같은 부호 안에서만.
        const group = [seed];
        taken[seed] = true;
        for (let cursor = 0; cursor < group.length; cursor += 1) {
          for (let other = 0; other < mine.length; other += 1) {
            if (taken[other]) continue;
            if (gap(mine[group[cursor]].at, mine[other].at) > nearSq) continue;
            taken[other] = true;
            group.push(other);
          }
        }
        if (group.length < 2) continue;   // 외톨이 하나는 구역이라 하지 않는다
        const worst = Math.max(...group.map((k) => Math.abs(mine[k].value)));
        made.push({
          id: `ZA-${sign > 0 ? 'W' : 'C'}${made.length}-${Date.now().toString(36)}`,
          auto: true,
          // 붙일 자리는 용접, 깎을 자리는 가공. 금형은 사람이 고른다.
          die: '하형',
          work: sign > 0 ? '용접' : '가공',
          note: `제로라인 기준 자동 · 포인트 ${group.length}개 · `
                + `최대 ${(sign * worst).toFixed(2)}mm`,
          stamps: group.map((k) => ({ at: mine[k].at, radius: near * 0.55 })),
        });
      }
    }
    // 손으로 그린 구역은 남기고 자동 구역만 갈아 끼운다.
    const byHand = (regions ?? []).filter((r) => !r.auto);
    onRegionsChange([...byHand, ...made]);
    setZoning(true);
  };

  const sheetCount = overlay?.points
    ?.filter((p) => typeof sheetValues?.[p.id] === 'number').length ?? 0;
  const overlayPoints = overlay?.points?.filter((p) => {
    const value = sheetValues?.[p.id];
    return typeof value === 'number'
      && Math.abs(value) >= threshold && Math.abs(value) <= ceiling;
  }).length ?? 0;

  return <>
    <div ref={mountRef}
      className={`cad-viewer__stage${light ? ' is-light' : ''}`} />

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

    {/* 좌표축 — 지금 어느 쪽에서 보고 있는지 늘 보이게 한다.
        축이 화면을 향해 다가오면(z>0) 점으로 줄어드는데, 그때는 흐리게
        해서 "이 축은 화면 쪽을 보고 있다" 를 알 수 있게 한다. */}
    <div className="cad-viewer__axis" aria-hidden="true"
         title={`정면·배면은 X(전후), 좌측·우측은 Y(좌우), 평면·저면은 Z(높이) 축을 따라 봅니다`}>
      <svg viewBox="-30 -30 60 60">
        <circle cx="0" cy="0" r="27" className="cad-viewer__axis-ring" />
        {axisView.map((dir, k) => {
          const tip = { x: dir[0] * 21, y: dir[1] * 21 };
          const paint = ['#ff6b6b', '#5fd38d', '#6fb4e8'][k];
          return <g key={k} opacity={dir[2] > 0.65 ? 0.4 : 1}>
            <line x1="0" y1="0" x2={tip.x} y2={tip.y} stroke={paint} strokeWidth="2" />
            <text x={tip.x * 1.28} y={tip.y * 1.28} fill={paint}
              textAnchor="middle" dominantBaseline="central"
              fontSize="11">{'XYZ'[k]}</text>
          </g>;
        })}
      </svg>
      <small>{AXIS_MEANING.join(' · ')}</small>
    </div>

    <div className="cad-viewer__views" role="group" aria-label="표준 뷰">
      {VIEWS.map((view) => (
        <button key={view.id} type="button"
          className={viewName === view.label ? 'is-on' : undefined}
          onClick={() => goToView(view.dir, view.label)}>
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
      {/* 시트에 실을 그림에서 반대편 살이나 옆 부품이 겹쳐 보일 때
          네모·동그라미로 훑어 감춘다. 형상을 지우는 것이 아니라 이
          화면에서만 안 그리는 것이라 언제든 되돌릴 수 있다. */}
      <button type="button" className={hiding ? 'is-on' : undefined}
        title="보고 싶지 않은 자리를 네모·동그라미로 훑어 감춥니다 (형상은 그대로)"
        onClick={() => {
          setHiding((v) => {
            if (!v) { setZoning(false); setMeasuring(false); setNoting(false); }
            return !v;
          });
        }}>
        가리기 {hides.length ? hides.length : ''}
      </button>
      {hides.length > 0 && (
        <button type="button" title="감춘 자리를 모두 되살립니다"
          onClick={() => setHideByCad((current) => {
            const next = { ...current };
            delete next[tintKey];
            return next;
          })}>
          가린 것 되돌리기
        </button>
      )}
      <button type="button" onClick={() => setLight((v) => !v)}
        title={light
          ? '어두운 배경으로 바꿉니다 (히트맵 색이 잘 읽힙니다)'
          : '흰 배경으로 바꿉니다 (시트에 실을 그림용)'}>
        {light ? '흰 바탕' : '검은 바탕'}
      </button>
      {/* 부품 색. 기본은 CATIA 가 STEP 에 넣어 둔 색이고, 파일이 색을
          안 들고 있거나 CATIA 에서 보던 것과 다르면 손으로 정한다. */}
      <label className="cad-viewer__tint"
        title={mesh.colour
          ? `CAD 에서 읽은 색 ${(mesh.colourGroups ?? []).map(
              ([tone, , count]) => `${tone} ${count.toLocaleString()}삼각형`
            ).join(' · ') || mesh.colour}`
          : '이 CAD 에는 색이 없습니다 — 손으로 정하세요'}>
        <input type="color" aria-label="부품 색"
          value={partTint ?? mesh.colour ?? '#8FA3B4'}
          onChange={(event) => setPartTint(event.target.value)} />
        {/* CAD 색이 몇 가지로 들어왔는지 눈으로 보이게 한다 — 안 읽혔으면
            여기가 비어 CAD 파일을 다시 열어야 한다는 걸 알 수 있다. */}
        {partTint ? '내가 정한 색'
          : (mesh.colourGroups?.length ?? 0) > 1
            ? `CAD 색 ${mesh.colourGroups!.length}가지`
            : mesh.colour ? 'CAD 색' : '색 없음'}
      </label>
      {(mesh.colourGroups?.length ?? 0) > 1 && !partTint && (
        <span className="cad-viewer__swatches" aria-hidden="true">
          {mesh.colourGroups!.map(([tone], k) => (
            <i key={`${tone}-${k}`} style={{ background: tone }} />
          ))}
        </span>
      )}
      {partTint && (
        <button type="button" onClick={() => setPartTint(null)}
          title="CAD 에 들어 있는 색으로 되돌립니다">CAD 색으로</button>
      )}
      <button type="button" onClick={saveImage} title="보이는 그대로 PNG 로 저장">
        저장
      </button>
      {onCapture && <button type="button"
        title={`지금 보이는 ${viewName} 화면을 인쇄 해상도로 시트에 담습니다`}
        onClick={() => {
          // 시트에 실을 그림만 2배로 굽는다. 화면용 해상도로 실으면
          // A4 인쇄에서 콜아웃 숫자가 뭉개진다.
          const url = viewApi.current?.snapshot(2);
          if (url) onCapture(url, viewName);
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
      {paintNote && <span className="cad-viewer__stat">{paintNote}</span>}
      <span className="cad-viewer__stat">
        삼각형 {mesh.summary.n_faces.toLocaleString()} · {holeLabel}
        {mesh.counts?.cylinders
          ? ` · 굽힘 R ${mesh.counts.cylinders - holes.length}`
          : ''}
      </span>
      <span className="cad-viewer__stat cad-viewer__hint">
        {hiding ? '형상 위에서 끌어 그 자리를 감춥니다 — 형상은 그대로입니다'
          : zoning ? (zoneTool === 'brush'
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
        {/* 제로라인이 곧 공정의 경계다 — 손으로 칠하지 않고 보정값 부호로
            구역을 잡는다. 손으로 그린 구역은 그대로 남는다. */}
        <div className="cad-viewer__zone-tools cad-viewer__zone-auto">
          <button type="button" onClick={zonesFromZeroLine}
            disabled={!sheetCount}
            title={sheetCount
              ? '보정량 부호가 바뀌는 자리(제로라인)를 경계로 삼아 + 는 용접, - 는 가공 구역으로 잡습니다'
              : '먼저 스캔을 골라 보정량을 올리세요'}>
            제로라인 기준 자동
          </button>
          {(regions ?? []).some((r) => r.auto) && (
            <button type="button"
              title="자동으로 잡은 구역만 지웁니다 (손으로 그린 것은 남습니다)"
              onClick={() => onRegionsChange?.(
                (regions ?? []).filter((r) => !r.auto))}>
              자동 구역 지우기
            </button>
          )}
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

    {/* 검사 원본에서 온 포인트는 맞춘 것이 아니라 부품 좌표 그대로다.
        얹힘을 말하는 것이 뜻이 없으므로 표면까지 실제 거리를 보인다. */}
    {overlay?.source === 'workspace' && (
      <p className="cad-viewer__warn cad-viewer__warn--ok">
        검사 원본에서 보정 포인트 {overlay.points.length}개를 그대로
        가져왔습니다{overlay.sourceName ? ` (${overlay.sourceName})` : ''} —
        판독도 정합도 하지 않았습니다.
        {overlay.surfaceGap && <> 형상 표면까지 중앙{' '}
          <b>{overlay.surfaceGap.median.toFixed(2)}mm</b> ·
          최대 {overlay.surfaceGap.max.toFixed(2)}mm 로 앉았습니다.</>}
        {' '}제로라인은 검사 원본에 없으므로 스캔 분석이나 시트 단면
        표기로 따로 얻어야 합니다.
      </p>
    )}

    {overlay && overlay.source !== 'workspace' && !overlay.fit.reliable && (
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
