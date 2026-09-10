'use client';

/* Blob/data URLs are local engine outputs and cannot use Next's remote image loader. */
/* eslint-disable @next/next/no-img-element */

import {
  Activity, AlertTriangle, ArrowLeft, ArrowRight, ArrowUpRight, BarChart3, Check, CheckCircle2, ChevronDown, ChevronRight,
  Circle, CircleHelp, Copy, Crosshair, Database, Eye, EyeOff, File, FileSpreadsheet, Files, Folder, FolderOpen, Gauge, HardDrive, Image as ImageIcon,
  Layers3, ListFilter, Maximize2, MousePointer2, Move, MoveRight, PanelLeftClose, Play, RefreshCw, Settings2,
  Printer, Server, ShieldCheck, Sparkles, Square, Trash2, Type, UploadCloud, X, ZoomIn, ZoomOut,
} from 'lucide-react';
import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { clearSession, downloadSession, emptySession, loadSession, readSessionFile, saveSession, type SessionSnapshot } from './session-store';
import { CIRCLED, DIE_CHOICES, WORK_CHOICES, CadViewer, type CadMesh, type CadNote, type CadOverlay, type CadRegion } from './cad-viewer';
import { WorkspaceHub, WorkspaceNavigation, type WorkspaceView } from './immersive-workspace';

/* 엔진 서버 주소.
 *
 * 127.0.0.1 로 못 박아 두면 다른 PC 에서 열었을 때 그 PC 자신을 찾아가
 * 화면만 뜨고 아무것도 안 된다. 브라우저가 지금 접속한 주소를 그대로 쓰고
 * 포트만 8000 으로 바꾼다. */
const API_BASE = typeof window === 'undefined'
  ? 'http://127.0.0.1:8000'
  : `http://${window.location.hostname}:8000`;

type View = WorkspaceView;
type Engine = 'label' | 'deviation' | 'zero';
type AnalysisStep = 'scan' | Engine;
type ScanStatus = 'ready' | 'analyzing' | 'done' | 'error';
/* source 가 'colormap' 이면 작업자가 찍은 추정 포인트다. 라벨을 읽어 얻은 실측값과
   섞이지 않도록 화면에서도 구분해 보여준다. */
/* xProduct/yProduct 는 같은 포인트를 제품데이터 이미지 기준 %로 다시 적은 값이다.
   정렬에 실패했거나 제품데이터가 없으면 비어 있다. */
type PointResult = { id: string; xPx: number; yPx: number; x: number; y: number; value: number; labelColor: string; confidence: string; source?: 'colormap'; xProduct?: number; yProduct?: number; keyReasons?: string[] };
type KeySelection = { ids: string[]; total: number; selected: number; peaks: number; signChanges: number; extremes: number };
type ZeroAnchor = { anchor_id: number; x: number; y: number; boundary_arclen: number; source?: string; kind?: 'point' | 'zone'; strength?: number };
type AdvanceLine = { points: [number, number][]; warnings: string[]; confidence: 'high' | 'low' };
type ZeroLineCandidate = { rank: number; anchor_start_id: number; anchor_end_id: number; points: [number, number][]; length_px: number; mean_abs_deviation: number; separation: number; balance: number; score: number };
type ZeroPointCluster = { cluster_id: number; loop: string; kind: 'point' | 'zone'; center: [number, number]; members: [number, number][]; contour: [number, number][]; strength: number; span: number };
type GreenBelt = { belt_id: number; contour: [number, number][]; center: [number, number]; length_px: number; area_px: number; mean_abs_deviation: number };
type SimpleZeroLine = { line_id: number; points: [number, number][]; route_type: string; bend_count: number; combined_coverage: number; tolerance_coverage: number; product_coverage: number; support_count: number; length_px: number };
type ZeroLineResult = { id: number | string; points: [number, number][] };
type LabShape = { shape_id: number; points: [number, number][]; is_closed: boolean };
type LabDistance = { to_lab_pct: number; to_predicted_pct: number; diagonal_px: number };
type LabelZeroLine = { points: [number, number][]; length_px: number; mean_abs_deviation: number };
type ReferenceLine = { kind: 'line' | 'areas'; points: [number, number][]; contours: [number, number][][]; partNo: string; sourceSheet: string; mirrored: boolean };
/* 스캔을 제품데이터 위로 옮기는 변환. margin 은 1위와 2위 방향의 점수 차이고,
   대칭 부품은 이 값이 0에 가까워 사람이 방향을 정해 줘야 한다. */
type AlignmentInfo = { matrix: number[]; flipX: boolean; flipY: boolean; rotation?: number; outlineIou: number; holeIou: number; bandIou: number; score: number; margin: number; confident: boolean; overridden: boolean; scanSize: number[]; productSize: number[]; candidates?: { flipX: boolean; flipY: boolean; rotation?: number; score: number }[]; warnings: string[] };
const mapAffinePoint = (matrix: number[], x: number, y: number): [number, number] => {
  const [a, b, tx, c, d, ty] = matrix;
  return [a * x + b * y + tx, c * x + d * y + ty];
};
const invertAffinePoint = (matrix: number[], x: number, y: number): [number, number] | null => {
  const [a, b, tx, c, d, ty] = matrix;
  const determinant = a * d - b * c;
  if (Math.abs(determinant) < 1e-10) return null;
  const px = x - tx; const py = y - ty;
  return [(d * px - b * py) / determinant, (-c * px + a * py) / determinant];
};
const invertAffineDelta = (matrix: number[], dx: number, dy: number): [number, number] | null => {
  const [a, b, , c, d] = matrix;
  const determinant = a * d - b * c;
  return Math.abs(determinant) < 1e-10 ? null : [(d * dx - b * dy) / determinant, (-c * dx + a * dy) / determinant];
};
type AnalysisResult = {
  analysisId: string | null;
  partNo?: string;
  knownParts?: string[];
  /* 현업 파일명 규칙에서 읽어낸 것들 — 차종_품번_품명_공정_날짜.
     보정시트 머리말을 이걸로 채운다. 못 읽은 칸은 null 이다. */
  naming?: {
    part_no: string | null; maker: string | null; part_name: string | null;
    process: string | null; applied_at: string | null; control_no: string | null;
  };
  /* 현업 파이프라인(lab_pipeline)이 만든 제로라인. 허용범위 밖 영역을
     윤곽 위 제로포인트 둘로 닫은 선이라 근거가 가장 분명하다. */
  labZeroLines?: [number, number][][];
  /* 제로 **영역**(67XX6). 서버가 이미 네모로 다듬어 보낸다 —
     시트와 3D 가 같은 도형을 그리려고 한 군데서 만든다. */
  labZeroAreas?: [number, number][][];
  labZeroRegions?: { label: string; area: number; status: string;
                     zeroPoints: string[]; attempts: number; coverage: number }[];
  source: { name: string; width: number; height: number };
  partNumber: string | null;
  cleanImage: string | null;
  productImage: string | null;
  productSource: string | null;
  alignment: AlignmentInfo | null;
  alignmentOverlay: string | null;
  zeroOverlay: string | null;
  zeroCase?: number | null;
  zeroMask: string | null;
  /* 스캔 좌표(픽셀) 기반 제로 폴리라인. id는 현재 UI의 개별 표시/숨김 제어에,
     points는 제품데이터 좌표 변환과 편집 오버레이에 함께 사용한다. */
  zeroLines?: ZeroLineResult[];
  zeroAnchors: ZeroAnchor[];
  advanceLine: AdvanceLine | null;
  zeroLineCandidates: ZeroLineCandidate[];
  zeroPointClusters: ZeroPointCluster[];
  greenBelts: GreenBelt[];
  simpleZeroLines: SimpleZeroLine[];
  labProfile: LabShape[];
  labDistance: LabDistance | null;
  labelZeroLine: LabelZeroLine | null;
  referenceLine: ReferenceLine | null;
  keySelection?: KeySelection;
  points: PointResult[];
  stats: {
    labelsRemoved: number;
    pointsDetected: number;
    detectedCandidates?: number;
    validCandidates?: number;
    qwenReads: number;
    qwenUnread?: number;
    pointsTransferred?: number;
    zeroRegions: number;
    zeroRatio: number;
    zeroTolerance: number | null;
  };
  warnings: string[];
  warningsByEngine?: Partial<Record<Engine | 'product', string[]>>;
  errors: Partial<Record<Engine | 'product', string>>;
  valueMode: string;
};
type ScanItem = { id: string; name: string; partNo: string; size: string; url: string; file: File; status: ScanStatus; tone: number; result?: AnalysisResult; error?: string; productFile?: File; productUrl?: string; cadFiles?: File[]; cadUploading?: boolean; assetError?: string; assetStatus?: string };
type FitAdjust = { angle: number; dx: number; dy: number; scale: number };
type ZeroPointOffset = { dx: number; dy: number };
type ZeroEdit = { index: number; dx: number; dy: number; hidden?: boolean; points?: Record<string, ZeroPointOffset>; vertices?: [number, number][]; spline?: boolean; splineSegments?: number[] };
/* 아직 한 번도 "3D에 적용"을 안 누른 스캔은 zeroEditsByScan[scan.id]가
   없어서 Home 이 매번 `... || []`로 새 빈 배열을 만들어 내려보낸다. 이걸
   그대로 ServicePreview의 draftZeroEdits 동기화 이펙트(의존성 배열에
   zeroEdits가 있음) 에 물리면, Home 이 재렌더될 때마다(제로라인 드래그와
   무관한 이유라도) 참조가 바뀐 걸로 보여 그 이펙트가 다시 돌아 드래그
   중이던 draftZeroEdits를 빈 배열로 되돌려 버린다 — "옮기다 보면 도로
   초기화된다" 증상. 매번 같은 참조를 주면 이 이펙트가 헛돌지 않는다. */
const EMPTY_ZERO_EDITS: ZeroEdit[] = [];
const NO_ADJUST: FitAdjust = { angle: 0, dx: 0, dy: 0, scale: 1 };

function partOfCad(mesh: CadMesh | null | undefined): string {
  const name = (mesh?.summary.name || '').toUpperCase().replace(/[-_]/g, '');
  const pairs: [string, string][] = [['64XX1', '64XX2'], ['71XX1', '71XX2'], ['67XX6', '67XX6']];
  return pairs.find(([cad]) => name.includes(cad))?.[1] || '';
}

function editableZeroLineCount(result: AnalysisResult): number {
  /* 현재 하이브리드 엔진의 실제 응답은 zeroLines에 들어온다. 이전 엔진
     필드만 세면 선이 화면에 보여도 수정 버튼이 비활성화된다. */
  return result.zeroLines?.filter((line) => Array.isArray(line.points) && line.points.length >= 2).length
    || result.labZeroLines?.length
    || result.simpleZeroLines?.length
    || 0;
}

const distanceToSegment = (px: number, py: number, [ax, ay]: [number, number], [bx, by]: [number, number]) => {
  const dx = bx - ax; const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared)) : 0;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
};

function pointInsideClosedLine(px: number, py: number, line: [number, number][]): boolean {
  if (line.length < 4 || Math.hypot(line[0][0] - line[line.length - 1][0], line[0][1] - line[line.length - 1][1]) > 3) return false;
  let inside = false;
  for (let index = 0, previous = line.length - 1; index < line.length; previous = index, index += 1) {
    const [x, y] = line[index]; const [oldX, oldY] = line[previous];
    if ((y > py) !== (oldY > py) && px < (oldX - x) * (py - y) / (oldY - y) + x) inside = !inside;
  }
  return inside;
}

function pointTouchesZeroLine(point: PointResult, lines: [number, number][][], width: number, height: number): boolean {
  /* 보정시트에서는 제로영역 안의 값과 선에 사실상 붙어 있는 값을 작업
     지시점으로 취급하지 않는다. 해상도 차이를 흡수하도록 대각선의 0.8%를
     근접 기준으로 쓰되 작은 이미지에서도 최소 6px은 확보한다. */
  const threshold = Math.max(6, Math.hypot(width, height) * 0.008);
  return lines.some((line) => pointInsideClosedLine(point.xPx, point.yPx, line)
    || line.slice(1).some((end, index) => distanceToSegment(point.xPx, point.yPx, line[index], end) <= threshold));
}
type FolderEntry = { name: string; path: string; isDirectory: boolean; size: number | null; modified: string };
type CorrectionMode = 'auto' | 'manual';
type CorrectionAction = 'edit' | 'reset_auto' | 'reset_all' | 'restore_before' | 'reapply' | 'revise';
type CorrectionHistoryEntry = {
  id: number;
  partNo: string;
  scanName: string;
  pointId: string;
  oldValue: number | null;
  newValue: number | null;
  oldMode: CorrectionMode | null;
  newMode: CorrectionMode | null;
  coefficient: number | null;
  action: CorrectionAction;
  sourceEntryId: number | null;
  worker: string | null;
  createdAt: string;
};
type FolderResponse = { available?: boolean; rootName?: string; path?: string; entries?: FolderEntry[]; error?: string };
type HealthResponse = { ok?: boolean };
type FileDatabaseStatus = { configured: boolean; label: string; connected: boolean | null; catalogCount: number; operationCount: number; version?: string; error?: string };
type FileOrganizerStatus = { sourceRoot: string; destinationRoot: string; sourceAvailable: boolean; destinationAvailable: boolean; database: FileDatabaseStatus };
type FileOrganizerItem = { id: string; name: string; sourcePath: string; sourceKind: 'source' | 'upload'; size: number; modified: string; customer: string; itemNo: string; family: string; productName: string; process: string; categoryKey: string; categoryLabel: string; confidence: number; reasons: string[]; targetDir: string; targetPath: string; matchedProductFolder: string; detailPath: string };
type FolderAxis = 'item' | 'vehicle' | 'category' | 'detail';
type FolderAxisOption = { id: FolderAxis; label: string };
type FolderOrderResponse = { folderOrder: FolderAxis[]; axes: FolderAxisOption[]; error?: string };
const FOLDER_AXES: FolderAxis[] = ['item', 'vehicle', 'category', 'detail'];
const FOLDER_AXIS_LABELS: Record<FolderAxis, string> = { item: '품번', vehicle: '차종', category: '카테고리', detail: '세부 하위폴더' };
type OrganizerPathsResponse = { sourceRoot: string; destinationRoot: string; sourceLocked: boolean; destinationLocked: boolean };
type UploadStorageTarget = { path: string; status: 'success' | 'skipped' | 'error'; message?: string | null };
type UploadStorageResult = { name: string; sourcePath: string; existing: UploadStorageTarget; reconstructed: UploadStorageTarget };
type SheetTitleField = 'heading' | 'managementLabel' | 'managementNo' | 'partNameLabel' | 'partName' | 'processLabel' | 'process' | 'partNoLabel' | 'partNo' | 'materialLabel' | 'material' | 'appliedDateLabel' | 'appliedDate';
type SheetTitleValues = Record<SheetTitleField, string>;
type SheetTitleFonts = Record<SheetTitleField, string>;
type SheetTitleFontSizes = Partial<Record<SheetTitleField, number>>;

/* 보정 시트 주석 — 좌표와 크기는 모두 이미지 대비 %라 확대/축소와 창 크기에 영향받지 않는다. */
type AnnotationKind = 'rect' | 'ellipse' | 'text' | 'arrow';
type AnnotationTool = 'select' | AnnotationKind;
/* 사각형·타원·텍스트는 x,y 가 좌상단이고 w,h 가 크기다. 화살표는 x,y 가 시작점이고 w,h 가 끝점까지의 변위라 음수가 될 수 있다. */
type Annotation = { id: string; kind: AnnotationKind; x: number; y: number; w: number; h: number; text?: string; fontSize?: number; fontFamily?: string; color?: string };
type DetailRegion = { id: string; x: number; y: number; w: number; h: number; label: string };
type SheetLayout = { id: string; kind: 'front' | 'detail'; x: number; y: number; w: number; h: number; regionId?: string };
type SheetRotation = 0 | 90 | 180 | 270;
type SheetImageTransform = { rotation: SheetRotation; flipX: boolean; flipY: boolean };

const IDENTITY_SHEET_TRANSFORM: SheetImageTransform = { rotation: 0, flipX: false, flipY: false };

function sheetTransformKey(transform: SheetImageTransform) {
  return `${transform.rotation}-${transform.flipX ? 1 : 0}-${transform.flipY ? 1 : 0}`;
}

function transformSheetPoint(transform: SheetImageTransform, x: number, y: number): [number, number] {
  const flippedX = transform.flipX ? 100 - x : x;
  const flippedY = transform.flipY ? 100 - y : y;
  if (transform.rotation === 90) return [100 - flippedY, flippedX];
  if (transform.rotation === 180) return [100 - flippedX, 100 - flippedY];
  if (transform.rotation === 270) return [flippedY, 100 - flippedX];
  return [flippedX, flippedY];
}

function invertSheetPoint(transform: SheetImageTransform, x: number, y: number): [number, number] {
  let rotatedX = x; let rotatedY = y;
  if (transform.rotation === 90) [rotatedX, rotatedY] = [y, 100 - x];
  else if (transform.rotation === 180) [rotatedX, rotatedY] = [100 - x, 100 - y];
  else if (transform.rotation === 270) [rotatedX, rotatedY] = [100 - y, x];
  return [transform.flipX ? 100 - rotatedX : rotatedX, transform.flipY ? 100 - rotatedY : rotatedY];
}

function invertSheetDelta(transform: SheetImageTransform, dx: number, dy: number): [number, number] {
  const origin = invertSheetPoint(transform, 50, 50);
  const moved = invertSheetPoint(transform, 50 + dx, 50 + dy);
  return [moved[0] - origin[0], moved[1] - origin[1]];
}

/* rotateVector/unrotateVector: transformSheetPoint 와 달리 "위치"가 아니라
   "벡터"(방향+길이)를 돌린다 -- 회전·반전은 크기를 바꾸지 않는 등거리
   변환이라, 입력 벡터의 길이(sqrt(dx²+dy²))가 그대로 보존된다. 라벨을
   점에서 얼마나 떼어 놓을지를 이 벡터로 다루면, 회전해도 지시선 길이가
   화면 비율(가로/세로 실제 픽셀 배율)에 흔들리지 않고 항상 같게 나온다. */
function rotateVector(transform: SheetImageTransform, dx: number, dy: number): [number, number] {
  const fx = transform.flipX ? -dx : dx;
  const fy = transform.flipY ? -dy : dy;
  if (transform.rotation === 90) return [-fy, fx];
  if (transform.rotation === 180) return [-fx, -fy];
  if (transform.rotation === 270) return [fy, -fx];
  return [fx, fy];
}

function unrotateVector(transform: SheetImageTransform, dx: number, dy: number): [number, number] {
  let x = dx; let y = dy;
  if (transform.rotation === 90) [x, y] = [y, -x];
  else if (transform.rotation === 180) [x, y] = [-x, -y];
  else if (transform.rotation === 270) [x, y] = [-y, x];
  return [transform.flipX ? -x : x, transform.flipY ? -y : y];
}

function renderSheetImage(source: string, transform: SheetImageTransform): Promise<string> {
  if (sheetTransformKey(transform) === sheetTransformKey(IDENTITY_SHEET_TRANSFORM)) return Promise.resolve(source);
  return new Promise((resolve, reject) => {
    const image = new window.Image();
    image.onload = () => {
      const quarterTurn = transform.rotation === 90 || transform.rotation === 270;
      const canvas = document.createElement('canvas');
      canvas.width = quarterTurn ? image.naturalHeight : image.naturalWidth;
      canvas.height = quarterTurn ? image.naturalWidth : image.naturalHeight;
      const context = canvas.getContext('2d');
      if (!context) { reject(new Error('이미지를 회전할 수 없습니다.')); return; }
      context.translate(canvas.width / 2, canvas.height / 2);
      context.rotate(transform.rotation * Math.PI / 180);
      context.scale(transform.flipX ? -1 : 1, transform.flipY ? -1 : 1);
      context.drawImage(image, -image.naturalWidth / 2, -image.naturalHeight / 2);
      resolve(canvas.toDataURL('image/png'));
    };
    image.onerror = () => reject(new Error('보정시트 이미지를 불러오지 못했습니다.'));
    image.src = source;
  });
}

function useTransformedSheetImage(source: string, requested: SheetImageTransform) {
  const [rendered, setRendered] = useState({ url: source, transform: IDENTITY_SHEET_TRANSFORM, busy: false, error: null as string | null });
  useEffect(() => {
    let cancelled = false;
    setRendered((current) => ({ ...current, busy: true, error: null }));
    void renderSheetImage(source, requested).then((url) => {
      if (!cancelled) setRendered({ url, transform: requested, busy: false, error: null });
    }).catch((error: unknown) => {
      if (!cancelled) setRendered({ url: source, transform: IDENTITY_SHEET_TRANSFORM, busy: false, error: error instanceof Error ? error.message : '이미지 방향을 바꾸지 못했습니다.' });
    });
    return () => { cancelled = true; };
  }, [source, requested.rotation, requested.flipX, requested.flipY]);
  return rendered;
}

const engineMeta: Record<Engine, { name: string; short: string; color: string }> = {
  label: { name: '라벨 제거 및 복원', short: 'label_removal', color: '#7058e8' },
  deviation: { name: '스캔 포인트 추출', short: 'deviation_extraction', color: '#ee6b3c' },
  zero: { name: '추천 제로라인', short: 'zero_line_detection', color: '#17a58b' },
};

const analysisStepMeta: { key: AnalysisStep; name: string; short: string; color: string }[] = [
  { key: 'scan', name: '스캔 데이터', short: 'scan_data', color: '#3b75c3' },
  ...(Object.keys(engineMeta) as Engine[]).map((key) => ({ key, ...engineMeta[key] })),
];

/* 보정 시트에서 쓰는 글꼴들. 아진산업 실제 양식 기준 — 돋움·맑은 고딕은 이 PC에도 설치돼 있지만
   휴먼옛체·현대하모니는 별도 설치가 필요한 사내 서체라, 이름만 걸어두고 설치된 PC에서 자동 적용되게 한다. */
const FONT_HUMAN_OLD = "'휴먼옛체', serif";
const FONT_MALGUN = "'Malgun Gothic', sans-serif";
const FONT_DOTUM = "Dotum, sans-serif";
const FONT_HARMONY_M = "'현대하모니 M', sans-serif";
const FONT_HARMONY_L = "'현대하모니 L', sans-serif";
const FONT_FAMILY_OPTIONS: { label: string; value: string }[] = [
  { label: '기본 글꼴', value: '' },
  { label: '휴먼옛체', value: FONT_HUMAN_OLD },
  { label: '맑은 고딕', value: FONT_MALGUN },
  { label: '돋움', value: FONT_DOTUM },
  { label: '현대하모니 M', value: FONT_HARMONY_M },
  { label: '현대하모니 L', value: FONT_HARMONY_L },
];
const DEFAULT_TITLE_FONTS: SheetTitleFonts = {
  heading: FONT_HUMAN_OLD,
  managementLabel: FONT_DOTUM, managementNo: FONT_MALGUN,
  partNameLabel: FONT_DOTUM, partName: FONT_MALGUN,
  processLabel: FONT_DOTUM, process: FONT_MALGUN,
  partNoLabel: FONT_DOTUM, partNo: FONT_MALGUN,
  materialLabel: FONT_DOTUM, material: FONT_MALGUN,
  appliedDateLabel: FONT_DOTUM, appliedDate: FONT_MALGUN,
};
const DEFAULT_POINT_LABEL_FONT = FONT_HARMONY_L;
const DEFAULT_ANNOTATION_TEXT_FONT = FONT_HARMONY_M;
const TITLE_FONT_SIZE_MIN = 6;
const TITLE_FONT_SIZE_MAX = 40;
const TITLE_FONT_SIZE_STEP = 1;
/* 글꼴 크기를 아직 아무도 바꾸지 않았을 때 도구막대 스테퍼가 보여줄 시작값 — 화면 기본 CSS 크기와 맞춘다. */
const TITLE_DEFAULT_FONT_SIZE: Record<SheetTitleField, number> = {
  heading: 22,
  managementLabel: 10, managementNo: 10,
  partNameLabel: 10, partName: 10,
  processLabel: 10, process: 10,
  partNoLabel: 10, partNo: 10,
  materialLabel: 10, material: 10,
  appliedDateLabel: 10, appliedDate: 10,
};

/* CSS font-family 문자열("'Malgun Gothic', sans-serif")에서 엑셀 셀 글꼴로 쓸 첫 글꼴 이름만 뽑는다. */
function extractFontName(cssFontFamily: string) {
  const first = cssFontFamily.split(',')[0]?.trim().replace(/^['"]|['"]$/g, '');
  return first || 'Malgun Gothic';
}

/* 파일명은 "관리 NO 보정내용" 형식으로 저장한다 (예: CD8 71XX2/22-XB000-01 → CD8 71XX2_22-XB000-01 보정내용.xlsx).
   윈도우 파일명에 못 쓰는 문자만 밑줄로 바꾸고, 나머지는 그대로 둔다. */
function excelFileName(managementNo: string) {
  const sanitized = managementNo.trim().replace(/[\\/:*?"<>|]/g, '_').replace(/\s+/g, ' ').trim();
  return `${sanitized || '보정 시트'} 보정내용.xlsx`;
}

/* 사용자 기준 엑셀 그림 크기. 1px=9525EMU, 1cm=360000EMU로 환산하면 Excel의
   그림 서식 창에서도 아래 cm 값이 그대로 표시된다. */
const EXCEL_SHEET_IMAGE_WIDTH_CM = 26.96;
const EXCEL_SHEET_IMAGE_HEIGHT_CM = 15.74;
const EXCEL_SHEET_IMAGE_INSET_CM = 0.03;
const EXCEL_SHEET_IMAGE_ASPECT = EXCEL_SHEET_IMAGE_WIDTH_CM / EXCEL_SHEET_IMAGE_HEIGHT_CM;
const excelCentimetersToPixels = (centimeters: number) => centimeters * 360000 / 9525;

/* 웹 시트(A4 비율)의 아래쪽 빈 공간만 잘라 엑셀 그림 비율로 만든다. 그림을 가로·세로로
   따로 늘이지 않으므로 보정치 글자와 흰 라벨의 비율이 웹 화면과 동일하게 유지된다. */
function cropCanvasToAspect(source: HTMLCanvasElement, targetAspect: number) {
  const sourceAspect = source.width / source.height;
  let sourceX = 0; let sourceY = 0; let sourceWidth = source.width; let sourceHeight = source.height;
  if (sourceAspect < targetAspect) {
    sourceHeight = Math.min(source.height, Math.round(source.width / targetAspect));
    /* 보정 도면은 시트 위쪽에 배치되므로 위를 고정하고 아래쪽 여백을 우선 잘라낸다. */
    sourceY = 0;
  } else if (sourceAspect > targetAspect) {
    sourceWidth = Math.min(source.width, Math.round(source.height * targetAspect));
    sourceX = Math.round((source.width - sourceWidth) / 2);
  }
  const cropped = document.createElement('canvas');
  cropped.width = sourceWidth;
  cropped.height = sourceHeight;
  const context = cropped.getContext('2d');
  if (!context) throw new Error('엑셀용 보정 시트 이미지를 만들지 못했습니다.');
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, cropped.width, cropped.height);
  context.drawImage(source, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, cropped.width, cropped.height);
  return cropped;
}

function formatBytes(value: number | null) {
  if (value == null) return '';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

/* 업로드 파일명이 그대로 PART NAME 기본값이 되는데, "3D 스캔" 류의 촬영 방식 표기까지
   부품명에 섞여 들어오는 경우가 많아 걷어낸다. */
function stripScanSuffix(name: string) {
  return name.replace(/[\s_-]*3d[\s_-]*스캔/gi, '').replace(/[\s_-]*3d[\s_-]*scan/gi, '').trim();
}

function createDefaultSheetTitleValues(scan: ScanItem): SheetTitleValues {
  return {
    heading: '보정 적용 내용',
    managementLabel: '관리 NO',
    managementNo: `ADC-${scan.partNo}`,
    partNameLabel: 'PART NAME',
    partName: stripScanSuffix(scan.name.replace(/\.[^.]+$/, '')),
    processLabel: '공정',
    process: '금형 보정',
    partNoLabel: 'PART NO',
    partNo: scan.partNo,
    materialLabel: '원소재',
    material: '3D SCAN DATA',
    appliedDateLabel: '적용일자',
    appliedDate: new Intl.DateTimeFormat('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' }).format(new Date()),
  };
}

function Heatmap({ imageUrl, width, height, children, lightBackground = false, containImage = false }: { imageUrl?: string | null; width: number; height: number; children?: React.ReactNode; lightBackground?: boolean; containImage?: boolean }) {
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [containedSize, setContainedSize] = useState<{ width: number; height: number } | null>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number; panX: number; panY: number } | null>(null);
  const clampPan = (next: { x: number; y: number }, nextScale: number) => {
    const bounds = viewportRef.current?.getBoundingClientRect();
    if (!bounds || nextScale <= 1) return { x: 0, y: 0 };
    const maxX = bounds.width * (nextScale - 1) / 2;
    const maxY = bounds.height * (nextScale - 1) / 2;
    return { x: Math.max(-maxX, Math.min(maxX, next.x)), y: Math.max(-maxY, Math.min(maxY, next.y)) };
  };
  const setZoom = (nextScale: number) => {
    const bounded = Math.max(1, Math.min(4, nextScale));
    setScale(bounded);
    setPan((current) => clampPan(current, bounded));
  };
  const resetView = () => { setScale(1); setPan({ x: 0, y: 0 }); };
  useEffect(() => {
    if (!containImage) { setContainedSize(null); return; }
    const viewport = viewportRef.current;
    if (!viewport || width <= 0 || height <= 0) return;
    const update = () => {
      const availableWidth = viewport.clientWidth;
      const availableHeight = viewport.clientHeight;
      if (!availableWidth || !availableHeight) return;
      const imageRatio = width / height;
      const viewportRatio = availableWidth / availableHeight;
      const next = viewportRatio > imageRatio
        ? { width: availableHeight * imageRatio, height: availableHeight }
        : { width: availableWidth, height: availableWidth / imageRatio };
      setContainedSize((current) => current && Math.abs(current.width - next.width) < 0.5 && Math.abs(current.height - next.height) < 0.5 ? current : next);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [containImage, width, height]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport || !imageUrl) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      const amount = event.deltaY < 0 ? 0.25 : -0.25;
      setScale((current) => {
        const next = Math.max(1, Math.min(4, current + amount));
        setPan((currentPan) => clampPan(currentPan, next));
        return next;
      });
    };
    viewport.addEventListener('wheel', handleWheel, { passive: false });
    return () => viewport.removeEventListener('wheel', handleWheel);
  }, [imageUrl]);

  return <div
    ref={viewportRef}
    className={`heatmap heatmap--actual ${lightBackground ? 'heatmap--light' : ''} ${children ? 'heatmap--annotated' : ''} ${scale > 1 ? 'heatmap--zoomed' : ''} ${dragging ? 'heatmap--dragging' : ''}`}
    onDoubleClick={(event) => { if ((event.target as Element).closest('.measure-point, .zoom-controls, .annotation-layer--armed, .annotation-shape, .annotation-handle, .annotation-delete, .annotation-fontsize, .annotation-arrow__hit')) return; resetView(); }}
    onPointerDown={(event) => {
      if (scale <= 1 || (event.target as Element).closest('.measure-point, .anchor-point, .zoom-controls, .annotation-layer--armed, .annotation-shape, .annotation-handle, .annotation-delete, .annotation-fontsize, .annotation-arrow__hit')) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
      setDragging(true);
    }}
    onPointerMove={(event) => {
      const start = dragRef.current;
      if (!start || start.pointerId !== event.pointerId) return;
      setPan(clampPan({ x: start.panX + event.clientX - start.x, y: start.panY + event.clientY - start.y }, scale));
    }}
    onPointerUp={(event) => {
      if (dragRef.current?.pointerId !== event.pointerId) return;
      dragRef.current = null; setDragging(false); event.currentTarget.releasePointerCapture(event.pointerId);
    }}
    onPointerCancel={() => { dragRef.current = null; setDragging(false); }}
  >
    {imageUrl && <div className="zoom-controls" aria-label="이미지 확대 및 이동 도구">
      <button type="button" onClick={() => setZoom(scale - 0.25)} disabled={scale <= 1} aria-label="축소" title="축소"><ZoomOut size={15} /></button>
      <span aria-live="polite">{Math.round(scale * 100)}%</span>
      <button type="button" onClick={() => setZoom(scale + 0.25)} disabled={scale >= 4} aria-label="확대" title="확대"><ZoomIn size={15} /></button>
      <button type="button" onClick={resetView} disabled={scale === 1 && pan.x === 0 && pan.y === 0} aria-label="화면 맞춤" title="화면 맞춤"><Maximize2 size={15} /></button>
    </div>}
    {imageUrl ? <div className="heatmap__media" style={{ aspectRatio: `${width} / ${height}`, ...(containImage && containedSize ? { width: `${containedSize.width}px`, height: `${containedSize.height}px` } : {}) }}>
      <div className="heatmap__transform" style={{ transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${scale})` }}>
        <img src={imageUrl} alt="엔진이 처리한 3D 스캔 편차 이미지" />
        {children}
      </div>
    </div> : <div className="heatmap__empty"><ImageIcon size={34} /><span>분석 결과 이미지가 없습니다.</span></div>}
  </div>;
}

function ZeroLineLayer({ lines, width, height }: { lines: ZeroLineResult[]; width: number; height: number }) {
  return <svg className="zero-line-result-layer" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
    {lines.map((line) => {
      const points = line.points.map(([x, y]) => `${x},${y}`).join(' ');
      return <g key={String(line.id)}>
        <polyline className="zero-line-result__outline" points={points} />
        <polyline className="zero-line-result__line" points={points} />
      </g>;
    })}
  </svg>;
}

const MIN_ANNOTATION_SIZE = 2;
const DEFAULT_ANNOTATION_SIZE: Record<AnnotationKind, { w: number; h: number }> = {
  rect: { w: 14, h: 10 }, ellipse: { w: 14, h: 12 }, text: { w: 16, h: 7 }, arrow: { w: 12, h: -8 },
};
/* 주석 색상. 도면에서 구분이 잘 되는 색만 골랐다. */
const ANNOTATION_COLORS = [
  { hex: '#e8802f', label: '주황' },
  { hex: '#d33f3f', label: '빨강' },
  { hex: '#2f7fd6', label: '파랑' },
  { hex: '#17a06f', label: '초록' },
  { hex: '#8b5cd6', label: '보라' },
  { hex: '#3d4550', label: '먹색' },
];
const DEFAULT_ANNOTATION_COLOR = ANNOTATION_COLORS[0].hex;

/* 받침이 있으면 '으로', 없거나 ㄹ 받침이면 '로'. (보라 → 보라로, 주황 → 주황으로) */
function withRo(word: string) {
  const code = word.charCodeAt(word.length - 1) - 0xac00;
  if (code < 0 || code > 11171) return `${word}로`;
  const jong = code % 28;
  return `${word}${jong === 0 || jong === 8 ? '' : '으'}로`;
}

/* 도형 채움은 같은 색을 옅게 깐다. CSS 만으로는 색을 반투명하게 못 만들어 여기서 계산한다. */
function withAlpha(hex: string, alpha: number) {
  const value = hex.replace('#', '');
  const full = value.length === 3 ? value.split('').map((c) => c + c).join('') : value;
  const int = parseInt(full, 16);
  if (!Number.isFinite(int)) return hex;
  return `rgba(${(int >> 16) & 255}, ${(int >> 8) & 255}, ${int & 255}, ${alpha})`;
}

/* 글자 크기는 이미지와 함께 확대·축소되도록 레이어 안의 px 로 다룬다. */
const DEFAULT_TEXT_SIZE = 10;
const TEXT_SIZE_MIN = 6;
const TEXT_SIZE_MAX = 48;
const TEXT_SIZE_STEP = 2;
const BOX_HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const;
const HANDLE_OFFSET: Record<string, { x: number; y: number }> = {
  nw: { x: 0, y: 0 }, n: { x: 0.5, y: 0 }, ne: { x: 1, y: 0 }, e: { x: 1, y: 0.5 },
  se: { x: 1, y: 1 }, s: { x: 0.5, y: 1 }, sw: { x: 0, y: 1 }, w: { x: 0, y: 0.5 },
};
type AnnotationHandle = (typeof BOX_HANDLES)[number] | 'start' | 'end';

function annotationHandleDescriptors(annotation: Annotation): { handle: AnnotationHandle; className: string; left: number; top: number }[] {
  if (annotation.kind === 'arrow') {
    return (['start', 'end'] as const).map((handle) => ({
      handle,
      className: 'annotation-handle annotation-handle--endpoint',
      left: handle === 'start' ? annotation.x : annotation.x + annotation.w,
      top: handle === 'start' ? annotation.y : annotation.y + annotation.h,
    }));
  }
  const left = Math.min(annotation.x, annotation.x + annotation.w);
  const top = Math.min(annotation.y, annotation.y + annotation.h);
  const width = Math.abs(annotation.w);
  const height = Math.abs(annotation.h);
  return BOX_HANDLES.map((handle) => ({
    handle,
    className: `annotation-handle annotation-handle--${handle}`,
    left: left + width * HANDLE_OFFSET[handle].x,
    top: top + height * HANDLE_OFFSET[handle].y,
  }));
}

let annotationSeq = 0;
const nextAnnotationId = () => `ann-${Date.now().toString(36)}-${++annotationSeq}`;
let detailSeq = 0;
const nextDetailId = () => `detail-${Date.now().toString(36)}-${++detailSeq}`;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

/* 포인터가 이미 놓였거나 취소된 경우 setPointerCapture 는 예외를 던진다.
   캡처는 커서가 도형 밖으로 나가도 이벤트를 계속 받기 위한 편의일 뿐이라,
   실패해도 조작 자체는 이어져야 한다. */
function capturePointer(element: Element, pointerId: number) {
  try { element.setPointerCapture(pointerId); } catch { /* 캡처 없이 진행 */ }
}

/* 드래그가 끝나면 음수 크기를 뒤집고 최소 크기를 보장한다. 화살표는 방향이 의미를 가지므로 그대로 둔다. */
function normalizeAnnotation(annotation: Annotation): Annotation {
  if (annotation.kind === 'arrow') return annotation;
  let { x, y, w, h } = annotation;
  if (w < 0) { x += w; w = -w; }
  if (h < 0) { y += h; h = -h; }
  w = Math.max(w, MIN_ANNOTATION_SIZE); h = Math.max(h, MIN_ANNOTATION_SIZE);
  return { ...annotation, x: clamp(x, 0, 100 - w), y: clamp(y, 0, 100 - h), w, h };
}

function AnnotationToolbar({ tool, setTool, hasAnnotations, onClearAll, selectedColor, onColorChange, detailMode, onDetailMode, labelAreaMode, onLabelAreaMode, addPointMode, onAddPointMode, zeroEditActive, zeroEditDisabled, onZeroEdit, keyPointsOnly, keyPointsDisabled, onKeyPointsOnlyChange }: { tool: AnnotationTool; setTool: (tool: AnnotationTool) => void; hasAnnotations: boolean; onClearAll: () => void; selectedColor: string | null; onColorChange: (hex: string) => void; detailMode?: boolean; onDetailMode?: () => void; labelAreaMode?: 'hide' | 'show' | null; onLabelAreaMode?: (mode: 'hide' | 'show') => void; addPointMode?: boolean; onAddPointMode?: () => void; zeroEditActive?: boolean; zeroEditDisabled?: boolean; onZeroEdit?: () => void; keyPointsOnly?: boolean; keyPointsDisabled?: boolean; onKeyPointsOnlyChange?: () => void }) {
  const tools: { id: AnnotationTool; icon: typeof Square; label: string }[] = [
    { id: 'select', icon: MousePointer2, label: '선택 · 이동' },
    { id: 'rect', icon: Square, label: '사각형 강조' },
    { id: 'ellipse', icon: Circle, label: '원형 강조' },
    { id: 'arrow', icon: ArrowUpRight, label: '화살표' },
    { id: 'text', icon: Type, label: '텍스트 상자' },
  ];
  return <div className="annotation-toolbar" role="toolbar" aria-label="주석 도구">
    {tools.map((item) => { const Icon = item.icon; return <button key={item.id} type="button" className={tool === item.id ? 'active' : ''} onClick={() => setTool(item.id)} aria-pressed={tool === item.id} aria-label={item.label} title={item.label}><Icon size={14} /></button>; })}
    <span className="annotation-toolbar__divider" />
    {/* 팔레트는 모드가 아니라 동작이다. 선택한 주석의 색을 그대로 비추므로 화면과 어긋나지 않는다. */}
    <div className={`annotation-palette ${selectedColor ? '' : 'annotation-palette--idle'}`} role="group" aria-label="주석 색상">
      {ANNOTATION_COLORS.map((item) => <button key={item.hex} type="button" className={`annotation-swatch ${selectedColor === item.hex ? 'annotation-swatch--active' : ''}`}
        style={{ ['--swatch' as string]: item.hex }} onClick={() => onColorChange(item.hex)} disabled={!selectedColor} aria-pressed={selectedColor === item.hex}
        aria-label={item.label} title={selectedColor ? `${withRo(item.label)} 변경` : '주석을 먼저 선택하세요'} />)}
    </div>
    <span className="annotation-toolbar__divider" />
    {onDetailMode && <button type="button" className={detailMode ? 'active detail' : ''} onClick={onDetailMode} aria-pressed={detailMode} aria-label="Detail View 영역 만들기" title="Detail View 영역 만들기"><ZoomIn size={14} /></button>}
    {onLabelAreaMode && <button type="button" className={labelAreaMode === 'hide' ? 'active hide-area' : ''} onClick={() => onLabelAreaMode('hide')} aria-pressed={labelAreaMode === 'hide'} aria-label="영역 내 라벨 숨기기" title="영역 내 라벨 숨기기"><EyeOff size={14} /></button>}
    {onLabelAreaMode && <button type="button" className={labelAreaMode === 'show' ? 'active show-area' : ''} onClick={() => onLabelAreaMode('show')} aria-pressed={labelAreaMode === 'show'} aria-label="영역 내 라벨 보이기" title="영역 내 라벨 보이기"><Eye size={14} /></button>}
    {onAddPointMode && <button type="button" className={addPointMode ? 'active add-point' : ''} onClick={onAddPointMode} aria-pressed={addPointMode} aria-label="보정 포인트 추가" title="보정 포인트 추가 — 부품 위를 누르면 그 자리의 편차값을 색에서 추정합니다"><Crosshair size={14} /></button>}
    {onDetailMode && <span className="annotation-toolbar__divider" />}
    <button type="button" onClick={onClearAll} disabled={!hasAnnotations} aria-label="주석 전체 삭제" title="주석 전체 삭제"><Trash2 size={14} /></button>
    {onZeroEdit && <button type="button" className={`annotation-toolbar__zero-edit ${zeroEditActive ? 'active' : ''}`} onClick={onZeroEdit} disabled={zeroEditDisabled} aria-pressed={Boolean(zeroEditActive)} title={zeroEditDisabled ? '편집 가능한 제로라인 좌표가 없습니다' : '제로라인의 꼭짓점과 위치를 수정합니다'}><Move size={18} /><span>제로라인 수정</span></button>}
    {onKeyPointsOnlyChange && <button type="button" role="switch" className={`annotation-toolbar__key-points ${keyPointsOnly ? 'active' : ''}`} aria-checked={Boolean(keyPointsOnly)} onClick={onKeyPointsOnlyChange} disabled={keyPointsDisabled} title={keyPointsDisabled ? '주요 포인트 정보가 없습니다. 이미지를 다시 분석해 주세요.' : '주요 포인트만 보정시트에 표시합니다.'}><i /><span>주요포인트만 표시</span></button>}
  </div>;
}

function AnnotationLayer({ annotations, tool, setTool, selectedId, onSelect, onCommit, onCreate, onDelete }: { annotations: Annotation[]; tool: AnnotationTool; setTool: (tool: AnnotationTool) => void; selectedId: string | null; onSelect: (id: string | null) => void; onCommit: (annotation: Annotation) => void; onCreate: (annotation: Annotation) => void; onDelete: (id: string) => void }) {
  const layerRef = useRef<HTMLDivElement>(null);
  const [layerSize, setLayerSize] = useState({ width: 0, height: 0 });
  /* 드래그 중에는 부모 state 를 건드리지 않고 여기서만 갱신한다. 포인트가 수십 개일 때 시트 전체가 매 프레임 다시 그려지는 걸 막는다. */
  const [draft, setDraftState] = useState<Annotation | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const opRef = useRef<{ mode: 'draw' | 'move' | 'resize'; handle?: string; startX: number; startY: number; origin: Annotation; moved: boolean } | null>(null);
  /* pointerup 이 리렌더 전에 도착해도 마지막 값을 잃지 않도록 ref 로도 들고 있는다. */
  const draftRef = useRef<Annotation | null>(null);
  const setDraft = (next: Annotation | null) => { draftRef.current = next; setDraftState(next); };

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    const updateSize = () => setLayerSize({ width: layer.clientWidth, height: layer.clientHeight });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(layer);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!selectedId || editingId) return;
    const handleKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); onDelete(selectedId); onSelect(null); }
      if (event.key === 'Escape') onSelect(null);
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [selectedId, editingId, onDelete, onSelect]);

  /* 주석 바깥을 누르면 선택을 푼다. 레이어 자체는 pointer-events 를 받지 않아서 문서 단위로 감시해야 한다. */
  useEffect(() => {
    if (!selectedId) return;
    const handleOutside = (event: PointerEvent) => {
      /* document 나 window 가 target 인 이벤트도 들어올 수 있어 Element 인지 먼저 확인한다. */
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest('.annotation-shape, .annotation-handle, .annotation-delete, .annotation-fontsize, .annotation-toolbar, .annotation-arrow__hit')) return;
      onSelect(null);
    };
    document.addEventListener('pointerdown', handleOutside);
    return () => document.removeEventListener('pointerdown', handleOutside);
  }, [selectedId, onSelect]);

  /* getBoundingClientRect 는 확대/이동 변형이 반영된 값이라 어떤 배율에서도 같은 % 가 나온다. */
  const toPercent = (clientX: number, clientY: number) => {
    const rect = layerRef.current?.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return { x: 0, y: 0 };
    return { x: (clientX - rect.left) / rect.width * 100, y: (clientY - rect.top) / rect.height * 100 };
  };

  const beginDraw = (event: React.PointerEvent<HTMLDivElement>) => {
    if (tool === 'select' || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    const seed: Annotation = { id: nextAnnotationId(), kind: tool, x: start.x, y: start.y, w: 0, h: 0, color: DEFAULT_ANNOTATION_COLOR, ...(tool === 'text' ? { text: '' } : {}) };
    opRef.current = { mode: 'draw', startX: start.x, startY: start.y, origin: seed, moved: false };
    setDraft(seed);
    capturePointer(event.currentTarget, event.pointerId);
  };

  const beginMove = (event: React.PointerEvent<Element>, annotation: Annotation) => {
    if (tool !== 'select' || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    opRef.current = { mode: 'move', startX: start.x, startY: start.y, origin: annotation, moved: false };
    onSelect(annotation.id);
    setDraft(annotation);
    capturePointer(event.currentTarget, event.pointerId);
  };

  const beginResize = (event: React.PointerEvent<Element>, annotation: Annotation, handle: string) => {
    if (event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    opRef.current = { mode: 'resize', handle, startX: start.x, startY: start.y, origin: annotation, moved: false };
    setDraft(annotation);
    capturePointer(event.currentTarget, event.pointerId);
  };

  const handleMove = (event: React.PointerEvent<Element>) => {
    const op = opRef.current;
    if (!op) return;
    const now = toPercent(event.clientX, event.clientY);
    const dx = now.x - op.startX;
    const dy = now.y - op.startY;
    if (Math.abs(dx) > 0.3 || Math.abs(dy) > 0.3) op.moved = true;
    const origin = op.origin;

    if (op.mode === 'draw') {
      setDraft({ ...origin, w: clamp(now.x, 0, 100) - origin.x, h: clamp(now.y, 0, 100) - origin.y });
      return;
    }
    if (op.mode === 'move') {
      if (origin.kind === 'arrow') {
        const minX = Math.min(0, -origin.w); const maxX = 100 - Math.max(0, origin.w);
        const minY = Math.min(0, -origin.h); const maxY = 100 - Math.max(0, origin.h);
        setDraft({ ...origin, x: clamp(origin.x + dx, minX, maxX), y: clamp(origin.y + dy, minY, maxY) });
      } else {
        setDraft({ ...origin, x: clamp(origin.x + dx, 0, 100 - origin.w), y: clamp(origin.y + dy, 0, 100 - origin.h) });
      }
      return;
    }
    if (origin.kind === 'arrow') {
      /* 화살표는 잡은 쪽 끝점만 따라오고 반대쪽은 제자리를 지킨다. */
      if (op.handle === 'start') {
        const nx = clamp(now.x, 0, 100); const ny = clamp(now.y, 0, 100);
        setDraft({ ...origin, x: nx, y: ny, w: origin.x + origin.w - nx, h: origin.y + origin.h - ny });
      } else {
        setDraft({ ...origin, w: clamp(now.x, 0, 100) - origin.x, h: clamp(now.y, 0, 100) - origin.y });
      }
      return;
    }
    const handle = op.handle || 'se';
    let { x, y, w, h } = origin;
    if (handle.includes('n')) { y = origin.y + dy; h = origin.h - dy; }
    if (handle.includes('s')) { h = origin.h + dy; }
    if (handle.includes('w')) { x = origin.x + dx; w = origin.w - dx; }
    if (handle.includes('e')) { w = origin.w + dx; }
    setDraft({ ...origin, x, y, w, h });
  };

  const endOperation = (event: React.PointerEvent<Element>) => {
    const op = opRef.current;
    const current = draftRef.current;
    opRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (!op || !current) { setDraft(null); return; }
    setDraft(null);

    if (op.mode === 'draw') {
      /* 끌지 않고 툭 찍기만 해도 기본 크기 도형이 생기게 한다. */
      const tiny = Math.abs(current.w) < MIN_ANNOTATION_SIZE && Math.abs(current.h) < MIN_ANNOTATION_SIZE;
      const sized = tiny ? { ...current, ...DEFAULT_ANNOTATION_SIZE[current.kind] } : current;
      const created = normalizeAnnotation(sized);
      onCreate(created);
      onSelect(created.id);
      setTool('select');
      if (created.kind === 'text') { setEditingId(created.id); setEditText(''); }
      return;
    }
    if (!op.moved) return;
    onCommit(normalizeAnnotation(current));
  };

  const commitText = () => {
    if (!editingId) return;
    const target = annotations.find((item) => item.id === editingId);
    if (target) onCommit({ ...target, text: editText });
    setEditingId(null);
  };

  const armed = tool !== 'select';
  const rendered = annotations.map((item) => draft && draft.id === item.id ? draft : item);
  const drawing = draft && !annotations.some((item) => item.id === draft.id) ? draft : null;
  const arrows = [...rendered, ...(drawing ? [drawing] : [])].filter((item) => item.kind === 'arrow');

  const selectedAnnotation = !armed && selectedId && !drawing ? rendered.find((item) => item.id === selectedId) : undefined;
  const selectedHandles = selectedAnnotation ? annotationHandleDescriptors(selectedAnnotation) : [];
  const selectedAnchorX = selectedAnnotation?.kind === 'arrow'
    ? Math.max(selectedAnnotation.x, selectedAnnotation.x + selectedAnnotation.w)
    : selectedAnnotation ? Math.min(selectedAnnotation.x, selectedAnnotation.x + selectedAnnotation.w) + Math.abs(selectedAnnotation.w) : 0;
  const selectedAnchorY = selectedAnnotation?.kind === 'arrow'
    ? Math.min(selectedAnnotation.y, selectedAnnotation.y + selectedAnnotation.h)
    : selectedAnnotation ? Math.min(selectedAnnotation.y, selectedAnnotation.y + selectedAnnotation.h) : 0;
  const selectedSize = selectedAnnotation?.fontSize ?? DEFAULT_TEXT_SIZE;
  const selectedHex = selectedAnnotation?.color || DEFAULT_ANNOTATION_COLOR;
  const selectedFontFamily = selectedAnnotation?.fontFamily || DEFAULT_ANNOTATION_TEXT_FONT;

  return <div ref={layerRef} className={`annotation-layer ${armed ? 'annotation-layer--armed' : ''} annotation-layer--${tool}`}
    onPointerDown={beginDraw} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation}>

    {layerSize.width > 0 && arrows.length > 0 && <svg className="annotation-arrows" viewBox={`0 0 ${layerSize.width} ${layerSize.height}`} aria-hidden="true">
      {/* 화살촉은 marker 안에서 색을 물려받지 못해 쓰이는 색마다 하나씩 만든다. */}
      <defs>{[...new Set(arrows.map((arrow) => arrow.color || DEFAULT_ANNOTATION_COLOR))].map((hex) => (
        <marker key={hex} id={`adc-arrowhead-${hex.replace('#', '')}`} markerWidth="9" markerHeight="7" refX="8.2" refY="3.5" orient="auto">
          <polygon points="0 0, 9 3.5, 0 7" fill={hex} />
        </marker>
      ))}</defs>
      {arrows.map((arrow) => {
        const x1 = layerSize.width * arrow.x / 100; const y1 = layerSize.height * arrow.y / 100;
        const x2 = layerSize.width * (arrow.x + arrow.w) / 100; const y2 = layerSize.height * (arrow.y + arrow.h) / 100;
        const hex = arrow.color || DEFAULT_ANNOTATION_COLOR;
        return <g key={arrow.id}>
          <line className="annotation-arrow__line" style={{ stroke: hex }} x1={x1} y1={y1} x2={x2} y2={y2} markerEnd={`url(#adc-arrowhead-${hex.replace('#', '')})`} />
          {!armed && <line className="annotation-arrow__hit" x1={x1} y1={y1} x2={x2} y2={y2}
            onPointerDown={(event) => beginMove(event, arrow)} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation} />}
        </g>;
      })}
    </svg>}

    {[...rendered, ...(drawing ? [drawing] : [])].map((annotation) => {
      if (annotation.kind === 'arrow') return null;
      const selected = selectedId === annotation.id && !drawing;
      const editing = editingId === annotation.id;
      const hex = annotation.color || DEFAULT_ANNOTATION_COLOR;
      const box = {
        left: `${Math.min(annotation.x, annotation.x + annotation.w)}%`, top: `${Math.min(annotation.y, annotation.y + annotation.h)}%`,
        width: `${Math.abs(annotation.w)}%`, height: `${Math.abs(annotation.h)}%`,
        ['--annot' as string]: hex, ['--annot-fill' as string]: withAlpha(hex, 0.22), ['--annot-glow' as string]: withAlpha(hex, 0.3),
      };
      return <div key={annotation.id} className={`annotation-shape annotation-shape--${annotation.kind} ${selected ? 'annotation-shape--selected' : ''}`} style={box}
        onPointerDown={(event) => beginMove(event, annotation)} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation}
        onDoubleClick={(event) => { if (annotation.kind !== 'text') return; event.stopPropagation(); setEditingId(annotation.id); setEditText(annotation.text || ''); }}>
        {annotation.kind === 'text' && (editing
          ? <textarea className="annotation-text__input" style={{ fontSize: `${annotation.fontSize ?? DEFAULT_TEXT_SIZE}px`, fontFamily: annotation.fontFamily || DEFAULT_ANNOTATION_TEXT_FONT }} value={editText} autoFocus onChange={(event) => setEditText(event.target.value)} onPointerDown={(event) => event.stopPropagation()}
              onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); commitText(); } else if (event.key === 'Escape') { event.preventDefault(); setEditingId(null); } }}
              onBlur={commitText} placeholder="가공 내용 입력" aria-label="주석 텍스트" />
          : <span className="annotation-text__value" style={{ fontSize: `${annotation.fontSize ?? DEFAULT_TEXT_SIZE}px`, fontFamily: annotation.fontFamily || DEFAULT_ANNOTATION_TEXT_FONT }}>{annotation.text || <em>더블클릭해 입력</em>}</span>)}
      </div>;
    })}

    {selectedAnnotation && selectedId && <div className="annotation-selection" style={{ ['--annot' as string]: selectedHex }}>
        {selectedHandles.map(({ handle, className, left, top }) => <span key={handle} className={className}
          style={{ left: `${left}%`, top: `${top}%` }} onPointerDown={(event) => beginResize(event, selectedAnnotation, handle)}
          onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation} />)}
        <button type="button" className="annotation-delete" style={{ left: `${selectedAnchorX}%`, top: `${selectedAnchorY}%` }} onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => { event.stopPropagation(); onDelete(selectedId); onSelect(null); }} aria-label="이 주석 삭제" title="삭제 (Delete)"><X size={11} /></button>
        {/* 기본 동작까지 막으면 Chrome에서 select 메뉴가 열리지 않는다. */}
        {selectedAnnotation.kind === 'text' && <div className="annotation-fontsize" style={{ left: `${Math.min(selectedAnnotation.x, selectedAnnotation.x + selectedAnnotation.w)}%`, top: `${Math.min(selectedAnnotation.y, selectedAnnotation.y + selectedAnnotation.h) + Math.abs(selectedAnnotation.h)}%` }}
          onPointerDown={(event) => event.stopPropagation()}>
          <select className="annotation-fontsize__font" value={selectedFontFamily} onChange={(event) => onCommit({ ...selectedAnnotation, fontFamily: event.target.value })} aria-label="주석 글꼴 선택" title="글꼴 선택">
            {FONT_FAMILY_OPTIONS.map((option) => <option key={option.label} value={option.value} style={{ fontFamily: option.value || undefined }}>{option.label}</option>)}
          </select>
          <button type="button" onClick={() => onCommit({ ...selectedAnnotation, fontSize: clamp(selectedSize - TEXT_SIZE_STEP, TEXT_SIZE_MIN, TEXT_SIZE_MAX) })} disabled={selectedSize <= TEXT_SIZE_MIN} aria-label="글자 작게" title="글자 작게">A<span>−</span></button>
          <span className="annotation-fontsize__value" aria-live="polite">{selectedSize}</span>
          <button type="button" onClick={() => onCommit({ ...selectedAnnotation, fontSize: clamp(selectedSize + TEXT_SIZE_STEP, TEXT_SIZE_MIN, TEXT_SIZE_MAX) })} disabled={selectedSize >= TEXT_SIZE_MAX} aria-label="글자 크게" title="글자 크게">A<span>+</span></button>
        </div>}
      </div>}
  </div>;
}

const MIN_DETAIL_SIZE = 5;
const MIN_LAYOUT_SIZE = 14;
const SHEET_ASPECT = 1.414;

function normalizeBox<T extends { x: number; y: number; w: number; h: number }>(box: T, minimum: number): T {
  let { x, y, w, h } = box;
  if (w < 0) { x += w; w = -w; }
  if (h < 0) { y += h; h = -h; }
  w = clamp(w, minimum, 100); h = clamp(h, minimum, 100);
  x = clamp(x, 0, 100 - w); y = clamp(y, 0, 100 - h);
  return { ...box, x, y, w, h };
}

function fitAspectSize(imageAspect: number, maxW: number, maxH: number) {
  let w = maxW; let h = w * SHEET_ASPECT / imageAspect;
  if (h > maxH) { h = maxH; w = h * imageAspect / SHEET_ASPECT; }
  return { w, h };
}

function DetailRegionLayer({ regions, active, selectedId, onSelect, onCreate, onChange, onDelete }: { regions: DetailRegion[]; active: boolean; selectedId: string | null; onSelect: (id: string | null) => void; onCreate: (region: DetailRegion) => void; onChange: (region: DetailRegion) => void; onDelete: (id: string) => void }) {
  const layerRef = useRef<HTMLDivElement>(null);
  const [draft, setDraftState] = useState<DetailRegion | null>(null);
  const draftRef = useRef<DetailRegion | null>(null);
  const opRef = useRef<{ mode: 'draw' | 'move' | 'resize'; startX: number; startY: number; origin: DetailRegion; handle?: string } | null>(null);
  const setDraft = (value: DetailRegion | null) => { draftRef.current = value; setDraftState(value); };
  const toPercent = (clientX: number, clientY: number) => {
    const bounds = layerRef.current?.getBoundingClientRect();
    if (!bounds?.width || !bounds.height) return { x: 0, y: 0 };
    return { x: (clientX - bounds.left) / bounds.width * 100, y: (clientY - bounds.top) / bounds.height * 100 };
  };
  const beginDraw = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!active || event.button !== 0 || (event.target as Element).closest('.detail-region')) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    const region: DetailRegion = { id: nextDetailId(), x: start.x, y: start.y, w: 0, h: 0, label: `DETAIL ${String.fromCharCode(65 + regions.length)}` };
    opRef.current = { mode: 'draw', startX: start.x, startY: start.y, origin: region };
    setDraft(region); capturePointer(event.currentTarget, event.pointerId);
  };
  const beginEdit = (event: React.PointerEvent<Element>, region: DetailRegion, mode: 'move' | 'resize', handle?: string) => {
    if (active || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    opRef.current = { mode, startX: start.x, startY: start.y, origin: region, handle };
    onSelect(region.id); setDraft(region); capturePointer(event.currentTarget, event.pointerId);
  };
  const handleMove = (event: React.PointerEvent<Element>) => {
    const op = opRef.current;
    if (!op) return;
    const now = toPercent(event.clientX, event.clientY); const dx = now.x - op.startX; const dy = now.y - op.startY;
    if (op.mode === 'draw') { setDraft({ ...op.origin, w: clamp(now.x, 0, 100) - op.origin.x, h: clamp(now.y, 0, 100) - op.origin.y }); return; }
    if (op.mode === 'move') { setDraft({ ...op.origin, x: clamp(op.origin.x + dx, 0, 100 - op.origin.w), y: clamp(op.origin.y + dy, 0, 100 - op.origin.h) }); return; }
    const handle = op.handle || 'se'; let { x, y, w, h } = op.origin;
    if (handle.includes('n')) { y += dy; h -= dy; } if (handle.includes('s')) h += dy;
    if (handle.includes('w')) { x += dx; w -= dx; } if (handle.includes('e')) w += dx;
    setDraft({ ...op.origin, x, y, w, h });
  };
  const endOperation = (event: React.PointerEvent<Element>) => {
    const op = opRef.current; const current = draftRef.current; opRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDraft(null); if (!op || !current) return;
    const normalized = normalizeBox(op.mode === 'draw' && Math.abs(current.w) < MIN_DETAIL_SIZE && Math.abs(current.h) < MIN_DETAIL_SIZE ? { ...current, w: 18, h: 18 } : current, MIN_DETAIL_SIZE);
    if (op.mode === 'draw') onCreate(normalized); else onChange(normalized);
  };
  const rendered = regions.map((region) => draft?.id === region.id ? draft : region);
  const drawing = draft && !regions.some((region) => region.id === draft.id) ? draft : null;
  return <div ref={layerRef} className={`detail-region-layer ${active ? 'detail-region-layer--active' : ''}`} onPointerDown={beginDraw} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation}>
    {[...rendered, ...(drawing ? [drawing] : [])].map((region) => {
      const box = normalizeBox(region, draft?.id === region.id ? 0 : MIN_DETAIL_SIZE); const selected = selectedId === region.id && !active;
      return <div key={region.id} className={`detail-region ${selected ? 'selected' : ''}`} style={{ left: `${box.x}%`, top: `${box.y}%`, width: `${box.w}%`, height: `${box.h}%` }} onPointerDown={(event) => beginEdit(event, region, 'move')} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation}>
        <span>{region.label}</span>
        {selected && BOX_HANDLES.map((handle) => <i key={handle} className={`layout-resize-handle layout-resize-handle--${handle}`} style={{ left: `${HANDLE_OFFSET[handle].x * 100}%`, top: `${HANDLE_OFFSET[handle].y * 100}%` }} onPointerDown={(event) => beginEdit(event, region, 'resize', handle)} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation} />)}
        {selected && <button type="button" onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onDelete(region.id); }} aria-label={`${region.label} 삭제`}><X size={10} /></button>}
      </div>;
    })}
  </div>;
}

function LabelAreaSelector({ mode, points, onApply, onComplete }: { mode: 'hide' | 'show' | null; points: PointResult[]; onApply: (ids: string[], mode: 'hide' | 'show') => void; onComplete: () => void }) {
  const layerRef = useRef<HTMLDivElement>(null);
  const [draft, setDraftState] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const draftRef = useRef<{ x: number; y: number; w: number; h: number } | null>(null);
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const setDraft = (next: { x: number; y: number; w: number; h: number } | null) => { draftRef.current = next; setDraftState(next); };
  const toPercent = (clientX: number, clientY: number) => {
    const bounds = layerRef.current?.getBoundingClientRect();
    if (!bounds?.width || !bounds.height) return { x: 0, y: 0 };
    return { x: clamp((clientX - bounds.left) / bounds.width * 100, 0, 100), y: clamp((clientY - bounds.top) / bounds.height * 100, 0, 100) };
  };
  const begin = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!mode || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation(); const start = toPercent(event.clientX, event.clientY);
    startRef.current = start; setDraft({ ...start, w: 0, h: 0 }); capturePointer(event.currentTarget, event.pointerId);
  };
  const move = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = startRef.current; if (!start) return; const now = toPercent(event.clientX, event.clientY);
    setDraft({ ...start, w: now.x - start.x, h: now.y - start.y });
  };
  const end = (event: React.PointerEvent<HTMLDivElement>) => {
    const current = draftRef.current; startRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDraft(null); if (!current) return;
    const box = normalizeBox(current, 0);
    if (mode && box.w >= 1 && box.h >= 1) onApply(points.filter((point) => point.x >= box.x && point.x <= box.x + box.w && point.y >= box.y && point.y <= box.y + box.h).map((point) => point.id), mode);
    onComplete();
  };
  const box = draft ? normalizeBox(draft, 0) : null;
  return <div ref={layerRef} className={`label-area-selector ${mode ? `active label-area-selector--${mode}` : ''}`} onPointerDown={begin} onPointerMove={move} onPointerUp={end} onPointerCancel={end}>
    {box && <div className="label-area-selector__box" style={{ left: `${box.x}%`, top: `${box.y}%`, width: `${box.w}%`, height: `${box.h}%` }}><span>라벨 {mode === 'show' ? '표시' : '숨김'} 영역</span></div>}
  </div>;
}

function SheetLayoutFrame({ layout, imageAspect, selected, onSelect, onChange, onDelete, title, children }: { layout: SheetLayout; imageAspect: number; selected: boolean; onSelect: () => void; onChange: (layout: SheetLayout) => void; onDelete?: () => void; title: string; children: React.ReactNode }) {
  const frameRef = useRef<HTMLDivElement>(null);
  const draftRef = useRef<SheetLayout | null>(null);
  const [draft, setDraftState] = useState<SheetLayout | null>(null);
  const opRef = useRef<{ mode: 'move' | 'resize'; pointerId: number; clientX: number; clientY: number; origin: SheetLayout; handle?: string } | null>(null);
  const setDraft = (value: SheetLayout | null) => { draftRef.current = value; setDraftState(value); };
  const begin = (event: React.PointerEvent<Element>, mode: 'move' | 'resize', handle?: string) => {
    if (event.button !== 0) return; event.preventDefault(); event.stopPropagation(); onSelect();
    opRef.current = { mode, pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, origin: layout, handle };
    setDraft(layout); capturePointer(event.currentTarget, event.pointerId);
  };
  const move = (event: React.PointerEvent<Element>) => {
    const op = opRef.current; const canvas = frameRef.current?.parentElement?.getBoundingClientRect(); if (!op || !canvas) return;
    const dx = (event.clientX - op.clientX) / canvas.width * 100; const dy = (event.clientY - op.clientY) / canvas.height * 100;
    if (op.mode === 'move') { setDraft({ ...op.origin, x: clamp(op.origin.x + dx, 0, 100 - op.origin.w), y: clamp(op.origin.y + dy, 0, 100 - op.origin.h) }); return; }
    const handle = op.handle || 'se'; let { x, y, w, h } = op.origin;
    const horizontal = handle.includes('e') || handle.includes('w');
    const vertical = handle.includes('n') || handle.includes('s');
    const rawW = handle.includes('w') ? op.origin.w - dx : handle.includes('e') ? op.origin.w + dx : op.origin.w;
    const rawH = handle.includes('n') ? op.origin.h - dy : handle.includes('s') ? op.origin.h + dy : op.origin.h;
    if (horizontal && (!vertical || Math.abs(dx) >= Math.abs(dy))) {
      const sign = Math.sign(rawW || 1); const minW = Math.max(MIN_LAYOUT_SIZE, MIN_LAYOUT_SIZE * imageAspect / (canvas.width / canvas.height)); const maxW = Math.min(100, 100 * imageAspect / (canvas.width / canvas.height));
      w = clamp(Math.abs(rawW), minW, maxW) * sign; h = Math.abs(w) * canvas.width / canvas.height / imageAspect * sign;
    } else {
      const sign = Math.sign(rawH || 1); const minH = Math.max(MIN_LAYOUT_SIZE, MIN_LAYOUT_SIZE * (canvas.width / canvas.height) / imageAspect); const maxH = Math.min(100, 100 * (canvas.width / canvas.height) / imageAspect);
      h = clamp(Math.abs(rawH), minH, maxH) * sign; w = Math.abs(h) * canvas.height / canvas.width * imageAspect * sign;
    }
    if (handle.includes('w')) x = op.origin.x + op.origin.w - w;
    if (handle.includes('n')) y = op.origin.y + op.origin.h - h;
    setDraft({ ...op.origin, x, y, w, h });
  };
  const end = (event: React.PointerEvent<Element>) => {
    const current = draftRef.current; opRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDraft(null); if (current) onChange(normalizeBox(current, MIN_LAYOUT_SIZE));
  };
  const shown = draft ? normalizeBox(draft, 0) : layout;
  return <article ref={frameRef} className={`sheet-layout sheet-layout--${layout.kind} ${selected ? 'selected' : ''}`} style={{ left: `${shown.x}%`, top: `${shown.y}%`, width: `${shown.w}%`, height: `${shown.h}%` }} onPointerDown={onSelect}>
    <div className="sheet-layout__bar" onPointerDown={(event) => begin(event, 'move')} onPointerMove={move} onPointerUp={end} onPointerCancel={end}><span>{title}</span><small>비율 고정 · 드래그 이동</small>{onDelete && <button type="button" onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onDelete(); }} aria-label={`${title} 삭제`}><X size={11} /></button>}</div>
    <div className="sheet-layout__content">{children}</div>
    {selected && BOX_HANDLES.map((handle) => <i key={handle} className={`layout-resize-handle layout-resize-handle--${handle}`} style={{ left: `${HANDLE_OFFSET[handle].x * 100}%`, top: `${HANDLE_OFFSET[handle].y * 100}%` }} onPointerDown={(event) => begin(event, 'resize', handle)} onPointerMove={move} onPointerUp={end} onPointerCancel={end} />)}
  </article>;
}

function CorrectionPoints({ coefficient, points, labels = true, visibleLabelIds, onLabelToggle, overrides, onOverrideChange, labelFontFamily, initialLabelPositions, rotationSeed, onLabelPositionsChange, onLayerSizeChange }: { coefficient: number; points: PointResult[]; labels?: boolean; visibleLabelIds?: Set<string>; onLabelToggle?: (id: string) => void; overrides?: Record<string, number>; onOverrideChange?: (id: string, value: number | null) => void; labelFontFamily?: string; initialLabelPositions?: Record<string, { x: number; y: number }>; rotationSeed?: { transform: SheetImageTransform; canonicalOffsets: Record<string, { x: number; y: number }> }; onLabelPositionsChange?: (positions: Record<string, { x: number; y: number }>, layerSize?: { width: number; height: number }) => void; onLayerSizeChange?: (size: { width: number; height: number }) => void }) {
  const labelHeight = 17;
  const displayFor = useCallback((point: PointResult) => overrides?.[point.id] !== undefined ? overrides[point.id]! : -(point.value * coefficient), [coefficient, overrides]);
  const formatCorrection = useCallback((value: number) => `${value > 0 ? '+' : ''}${value.toFixed(1)}`, []);
  const getLabelWidth = useCallback((point: PointResult) => Math.max(24, formatCorrection(displayFor(point)).length * 5.2 + 8), [displayFor, formatCorrection]);
  const layerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ id: string; x: number; y: number; clientX: number; clientY: number; moved: boolean } | null>(null);
  const ignoreClickRef = useRef(false);
  const layoutKeyRef = useRef('');
  const previousLayerSizeRef = useRef({ width: 0, height: 0 });
  const [layerSize, setLayerSize] = useState({ width: 0, height: 0 });
  const [labelPositions, setLabelPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const editOriginalRef = useRef<number | null>(null);
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    const updateSize = () => setLayerSize({ width: layer.clientWidth, height: layer.clientHeight });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(layer);
    return () => observer.disconnect();
  }, []);
  /* 부모(ServicePreview)가 라벨-점 오프셋을 "실제 화면 픽셀"로 저장하려면
     이 레이어의 진짜 픽셀 크기를 알아야 한다 -- 정면도 창은 회전할 때마다
     실제 크기가 바뀌는데(비율만 유지), % 만으로는 그 실제 크기를 부모가
     알 수 없어 지시선 길이가 회전마다 미묘하게 달라졌다. */
  useEffect(() => {
    if (!layerSize.width || !layerSize.height) return;
    onLayerSizeChange?.(layerSize);
  }, [layerSize, onLayerSizeChange]);
  useEffect(() => {
    const previous = previousLayerSizeRef.current;
    if (previous.width > 0 && previous.height > 0 && (previous.width !== layerSize.width || previous.height !== layerSize.height)) {
      setLabelPositions((current) => Object.fromEntries(Object.entries(current).map(([id, position]) => [id, { x: position.x * layerSize.width / previous.width, y: position.y * layerSize.height / previous.height }])));
    }
    previousLayerSizeRef.current = layerSize;
  }, [layerSize]);
  useEffect(() => {
    if (!layerSize.width) return;
    const layoutKey = `${labelHeight}:${points.map((point) => `${point.id}:${formatCorrection(displayFor(point)).length}`).join('|')}`;
    const resetLayout = layoutKeyRef.current !== layoutKey;
    layoutKeyRef.current = layoutKey;
    setLabelPositions((current) => {
      const next = resetLayout ? {} : { ...current };
      const centerX = points.length > 1 ? points.reduce((sum, point) => sum + layerSize.width * point.x / 100, 0) / points.length : layerSize.width / 2;
      const centerY = points.length > 1 ? points.reduce((sum, point) => sum + layerSize.height * point.y / 100, 0) / points.length : layerSize.height / 2;
      points.forEach((point) => {
        if (next[point.id]) return;
        const labelWidth = getLabelWidth(point);
        const pointX = layerSize.width * point.x / 100;
        const pointY = layerSize.height * point.y / 100;
        /* 회전마다 이 컴포넌트가 통째로 리마운트되는데(위 참고), 회전 직후엔
           부모(ServicePreview)가 아직 "이 회전에서 실제로 몇 px 짜리 창이
           될지"를 모른다 -- 그 값은 이 레이어가 실제로 마운트돼 측정돼야만
           나온다. 그런데 부모가 %(initialLabelPositions)로 미리 계산해 넘긴
           값은 회전 *직전* 창 크기로 계산된 것이라 어긋난다(부모의 리렌더가
           이 컴포넌트의 첫 렌더보다 한 박자 늦다). 그래서 점→라벨 오프셋을
           "회전 전 기준"의 등방(isotropic) 벡터(canonicalOffsets, 부모가
           들고 있는 값)로 받아, 지금 막 실측한 이 레이어의 layerSize 로 직접
           풀어낸다 -- 그러면 부모의 타이밍과 무관하게 항상 맞는 크기로 계산
           된다(지시선 길이가 회전마다 미묘하게 달라지던 원인). */
        const canonical = rotationSeed?.canonicalOffsets[point.id];
        if (canonical && Number.isFinite(canonical.x) && Number.isFinite(canonical.y)) {
          const [isoDx, isoDy] = rotateVector(rotationSeed!.transform, canonical.x, canonical.y);
          const layerScale = Math.sqrt(layerSize.width * layerSize.height) || 1;
          const pxDx = isoDx * (layerSize.width / layerScale);
          const pxDy = isoDy * (layerSize.height / layerScale);
          next[point.id] = { x: pointX + pxDx - labelWidth / 2, y: pointY + pxDy - labelHeight / 2 };
          return;
        }
        /* 부모가 마지막으로 기억한(사용자가 드래그해 옮긴) 위치가 있으면
           그걸 되살린다 — 이 컴포넌트는 화면(스캔/제품데이터/정렬확인)을
           바꿀 때마다 통째로 다시 마운트되는 SheetCanvas 의 자식이라, 이
           시딩이 없으면 매번 아래 자동배치로 되돌아가 겹친다. */
        const seed = initialLabelPositions?.[point.id];
        if (seed && Number.isFinite(seed.x) && Number.isFinite(seed.y)) {
          next[point.id] = { x: seed.x / 100 * layerSize.width, y: seed.y / 100 * layerSize.height };
          return;
        }
        let dx = pointX - centerX;
        let dy = pointY - centerY;
        if (Math.abs(dx) + Math.abs(dy) < 1) { dx = 1; dy = 0; }
        const length = Math.hypot(dx, dy);
        const unitX = dx / length;
        const unitY = dy / length;
        const leaderEndX = pointX + unitX * 20;
        const leaderEndY = pointY + unitY * 20;
        next[point.id] = {
          x: unitX >= 0 ? leaderEndX : leaderEndX - labelWidth,
          y: unitY >= 0 ? leaderEndY : leaderEndY - labelHeight,
        };
      });
      return next;
    });
  }, [layerSize, points, labelHeight, displayFor, formatCorrection, getLabelWidth, initialLabelPositions, rotationSeed]);
  /* 엑셀 내보내기가 화면과 똑같은 라벨 위치를 쓰도록, 실제로 계산·드래그된
     좌표(픽셀, .point-layer 기준)를 점과 같은 규칙(레이어 대비 0~100%)으로
     정규화해 위로 올려 준다. DOM에서 나중에 다시 측정하지 않는 이유: 이
     state 가 이미 정답이고, getBoundingClientRect 는 시트가 화면에 실제로
     그 크기로 떠 있어야만(스크롤 밖·숨김 탭이면 0) 값을 주는 데다 border/
     padding 같은 걸 다 다시 맞춰야 해서 어긋나기 쉽다.

     layerSize 를 %와 함께 그대로 실어 보내는 이유: 부모(ServicePreview)도
     "정면도 지시선 길이 고정" 계산에 이 레이어의 실제 픽셀 크기가 필요한데,
     부모가 스스로 들고 있는 값(frontLayerSizePx, onLayerSizeChange 로 받음)은
     탭을 오가며 ServicePreview 가 통째로 리마운트되면 한 렌더 늦게 갱신된다
     -- 그 한 박자 동안 부모가 옛(또는 기본값 1×1) 크기로 방금 세팅한 위치를
     되돌려 계산하면 원래 저장해 둔 값이 틀어진다. 이 이펙트가 도는 시점의
     layerSize 는 이 컴포넌트 자신이 막 실측한 값이라 항상 맞다 -- 부모가
     그 값을 그대로 쓰면 자기 state 의 타이밍에 기대지 않아도 된다.

     points 를 다 세팅할 때까지 기다리는 이유: layerSize 가 막 실측돼 0에서
     실제값으로 바뀌는 바로 그 렌더에서는, 바로 위 시딩 이펙트가 부른
     setLabelPositions 가 아직 커밋되지 않은 채로(React 는 같은 커밋 안에서
     다음 렌더까지 미룬다) 이 이펙트가 옛(대개 빈 {}) labelPositions 를 들고
     먼저 돈다 -- 그 순간의 "거의 빈" 스냅샷을 그대로 부모에 보고하면, 부모가
     이미 갖고 있던 옳은 canonical 값을 이 빈 값으로 덮어써 버린다(포인트가
     새로 생기거나 리마운트될 때마다 그 포인트의 저장된 라벨 위치가 지워지는
     증상). 지금 있어야 할 점(points) 이 아직 labelPositions 에 다 안 채워진
     스냅샷은 과도기 상태이므로 보고를 건너뛰고, 다음 렌더(시딩이 실제로
     반영된 뒤)를 기다린다. */
  useEffect(() => {
    if (!onLabelPositionsChange) return;
    if (!layerSize.width || !layerSize.height) return;
    if (points.some((point) => !labelPositions[point.id])) return;
    const normalized: Record<string, { x: number; y: number }> = {};
    for (const [id, position] of Object.entries(labelPositions)) {
      normalized[id] = { x: position.x / layerSize.width * 100, y: position.y / layerSize.height * 100 };
    }
    onLabelPositionsChange(normalized, layerSize);
  }, [labelPositions, layerSize, onLabelPositionsChange, points]);
  const beginLabelDrag = (event: React.PointerEvent<HTMLSpanElement>, id: string) => {
    const position = labelPositions[id];
    if (!position) return;
    event.preventDefault(); event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { id, x: position.x, y: position.y, clientX: event.clientX, clientY: event.clientY, moved: false };
  };
  const moveLabel = (event: React.PointerEvent<HTMLSpanElement>) => {
    const drag = dragRef.current;
    const layer = layerRef.current;
    if (!drag || drag.id !== event.currentTarget.dataset.pointId || !layer) return;
    const scale = layer.getBoundingClientRect().width / Math.max(layer.clientWidth, 1);
    const movedX = drag.x + (event.clientX - drag.clientX) / scale;
    drag.moved ||= Math.hypot(event.clientX - drag.clientX, event.clientY - drag.clientY) > 3;
    setLabelPositions((current) => ({ ...current, [drag.id]: {
      x: Math.max(-140, Math.min(layer.clientWidth + 62, movedX)),
      y: Math.max(-96, Math.min(layer.clientHeight + 64, drag.y + (event.clientY - drag.clientY) / scale)),
    }}));
  };
  const endLabelDrag = (event: React.PointerEvent<HTMLSpanElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.id !== event.currentTarget.dataset.pointId) return;
    dragRef.current = null; ignoreClickRef.current = drag.moved;
    if (drag.moved) window.setTimeout(() => { ignoreClickRef.current = false; }, 0);
    event.currentTarget.releasePointerCapture(event.pointerId);
  };
  const startEdit = (id: string, currentValue: number) => {
    setEditingId(id);
    setEditValue(currentValue.toFixed(1));
    editOriginalRef.current = currentValue;
  };
  const commitEdit = () => {
    if (!editingId || !onOverrideChange) { setEditingId(null); return; }
    const parsed = parseFloat(editValue);
    /* 라벨을 눌렀다가 아무것도 안 바꾸고 포커스만 벗어나도 blur 로 commitEdit 이 불린다.
       값이 실제로 안 바뀌었으면 onOverrideChange 를 아예 부르지 않아야, 그냥 눌러보기만 해도
       자동값이 수동값으로 바뀌고 이력에 기록되는 일이 없다. */
    const unchanged = editOriginalRef.current !== null && Math.abs(parsed - editOriginalRef.current) < 0.0001;
    if (Number.isFinite(parsed) && !unchanged) onOverrideChange(editingId, parsed);
    setEditingId(null);
  };
  const cancelEdit = () => setEditingId(null);
  const resetOverride = () => {
    if (!editingId || !onOverrideChange) return;
    onOverrideChange(editingId, null);
    setEditingId(null);
  };
  return <div className="point-layer" ref={layerRef}>
    <svg className="point-leaders" aria-hidden="true" viewBox={`0 0 ${layerSize.width} ${layerSize.height}`}>{points.map((point) => {
       const position = labelPositions[point.id]; const visible = !visibleLabelIds || visibleLabelIds.has(point.id);
       if (!position || !visible) return null;
       const labelWidth = getLabelWidth(point);
       const isOverridden = overrides?.[point.id] !== undefined;
       const x1 = layerSize.width * point.x / 100; const y1 = layerSize.height * point.y / 100;
      let x2 = Math.max(position.x, Math.min(position.x + labelWidth, x1));
      let y2 = Math.max(position.y, Math.min(position.y + labelHeight, y1));
      if (x2 === x1 && y2 === y1) {
        const distances = [
          { value: x1 - position.x, edge: 'left' },
          { value: position.x + labelWidth - x1, edge: 'right' },
          { value: y1 - position.y, edge: 'top' },
          { value: position.y + labelHeight - y1, edge: 'bottom' },
        ].sort((a, b) => a.value - b.value);
        if (distances[0].edge === 'left') x2 = position.x;
        if (distances[0].edge === 'right') x2 = position.x + labelWidth;
        if (distances[0].edge === 'top') y2 = position.y;
        if (distances[0].edge === 'bottom') y2 = position.y + labelHeight;
      }
      return <line key={point.id} className={isOverridden ? 'point-leader--overridden' : undefined} x1={x1} y1={y1} x2={x2} y2={y2} />;
    })}</svg>{points.map((point) => {
    const display = displayFor(point);
    const labelVisible = !visibleLabelIds || visibleLabelIds.has(point.id);
    const position = labelPositions[point.id];
    const isOverridden = overrides?.[point.id] !== undefined;
    const isEditing = editingId === point.id;
    const editable = Boolean(onOverrideChange);
    const labelStyle = position ? { left: `${position.x - layerSize.width * point.x / 100}px`, top: `${position.y - layerSize.height * point.y / 100}px`, fontFamily: labelFontFamily || undefined } : undefined;
    const labelClasses = ['measure-point__label', display >= 0 ? 'measure-point__label--plus' : 'measure-point__label--minus'];
    if (editable) labelClasses.push('measure-point__label--editable');
    if (isOverridden) labelClasses.push('measure-point__label--overridden');
    if (isEditing) labelClasses.push('measure-point__label--editing');
    if (point.source === 'colormap') labelClasses.push('measure-point__label--estimated');
    return <div className={`measure-point ${display >= 0 ? 'measure-point--plus' : 'measure-point--minus'} ${onLabelToggle ? 'measure-point--interactive' : ''} ${labelVisible ? '' : 'measure-point--hidden'} ${point.source === 'colormap' ? 'measure-point--estimated' : ''}`} style={{ left: `${point.x}%`, top: `${point.y}%` }} key={point.id}>
      <button type="button" className="measure-point__dot" onClick={() => onLabelToggle?.(point.id)} aria-label={`${point.id} 라벨 ${labelVisible ? '숨기기' : '표시하기'}`} aria-pressed={labelVisible} title={`${point.id} 편차 ${point.value > 0 ? '+' : ''}${point.value.toFixed(3)} · 점 클릭으로 표시 전환`} />
      {labels && labelVisible && position && (isEditing ? <span className={labelClasses.join(' ')} style={labelStyle}>
        <input type="text" inputMode="decimal" className="measure-point__label__input" value={editValue} onChange={(e) => setEditValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); commitEdit(); } else if (e.key === 'Escape') { e.preventDefault(); cancelEdit(); } }} onBlur={commitEdit} autoFocus onFocus={(e) => e.currentTarget.select()} aria-label={`${point.id} 보정치 편집`} />
        {isOverridden && <button type="button" className="measure-point__label__reset" onMouseDown={(e) => e.preventDefault()} onClick={resetOverride} aria-label="자동값으로 되돌리기" title="자동값으로 되돌리기">↺</button>}
      </span> : <span className={labelClasses.join(' ')} data-point-id={point.id} style={labelStyle} onPointerDown={(event) => beginLabelDrag(event, point.id)} onPointerMove={moveLabel} onPointerUp={endLabelDrag} onPointerCancel={endLabelDrag} onClick={() => { if (ignoreClickRef.current) { ignoreClickRef.current = false; return; } if (editable) startEdit(point.id, display); }} title={isOverridden ? `수정된 값 (계수 영향 없음) · 클릭하여 편집` : (editable ? '클릭하여 값 편집 · 드래그로 이동' : undefined)}><span className="measure-point__label__value">{formatCorrection(display)}</span></span>)}
    </div>;
  })}</div>;
}

/* 시트에 얹는 제로라인 오버레이. lines 는 프레임(정면도 또는 detail 크롭이 표시하는 이미지 전체)
   에서 % 좌표로 이미 변환되어 온다. region 이 주어지면 그 크롭 영역 안으로 좌표를 다시 옮긴다.
   SVG viewBox 를 0-100 으로 고정해 부모의 절대 크기와 무관하게 % 좌표를 그대로 쓴다. */
function zeroLineSegmentPath(points: [number, number][], index: number, spline: boolean): string {
  const start = points[index];
  const end = points[index + 1];
  if (!start || !end) return '';
  if (!spline || points.length < 3) return `M ${start[0]} ${start[1]} L ${end[0]} ${end[1]}`;
  const before = points[Math.max(0, index - 1)];
  const after = points[Math.min(points.length - 1, index + 2)];
  const control1: [number, number] = [start[0] + (end[0] - before[0]) / 6, start[1] + (end[1] - before[1]) / 6];
  const control2: [number, number] = [end[0] - (after[0] - start[0]) / 6, end[1] - (after[1] - start[1]) / 6];
  return `M ${start[0]} ${start[1]} C ${control1[0]} ${control1[1]} ${control2[0]} ${control2[1]} ${end[0]} ${end[1]}`;
}

function zeroLinePath(points: [number, number][], splineSegments: number[]): string {
  if (!points.length) return '';
  const parts = [`M ${points[0][0]} ${points[0][1]}`];
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    if (!splineSegments.includes(index) || points.length < 3) {
      parts.push(`L ${end[0]} ${end[1]}`);
      continue;
    }
    const before = points[Math.max(0, index - 1)];
    const after = points[Math.min(points.length - 1, index + 2)];
    const control1: [number, number] = [start[0] + (end[0] - before[0]) / 6, start[1] + (end[1] - before[1]) / 6];
    const control2: [number, number] = [end[0] - (after[0] - start[0]) / 6, end[1] - (after[1] - start[1]) / 6];
    parts.push(`C ${control1[0]} ${control1[1]} ${control2[0]} ${control2[1]} ${end[0]} ${end[1]}`);
  }
  return parts.join(' ');
}

function nearestZeroLinePoint(points: [number, number][], splineSegments: number[], target: [number, number]): { segmentIndex: number; point: [number, number] } | null {
  if (points.length < 2) return null;
  let best: { segmentIndex: number; point: [number, number]; distance: number } | null = null;
  const consider = (segmentIndex: number, point: [number, number]) => {
    const distance = (point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2;
    if (!best || distance < best.distance) best = { segmentIndex, point, distance };
  };
  for (let index = 0; index < points.length - 1; index += 1) {
    const before = points[Math.max(0, index - 1)];
    const start = points[index];
    const end = points[index + 1];
    const after = points[Math.min(points.length - 1, index + 2)];
    const spline = splineSegments.includes(index);
    const steps = spline && points.length >= 3 ? 24 : 1;
    for (let step = 0; step <= steps; step += 1) {
      const t = step / steps;
      if (!spline || points.length < 3) {
        consider(index, [start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t]);
        continue;
      }
      const t2 = t * t; const t3 = t2 * t;
      consider(index, [
        0.5 * ((2 * start[0]) + (-before[0] + end[0]) * t + (2 * before[0] - 5 * start[0] + 4 * end[0] - after[0]) * t2 + (-before[0] + 3 * start[0] - 3 * end[0] + after[0]) * t3),
        0.5 * ((2 * start[1]) + (-before[1] + end[1]) * t + (2 * before[1] - 5 * start[1] + 4 * end[1] - after[1]) * t2 + (-before[1] + 3 * start[1] - 3 * end[1] + after[1]) * t3),
      ]);
    }
  }
  const nearest = best as { segmentIndex: number; point: [number, number]; distance: number } | null;
  return nearest ? { segmentIndex: nearest.segmentIndex, point: nearest.point } : null;
}

function ZeroLineOverlay({ lines, splineSegments = [], region, editable = false, addPointMode = false, deletePointMode = false, onPointMove, onSegmentDoubleClick, onPointAdd, onPointDelete }: { lines: [number, number][][]; splineSegments?: number[][]; region?: DetailRegion; editable?: boolean; addPointMode?: boolean; deletePointMode?: boolean; onPointMove?: (lineIndex: number, pointIndex: number, dxPercent: number, dyPercent: number) => void; onSegmentDoubleClick?: (lineIndex: number, segmentIndex: number) => void; onPointAdd?: (lineIndex: number, segmentIndex: number, xPercent: number, yPercent: number) => void; onPointDelete?: (lineIndex: number, pointIndex: number) => void }) {
  const canEdit = editable && !region;
  const [drag, setDrag] = useState<{ lineIndex: number; pointIndex: number; startX: number; startY: number; dx: number; dy: number; pointerId: number } | null>(null);
  const clickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (clickTimerRef.current) clearTimeout(clickTimerRef.current); }, []);
  const transformed = lines.map((line, sourceIndex) => ({
    sourceIndex,
    points: region ? line.map(([x, y]) => [(x - region.x) / region.w * 100, (y - region.y) / region.h * 100] as [number, number]) : line,
  })).filter(({ points }) => points.length >= 2 && (!region || points.some(([x, y]) => x >= -5 && x <= 105 && y >= -5 && y <= 105)));
  if (!transformed.length) return null;
  const displayed = transformed.map(({ sourceIndex, points }) => points.map(([x, y], pointIndex) => drag?.lineIndex === sourceIndex && drag.pointIndex === pointIndex ? [x + drag.dx, y + drag.dy] as [number, number] : [x, y] as [number, number]));
  const paths = displayed.map((line, index) => zeroLinePath(line, splineSegments[transformed[index].sourceIndex] || []));
  const pointerPosition = (event: React.PointerEvent<HTMLElement>) => {
    const rect = event.currentTarget.parentElement?.getBoundingClientRect();
    return rect?.width && rect.height ? { x: (event.clientX - rect.left) / rect.width * 100, y: (event.clientY - rect.top) / rect.height * 100 } : null;
  };
  return <div className={`zero-line-overlay ${canEdit ? 'zero-line-overlay--editable' : ''}${addPointMode ? ' zero-line-overlay--add-point' : ''}${deletePointMode ? ' zero-line-overlay--delete-point' : ''}`} aria-hidden={canEdit ? undefined : true}>
    <svg className="zero-line-overlay__svg" viewBox="0 0 100 100" preserveAspectRatio="none">
    {/* 부품 재질색(초록·파랑 등)과 겹치면 안 보인다는 피드백 — "분석 결과" 화면의
        래스터 제로라인과 같은 빨강(#dc1414, zero_polyline.draw_zero_polylines 기본값)
        을 쓰고, 흰 테두리(halo)를 먼저 굵게 깐 뒤 그 위에 얹어 어떤 배경에서도
        확실히 도드라지게 한다. */}
    {paths.map((path, idx) => <path key={`halo-${idx}`} d={path} className="zero-line-path zero-line-path--halo" />)}
    {paths.map((path, idx) => <path key={`line-${idx}`} d={path} className="zero-line-path zero-line-path--main" />)}
    {canEdit && displayed.flatMap((line, idx) => {
      const { sourceIndex } = transformed[idx];
      const curved = splineSegments[sourceIndex] || [];
      return line.slice(0, -1).map((_, segmentIndex) => {
        const isSpline = curved.includes(segmentIndex);
        const segmentPath = zeroLineSegmentPath(line, segmentIndex, isSpline);
        return <path key={`hit-${sourceIndex}-${segmentIndex}`} d={segmentPath} className="zero-line-path-hit" tabIndex={0} role="button" aria-label={`제로라인 ${sourceIndex + 1}의 ${segmentIndex + 1}번 구간 · 더블클릭하여 ${isSpline ? '직선' : '스플라인'}으로 변경${addPointMode ? ' · 클릭하여 점 추가' : ''}`} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSegmentDoubleClick?.(sourceIndex, segmentIndex); } }} onDoubleClick={(event) => { event.preventDefault(); event.stopPropagation(); if (clickTimerRef.current) clearTimeout(clickTimerRef.current); clickTimerRef.current = null; onSegmentDoubleClick?.(sourceIndex, segmentIndex); }} onClick={(event) => { if (!addPointMode) return; const rect = event.currentTarget.ownerSVGElement?.getBoundingClientRect(); if (!rect?.width || !rect.height) return; const target: [number, number] = [(event.clientX - rect.left) / rect.width * 100, (event.clientY - rect.top) / rect.height * 100]; if (clickTimerRef.current) clearTimeout(clickTimerRef.current); clickTimerRef.current = setTimeout(() => { const nearest = nearestZeroLinePoint(displayed[idx], curved, target); if (nearest) onPointAdd?.(sourceIndex, nearest.segmentIndex, nearest.point[0], nearest.point[1]); clickTimerRef.current = null; }, 280); }} />;
      });
    })}
    </svg>
    {canEdit && transformed.flatMap(({ sourceIndex }, linePosition) => displayed[linePosition].map(([x, y], pointIndex) => <button type="button" key={`handle-${sourceIndex}-${pointIndex}`} className="zero-line-handle" style={{ left: `${x}%`, top: `${y}%` }} aria-label={`제로라인 ${sourceIndex + 1}의 ${pointIndex + 1}번 점`} title={deletePointMode ? '클릭하여 꼭짓점 삭제' : '끌어서 꼭짓점 이동'} onClick={(event) => { if (!deletePointMode) return; event.preventDefault(); event.stopPropagation(); onPointDelete?.(sourceIndex, pointIndex); }} onKeyDown={(event) => { if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); onPointDelete?.(sourceIndex, pointIndex); } }} onPointerDown={(event) => { if (deletePointMode) { event.preventDefault(); event.stopPropagation(); return; } const point = pointerPosition(event); if (!point) return; event.preventDefault(); event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); setDrag({ lineIndex: sourceIndex, pointIndex, startX: point.x, startY: point.y, dx: 0, dy: 0, pointerId: event.pointerId }); }} onPointerMove={(event) => { if (!drag || drag.pointerId !== event.pointerId || drag.lineIndex !== sourceIndex || drag.pointIndex !== pointIndex) return; const point = pointerPosition(event); if (point) setDrag({ ...drag, dx: point.x - drag.startX, dy: point.y - drag.startY }); }} onPointerUp={(event) => { if (!drag || drag.pointerId !== event.pointerId) return; event.currentTarget.releasePointerCapture(event.pointerId); if (Math.abs(drag.dx) + Math.abs(drag.dy) > 0.01) onPointMove?.(sourceIndex, pointIndex, drag.dx, drag.dy); setDrag(null); }} onPointerCancel={() => setDrag(null)} />))}
  </div>;
}

function SheetCanvas({ scan, imageUrl, frameWidth, frameHeight, initialRegions, initialLayouts, initialLabelPositionsByLayout, frontRotationSeed, onRegionsChange, onLayoutsChange, onLabelPositionsChange, onLayerSizeChange, points, coefficient, showPoints, visiblePointIds, onPointToggle, pointOverrides, onOverrideChange, labelFontFamily, annotations, showAnnotations, annotationTool, setAnnotationTool, selectedAnnotationId, setSelectedAnnotationId, onAnnotationCommit, onAnnotationCreate, onAnnotationDelete, detailMode, setDetailMode, labelAreaMode, setLabelAreaMode, addPointMode, onAddPointAt, sampling, sampleError, addedPoints, onRemoveAddedPoint, zeroLines = [], zeroSplineSegments = [], showZero = false, zeroEditable = false, zeroPointAddMode = false, zeroPointDeleteMode = false, onZeroPointMove, onZeroSegmentDoubleClick, onZeroPointAdd, onZeroPointDelete }: { scan: ScanItem; imageUrl: string; frameWidth: number; frameHeight: number; initialRegions?: DetailRegion[]; initialLayouts?: SheetLayout[]; initialLabelPositionsByLayout?: Record<string, Record<string, { x: number; y: number }>>; frontRotationSeed?: { transform: SheetImageTransform; canonicalOffsets: Record<string, { x: number; y: number }> }; onRegionsChange?: (regions: DetailRegion[]) => void; onLayoutsChange?: (layouts: SheetLayout[]) => void; onLabelPositionsChange?: (layoutId: string, positions: Record<string, { x: number; y: number }>, layerSize?: { width: number; height: number }) => void; onLayerSizeChange?: (layoutId: string, size: { width: number; height: number }) => void; points: PointResult[]; coefficient: number; showPoints: boolean; visiblePointIds: Set<string>; onPointToggle: (id: string) => void; pointOverrides: Record<string, number>; onOverrideChange: (id: string, value: number | null) => void; labelFontFamily?: string; annotations: Annotation[]; showAnnotations: boolean; annotationTool: AnnotationTool; setAnnotationTool: (tool: AnnotationTool) => void; selectedAnnotationId: string | null; setSelectedAnnotationId: (id: string | null) => void; onAnnotationCommit: (annotation: Annotation) => void; onAnnotationCreate: (annotation: Annotation) => void; onAnnotationDelete: (id: string) => void; detailMode: boolean; setDetailMode: (value: boolean) => void; labelAreaMode: 'hide' | 'show' | null; setLabelAreaMode: (value: 'hide' | 'show' | null) => void; addPointMode: boolean; onAddPointAt: (x: number, y: number) => void; sampling: boolean; sampleError: string | null; addedPoints: PointResult[]; onRemoveAddedPoint: (id: string) => void; zeroLines?: [number, number][][]; zeroSplineSegments?: number[][]; showZero?: boolean; zeroEditable?: boolean; zeroPointAddMode?: boolean; zeroPointDeleteMode?: boolean; onZeroPointMove?: (lineIndex: number, pointIndex: number, dxPercent: number, dyPercent: number) => void; onZeroSegmentDoubleClick?: (lineIndex: number, segmentIndex: number) => void; onZeroPointAdd?: (lineIndex: number, segmentIndex: number, xPercent: number, yPercent: number) => void; onZeroPointDelete?: (lineIndex: number, pointIndex: number) => void }) {
  /* 정렬 합성 이미지는 스캔 원본과 크기가 다를 수 있어 프레임 치수를 직접 받는다. */
  const sourceAspect = frameWidth / frameHeight;
  /* 시트 폭·높이 상한. 한 번은 62/64 -> 42/44 로 줄였다가, 이번엔 그
     42/44 기준 디폴트 창이 너무 작다는 요청으로 1.5배(63/66) 키웠다. */
  const initialFrontSize = fitAspectSize(sourceAspect, 42 * 1.5, 44 * 1.5);
  /* 이 컴포넌트는 스캔/제품데이터/정렬확인 화면을 오갈 때 key 가 바뀌어
     통째로 리마운트된다(다른 이미지라 내부 DOM 을 새로 만드는 게 안전
     해서). 그때 Detail 확대 영역·배치를 빈 배열로 초기화해 버리면 방금
     만든 확대 영역이 사라지고, 위(부모)의 detailRegions 는 남아있는데
     layouts 만 리셋돼 배치가 안 맞아 엑셀 내보내기가 자동배치(작게,
     구석)로 떨어진다. 그래서 부모가 마지막으로 알고 있던 값을 초기값
     으로 그대로 물려받는다. */
  const [regions, setRegions] = useState<DetailRegion[]>(() => initialRegions ?? []);
  /* 엑셀 내보내기가 Detail 영역을 알아야 해서 위로 올려 준다. */
  useEffect(() => { onRegionsChange?.(regions); }, [regions, onRegionsChange]);
  const [layouts, setLayouts] = useState<SheetLayout[]>(() => {
    if (!initialLayouts || initialLayouts.length === 0) {
      return [{ id: 'front', kind: 'front', x: 4, y: 7, ...initialFrontSize }];
    }
    /* 90도 회전은 frameWidth/frameHeight 를 맞바꿔 sourceAspect 를 뒤집는데,
       이어받은 정면도 창은 이전(회전 전) 비율로 맞춰진 w/h 라 화면에는 옛
       비율 그대로 나온다 -- 사용자가 크기조절 손잡이를 한 번 눌러야("어떤
       값으로든 다시 계산해라"라는 신호) 그제서야 새 비율로 맞춰지는 게 그
       증상이었다. fitAspectSize 는 어느 쪽이 상한에 걸리든 항상
       w/h = sourceAspect/SHEET_ASPECT 를 만족하는 값을 낸다 -- 그러니
       지금 w/h 비율이 그 식과 맞으면(스캔/제품데이터 전환처럼 방향은 안
       바뀐 리마운트) 사용자가 손으로 맞춘 크기이므로 그대로 두고, 안
       맞으면(90/270 회전으로 실제로 뒤집힌 경우) 넓이(w*h)는 그대로 두고
       비율만 새로 맞춘다. [버그였던 부분] 가로/세로 상한을 맞바꿔
       fitAspectSize 를 다시 돌리는 방식은 상한 중 좁은 쪽에 맞춰
       매번 넓이가 줄어, 90도 회전을 반복할 때마다 창이 계속 작아졌다
       (한 바퀴 4번 돌리면 넓이가 1/4로). 넓이를 고정하고 비율만 풀면
       (w=√(넓이×비율), h=√(넓이/비율)) 몇 번을 돌려도 크기가 그대로다. */
    return initialLayouts.map((layout) => {
      if (layout.kind !== 'front' || layout.h <= 0) return layout;
      const expectedRatio = sourceAspect / SHEET_ASPECT;
      const actualRatio = layout.w / layout.h;
      if (Math.abs(actualRatio / expectedRatio - 1) < 0.05) return layout;
      const area = layout.w * layout.h;
      return { ...layout, w: Math.sqrt(area * expectedRatio), h: Math.sqrt(area / expectedRatio) };
    });
  });
  useEffect(() => { onLayoutsChange?.(layouts); }, [layouts, onLayoutsChange]);
  /* layout.id 별로 CorrectionPoints 에 넘길 콜백을 캐싱해 참조를 고정한다.
     .map() 렌더 안에서 매번 새 화살표 함수를 만들면, 그게 CorrectionPoints
     의 useEffect 의존성이라 매 렌더마다 그 이펙트가 다시 돌고, 그때마다
     새 positions 객체로 부모 state 를 갱신해 부모가 다시 렌더되고, 그
     리렌더가 다시 새 화살표 함수를 만드는 무한 루프가 된다. */
  const labelPositionHandlers = useRef<Map<string, (positions: Record<string, { x: number; y: number }>, layerSize?: { width: number; height: number }) => void>>(new Map());
  useEffect(() => { labelPositionHandlers.current.clear(); }, [onLabelPositionsChange]);
  const getLabelPositionsHandler = (layoutId: string) => {
    if (!onLabelPositionsChange) return undefined;
    const cache = labelPositionHandlers.current;
    let handler = cache.get(layoutId);
    if (!handler) {
      handler = (positions: Record<string, { x: number; y: number }>, layerSize?: { width: number; height: number }) => onLabelPositionsChange(layoutId, positions, layerSize);
      cache.set(layoutId, handler);
    }
    return handler;
  };
  const layerSizeHandlers = useRef<Map<string, (size: { width: number; height: number }) => void>>(new Map());
  useEffect(() => { layerSizeHandlers.current.clear(); }, [onLayerSizeChange]);
  const getLayerSizeHandler = (layoutId: string) => {
    if (!onLayerSizeChange) return undefined;
    const cache = layerSizeHandlers.current;
    let handler = cache.get(layoutId);
    if (!handler) {
      handler = (size: { width: number; height: number }) => onLayerSizeChange(layoutId, size);
      cache.set(layoutId, handler);
    }
    return handler;
  };
  const [hiddenDetailPointIds, setHiddenDetailPointIds] = useState<Record<string, Set<string>>>({});
  const [selectedLayoutId, setSelectedLayoutId] = useState('front');
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const updateLayout = (next: SheetLayout) => setLayouts((current) => current.map((layout) => layout.id === next.id ? next : layout));
  const createDetail = (region: DetailRegion) => {
    const detailCount = layouts.filter((layout) => layout.kind === 'detail').length;
    const detailAspect = region.w * frameWidth / (region.h * frameHeight);
    const nextLayout: SheetLayout = { id: `layout-${region.id}`, kind: 'detail', regionId: region.id, x: 68, y: 7 + (detailCount % 3) * 29, ...fitAspectSize(detailAspect, 28, 25) };
    setRegions((current) => [...current, region]); setLayouts((current) => [...current, nextLayout]);
    setSelectedRegionId(region.id); setSelectedLayoutId(nextLayout.id); setDetailMode(false);
  };
  const deleteDetail = (regionId: string) => {
    const targetLayout = layouts.find((layout) => layout.regionId === regionId);
    setRegions((current) => current.filter((region) => region.id !== regionId));
    setLayouts((current) => current.filter((layout) => layout.regionId !== regionId));
    if (targetLayout) setHiddenDetailPointIds((current) => { const next = { ...current }; delete next[targetLayout.id]; return next; });
    setSelectedRegionId(null); setSelectedLayoutId('front');
  };
  const selectedLayout = layouts.find((layout) => layout.id === selectedLayoutId) || layouts[0];
  const aspectFor = (layout: SheetLayout) => {
    const region = layout.regionId ? regions.find((item) => item.id === layout.regionId) : undefined;
    return region ? region.w * frameWidth / (region.h * frameHeight) : sourceAspect;
  };
  const setSelectedSize = (key: 'w' | 'h', value: number) => {
    if (!selectedLayout) return;
    const aspect = aspectFor(selectedLayout);
    const minW = Math.max(MIN_LAYOUT_SIZE, MIN_LAYOUT_SIZE * aspect / SHEET_ASPECT); const maxW = Math.min(100, 100 * aspect / SHEET_ASPECT);
    const minH = Math.max(MIN_LAYOUT_SIZE, MIN_LAYOUT_SIZE * SHEET_ASPECT / aspect); const maxH = Math.min(100, 100 * SHEET_ASPECT / aspect);
    const w = key === 'w' ? clamp(value, minW, maxW) : clamp(value, minH, maxH) * aspect / SHEET_ASPECT;
    const h = key === 'h' ? clamp(value, minH, maxH) : clamp(value, minW, maxW) * SHEET_ASPECT / aspect;
    updateLayout(normalizeBox({ ...selectedLayout, w, h }, 0));
  };
  const updateDetailRegion = (next: DetailRegion) => {
    setRegions((current) => current.map((item) => item.id === next.id ? next : item));
    const nextAspect = next.w * frameWidth / (next.h * frameHeight);
    setLayouts((current) => current.map((layout) => layout.regionId === next.id ? normalizeBox({ ...layout, ...fitAspectSize(nextAspect, layout.w, 100) }, 0) : layout));
  };

  return <div className={`sheet-canvas ${detailMode ? 'sheet-canvas--detail-mode' : ''}`} onPointerDown={(event) => { if (event.target === event.currentTarget) { setSelectedLayoutId(''); setSelectedRegionId(null); setSelectedAnnotationId(null); } }}>
    {layouts.map((layout) => {
      const region = layout.regionId ? regions.find((item) => item.id === layout.regionId) : undefined;
      if (layout.kind === 'detail' && !region) return null;
      const title = layout.kind === 'front' ? '정면도 · FRONT VIEW' : region!.label;
      const imageAspect = region ? region.w * frameWidth / (region.h * frameHeight) : sourceAspect;
      const detailPoints = region ? points.filter((point) => point.x >= region.x && point.x <= region.x + region.w && point.y >= region.y && point.y <= region.y + region.h).map((point) => ({ ...point, x: (point.x - region.x) / region.w * 100, y: (point.y - region.y) / region.h * 100 })) : points;
      /* 상세도도 정면도의 표시 집합을 상속한다. 주요 포인트가 아닌 항목이
         정면도에서는 숨고 Detail View에서 다시 나타나는 일을 막는다. */
      const layoutVisiblePointIds = layout.kind === 'front' ? visiblePointIds : new Set(detailPoints.filter((point) => visiblePointIds.has(point.id) && !hiddenDetailPointIds[layout.id]?.has(point.id)).map((point) => point.id));
      const toggleLayoutPoint = layout.kind === 'front' ? onPointToggle : (id: string) => setHiddenDetailPointIds((current) => { const hidden = new Set(current[layout.id] || []); if (hidden.has(id)) hidden.delete(id); else hidden.add(id); return { ...current, [layout.id]: hidden }; });
      const applyAreaPoints = (ids: string[], mode: 'hide' | 'show') => {
        if (layout.kind === 'front') ids.filter((id) => mode === 'hide' ? layoutVisiblePointIds.has(id) : !layoutVisiblePointIds.has(id)).forEach(onPointToggle);
        else setHiddenDetailPointIds((current) => { const hidden = new Set(current[layout.id] || []); ids.forEach((id) => mode === 'hide' ? hidden.add(id) : hidden.delete(id)); return { ...current, [layout.id]: hidden }; });
      };
      return <SheetLayoutFrame key={layout.id} layout={layout} imageAspect={imageAspect} selected={selectedLayoutId === layout.id} onSelect={() => setSelectedLayoutId(layout.id)} onChange={updateLayout} onDelete={region ? () => deleteDetail(region.id) : undefined} title={title}>
        {region ? <div className="detail-crop"><div className="layout-image-clip"><img src={imageUrl} alt={`${region.label} 확대 정면도`} style={{ width: `${10000 / region.w}%`, height: `${10000 / region.h}%`, left: `${-region.x / region.w * 100}%`, top: `${-region.y / region.h * 100}%` }} />{showZero && zeroLines.length > 0 && <ZeroLineOverlay lines={zeroLines} splineSegments={zeroSplineSegments} region={region} />}</div>{showPoints && <CorrectionPoints coefficient={coefficient} points={detailPoints} visibleLabelIds={layoutVisiblePointIds} onLabelToggle={toggleLayoutPoint} overrides={pointOverrides} onOverrideChange={onOverrideChange} labelFontFamily={labelFontFamily} initialLabelPositions={initialLabelPositionsByLayout?.[layout.id]} onLabelPositionsChange={getLabelPositionsHandler(layout.id)} onLayerSizeChange={getLayerSizeHandler(layout.id)} />}</div>
          : <div className="front-view-layout"><img src={imageUrl} alt="스캔 데이터에서 추출한 정면도" />{showZero && zeroLines.length > 0 && <ZeroLineOverlay lines={zeroLines} splineSegments={zeroSplineSegments} editable={zeroEditable} addPointMode={zeroPointAddMode} deletePointMode={zeroPointDeleteMode} onPointMove={onZeroPointMove} onSegmentDoubleClick={onZeroSegmentDoubleClick} onPointAdd={onZeroPointAdd} onPointDelete={onZeroPointDelete} />}{addPointMode && layout.kind === 'front' && <><div className="add-point-catcher" onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); const rect = event.currentTarget.getBoundingClientRect(); if (!rect.width || !rect.height) return; onAddPointAt((event.clientX - rect.left) / rect.width * 100, (event.clientY - rect.top) / rect.height * 100); }} />
          {/* 지우기는 거리 판정 대신 포인트 위 전용 버튼으로 받는다. 점이 작아 손으로 정확히 겨누기 어렵다. */}
          {addedPoints.map((added) => <button key={added.id} type="button" className="add-point-remove" style={{ left: `${added.x}%`, top: `${added.y}%` }}
            onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); onRemoveAddedPoint(added.id); }}
            aria-label={`${added.id} 추가 포인트 삭제`} title="이 추가 포인트 삭제"><X size={9} /></button>)}</>}{showPoints && <CorrectionPoints coefficient={coefficient} points={points} visibleLabelIds={layoutVisiblePointIds} onLabelToggle={toggleLayoutPoint} overrides={pointOverrides} onOverrideChange={onOverrideChange} labelFontFamily={labelFontFamily} initialLabelPositions={initialLabelPositionsByLayout?.[layout.id]} rotationSeed={layout.kind === 'front' ? frontRotationSeed : undefined} onLabelPositionsChange={getLabelPositionsHandler(layout.id)} onLayerSizeChange={getLayerSizeHandler(layout.id)} />}<DetailRegionLayer regions={regions} active={detailMode} selectedId={selectedRegionId} onSelect={setSelectedRegionId} onCreate={createDetail} onChange={updateDetailRegion} onDelete={deleteDetail} /></div>}
        <LabelAreaSelector mode={labelAreaMode} points={detailPoints} onApply={applyAreaPoints} onComplete={() => setLabelAreaMode(null)} />
      </SheetLayoutFrame>;
    })}
    {showAnnotations && !detailMode && !labelAreaMode && <AnnotationLayer annotations={annotations} tool={annotationTool} setTool={setAnnotationTool} selectedId={selectedAnnotationId} onSelect={setSelectedAnnotationId} onCommit={onAnnotationCommit} onCreate={onAnnotationCreate} onDelete={onAnnotationDelete} />}
    {detailMode && <div className="detail-mode-guide"><ZoomIn size={14} /><span>정면도 위에서 확대할 영역을 드래그하세요.</span><button type="button" onClick={() => setDetailMode(false)}>취소</button></div>}
    {addPointMode && <div className="detail-mode-guide add-point-guide"><Crosshair size={14} /><span>{sampleError || (sampling ? '편차값을 읽는 중입니다…' : '정면도를 눌러 보정 포인트를 추가합니다. 값은 히트맵 색에서 추정하며, 추가한 포인트를 다시 누르면 지워집니다.')}</span></div>}
    {labelAreaMode && <div className={`detail-mode-guide label-area-guide label-area-guide--${labelAreaMode}`}>{labelAreaMode === 'hide' ? <EyeOff size={14} /> : <Eye size={14} />}<span>레이아웃 위에서 {labelAreaMode === 'hide' ? '숨길' : '표시할'} 라벨 영역을 드래그하세요.</span><button type="button" onClick={() => setLabelAreaMode(null)}>취소</button></div>}
    {selectedLayout && <div className="layout-size-control" onPointerDown={(event) => event.stopPropagation()}><b>{selectedLayout.kind === 'front' ? '정면도' : regions.find((item) => item.id === selectedLayout.regionId)?.label} 크기 · 비율 고정</b><label>W <input type="range" min="5" max="100" value={selectedLayout.w} onChange={(event) => setSelectedSize('w', Number(event.target.value))} /><span>{Math.round(selectedLayout.w)}%</span></label><label>H <input type="range" min="5" max="100" value={selectedLayout.h} onChange={(event) => setSelectedSize('h', Number(event.target.value))} /><span>{Math.round(selectedLayout.h)}%</span></label></div>}
  </div>;
}

function Sidebar({ view, setView, collapsed, setCollapsed, hasResult }: { view: View; setView: (view: View) => void; collapsed: boolean; setCollapsed: (value: boolean) => void; hasResult: boolean }) {
  const items = [
    { id: 'workspace' as const, label: '엔진 결과', icon: BarChart3 },
    { id: 'service' as const, label: '보정 시트', icon: Layers3 },
    { id: 'cad' as const, label: '3D CAD 뷰어', icon: Layers3 },
    { id: 'files' as const, label: '품번 파일 정리', icon: Files, separated: true },
  ];
  return <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
    <div className="brand"><span className="brand__logo"><span className="brand__positive-symbol" aria-hidden="true" /><img className="brand__wordmark" src="/ajin-industrial-logo.png" alt="아진산업" /></span></div>
    <nav className="sidebar__nav" aria-label="주 메뉴"><span className="sidebar__eyebrow">WORKSPACE</span>{items.map((item) => { const Icon = item.icon; const disabled = item.id === 'service' && !hasResult; const active = item.id === 'workspace' ? view === 'workspace' || view === 'results' : view === item.id; return <button key={item.id} disabled={disabled} onClick={() => !disabled && setView(item.id)} className={`${active ? 'active' : ''}${item.separated ? ' sidebar__nav-item--separated' : ''}`}><Icon size={19} /><span>{item.label}</span></button>; })}</nav>
    <button className="sidebar__collapse" onClick={() => setCollapsed(!collapsed)} aria-label="사이드바 접기"><PanelLeftClose size={18} /><span>메뉴 접기</span></button>
  </aside>;
}

function AnalysisTabs({ active, result, scanReady, onSelect }: { active: AnalysisStep; result?: AnalysisResult; scanReady: boolean; onSelect: (step: AnalysisStep) => void }) {
  return <div className="result-tabs" role="tablist">{analysisStepMeta.map((step, index) => {
    const engine = step.key === 'scan' ? null : step.key;
    const failed = engine ? Boolean(result?.errors[engine]) : false;
    const available = step.key === 'scan' || Boolean(result);
    const complete = step.key === 'scan' ? scanReady : Boolean(result) && !failed;
    return <button role="tab" aria-selected={active === step.key} className={active === step.key ? 'active' : ''} onClick={() => onSelect(step.key)} disabled={!available} key={step.key}><span style={{ color: failed ? '#bd4650' : step.color }}>0{index + 1}</span><div><b>{step.name}</b>{failed && <small>실행 오류</small>}</div>{complete && <Check size={17} />}</button>;
  })}</div>;
}

function Header({ scans, activeId, setActiveId }: { scans: ScanItem[]; activeId?: string; setActiveId: (id: string) => void; onSaveFile: () => void; onLoadFile: (file: File) => void; onReset: () => void; note?: string | null }) {
  return <header className="topbar"><div><span className="topbar__context">AJIN INDUSTRIAL · DIE ENGINEERING</span></div><div className="topbar__actions">
    <label className="item-select"><span>현재 품번</span><select value={activeId || ''} disabled={!scans.length} onChange={(e) => setActiveId(e.target.value)}><option value="">등록된 이미지 없음</option>{scans.map((scan) => <option value={scan.id} key={scan.id}>{scan.partNo} · {scan.name}</option>)}</select></label>
    <div className="profile"><span>KJ</span><div><b>금형생산팀</b><small>관리자</small></div></div>
  </div></header>;
}

/* 파일명 어디에 있든 회사 표준 전체 품번(예: 67XX6-DR000)을 뽑는다.
   프런트에서 67XX6까지만 잘라 서버로 보내면, 전체 품번을 요구하는 등록
   API가 "품번 형식이 올바르지 않습니다"로 거부하므로 백엔드와 동일한
   정규식으로 유지한다. */
function partNoFromName(name: string): string | null {
  return name.toUpperCase().match(/[0-9A-Z]{5}-[A-Z]{2}[0-9]{3}/)?.[0] || null;
}

/* 컬러바 범위가 등록된 품번. 백엔드 PRODUCT_COLORBAR_MM 과 같은 표이며
   분석 응답의 knownParts 로도 확인할 수 있다. */
const KNOWN_PARTS = ['64XX2', '67XX6', '71XX2'];

function Workspace({ scans, selectedScan, setScans, result, onOpenResults, onOpenEngine, backendOnline }: { scans: ScanItem[]; selectedScan?: ScanItem; setScans: React.Dispatch<React.SetStateAction<ScanItem[]>>; result?: AnalysisResult; onOpenResults: (id: string) => void; onOpenEngine: (engine: Engine) => void; backendOnline: boolean | null }) {
  const [dragging, setDragging] = useState(false);
  const analyzingCount = scans.filter((scan) => scan.status === 'analyzing').length;
  const previewScan = selectedScan || scans[0];
  const addFiles = (files: FileList | File[]) => {
    const accepted = Array.from(files).filter((file) => file.type.startsWith('image/'));
    const next = accepted.map((file, index): ScanItem => ({
      id: `${file.name}-${file.lastModified}-${crypto.randomUUID()}`,
      name: file.name,
      partNo: partNoFromName(file.name) || `NEW-${String(scans.length + index + 1).padStart(2, '0')}`,
      size: `${(file.size / 1024 / 1024).toFixed(1)} MB`, url: URL.createObjectURL(file), file,
      status: 'ready', tone: (scans.length + index) % 3,
    }));
    setScans((current) => [...current, ...next]);
  };
  const analyzeAll = async () => {
    const targets = scans.filter((scan) => scan.status === 'ready' || scan.status === 'error');
    for (const target of targets) {
      setScans((current) => current.map((scan) => scan.id === target.id ? { ...scan, status: 'analyzing', error: undefined } : scan));
      try {
        const form = new FormData(); form.append('file', target.file, target.name);
        /* 제품데이터를 직접 붙였으면 함께 보낸다. 안 붙였으면 서버가 품번으로 등록분을 찾는다. */
        if (target.productFile) form.append('product', target.productFile, target.productFile.name);
        const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
        const data = await response.json() as AnalysisResult & { error?: string };
        if (!response.ok) throw new Error(data.error || '분석 중 오류가 발생했습니다.');
        setScans((current) => current.map((scan) => scan.id === target.id ? { ...scan, status: 'done', result: data } : scan));
      } catch (error) {
        const message = error instanceof Error ? error.message : '분석 서버에 연결할 수 없습니다.';
        setScans((current) => current.map((scan) => scan.id === target.id ? { ...scan, status: 'error', error: message } : scan));
      }
    }
  };
  const removeScan = (id: string) => setScans((current) => { const target = current.find((item) => item.id === id); if (target) { URL.revokeObjectURL(target.url); if (target.productUrl) URL.revokeObjectURL(target.productUrl); } return current.filter((item) => item.id !== id); });
  /* 제품데이터는 품번당 한 장이라 보통은 서버에 등록된 걸 자동으로 쓴다. 아직 등록이
     없는 품번만 여기서 직접 붙여 주면 되고, 붙인 뒤에는 서버가 등록해 다음부터 자동이다. */
  const attachReferenceFiles = async (id: string, files: FileList | File[]) => {
    const selected = Array.from(files);
    const image = selected.find((file) => file.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name));
    const cadFiles = selected.filter((file) => /\.(catpart|catproduct|step|stp|stl)$/i.test(file.name));
    const scan = scans.find((item) => item.id === id);
    if (!scan) return;
    if (image) setScans((current) => current.map((item) => {
      if (item.id !== id) return item;
      if (item.productUrl) URL.revokeObjectURL(item.productUrl);
      return { ...item, productFile: image, productUrl: URL.createObjectURL(image), status: item.status === 'done' ? 'ready' : item.status, assetError: undefined };
    }));
    if (!cadFiles.length) return;
    setScans((current) => current.map((item) => item.id === id ? {
      ...item,
      status: item.status === 'done' ? 'ready' : item.status,
      cadUploading: true,
      assetError: undefined,
      assetStatus: 'CAD 업로드 준비 중…',
    } : item));
    for (const file of cadFiles) {
      try {
        const partNumber = partNoFromName(scan.partNo) || partNoFromName(file.name);
        if (!partNumber) throw new Error('스캔 또는 CAD 파일명에서 전체 품번(예: 64XX2-DR000)을 찾지 못했습니다.');
        setScans((current) => current.map((item) => item.id === id ? { ...item, assetStatus: `${file.name} 등록·변환 확인 중…` } : item));
        const form = new FormData();
        form.append('file', file, file.name);
        form.append('partNumber', partNumber);
        const response = await fetch(`${API_BASE}/api/mesh`, { method: 'POST', body: form });
        const data = await response.json() as { error?: string; message?: string };
        if (!response.ok) throw new Error(data.error || `${file.name} 등록 실패`);
        setScans((current) => current.map((item) => item.id === id ? { ...item, assetError: undefined, assetStatus: data.message || `${file.name} 등록 완료` } : item));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setScans((current) => current.map((item) => item.id === id ? { ...item, assetError: message, assetStatus: undefined } : item));
      }
    }
    /* 라이브러리 저장이나 파일명 판정 실패가 뷰어 추가를 막아서는 안 된다.
       형상은 먼저 열고 실제 정합 결과로 맞는 파일인지 판단한다. */
    setScans((current) => current.map((item) => item.id === id
      ? { ...item, cadUploading: false, cadFiles: [...(item.cadFiles || []).filter((old) => !cadFiles.some((file) => file.name === old.name)), ...cadFiles] }
      : item));
  };
  const detachProduct = (id: string) => setScans((current) => current.map((scan) => {
    if (scan.id !== id) return scan;
    if (scan.productUrl) URL.revokeObjectURL(scan.productUrl);
      return { ...scan, productFile: undefined, productUrl: undefined };
  }));
  return <section className="page page--workspace">
    <div className="page-heading"><div><h2>3D 스캔 데이터 분석</h2></div></div>
    <AnalysisTabs active="scan" result={result} scanReady={scans.length > 0} onSelect={(step) => step !== 'scan' && onOpenEngine(step)} />
    <div className="workspace-grid">
      <div className="scan-data-preview card">
        <div className="viewer-toolbar"><div><b>{previewScan?.name || '업로드된 자료가 없습니다.'}</b></div>{previewScan && <span className="count-chip">{previewScan.partNo}</span>}</div>
        <div className="scan-data-preview__stage">{previewScan ? <img src={previewScan.url} alt={`${previewScan.name} 원본 스캔 데이터`} /> : <div className="scan-data-preview__empty"><ImageIcon size={34} /></div>}</div>
      </div>
      <div className="upload-panel card">
        <div className="card-title"><div><h3>파일 업로드</h3><p>PNG, JPG, WEBP</p></div><span className="count-chip">{scans.length}개 등록</span></div>
        <label className={`dropzone ${dragging ? 'dropzone--active' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(e: DragEvent<HTMLLabelElement>) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}>
          <input type="file" multiple accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff" onChange={(e: ChangeEvent<HTMLInputElement>) => e.target.files && addFiles(e.target.files)} />
          <span className="dropzone__icon"><UploadCloud size={29} /></span><b>파일 업로드</b>
        </label>
        <div className="file-list">
          <div className="file-list__head"><span>업로드된 파일</span></div>
          {!scans.length && <div className="empty-file-list">아직 업로드된 파일이 없습니다.</div>}
          {scans.map((scan) => <div className="file-row" key={scan.id}>
            <div className={`file-thumb tone-${scan.tone}`}><img src={scan.url} alt="" /></div>
            <div className="file-row__name">
              <b>{scan.name}</b><span>{scan.partNo} · {scan.error || scan.size}</span>
            </div>
            <div className="file-row__actions">
              {scan.status === 'done'
                ? <button type="button" className="status status--done status--result" onClick={() => onOpenResults(scan.id)} title="분석 결과 보기"><Check size={13} /> 분석 완료</button>
                : <span className={`status status--${scan.status}`}>{scan.status === 'analyzing' ? <><Activity size={13} /> 분석 중</> : scan.status === 'error' ? '오류' : '대기'}</span>}
              <label className={`cad-upload-action${scan.cadFiles?.length ? ' cad-upload-action--done' : ''}${scan.assetError ? ' cad-upload-action--error' : ''}`} title={scan.assetError ? `CAD 등록 오류: ${scan.assetError}` : scan.cadFiles?.length ? scan.cadFiles.map((file) => file.name).join(', ') : '이 스캔과 연결할 CAD 파일을 등록합니다'}>
                <input type="file" multiple accept=".catpart,.catproduct,.step,.stp,.stl" disabled={scan.cadUploading} onChange={(event: ChangeEvent<HTMLInputElement>) => { if (event.target.files?.length) void attachReferenceFiles(scan.id, event.target.files); event.currentTarget.value = ''; }} />
                <Layers3 size={13} /> CAD 파일 업로드
                {scan.cadUploading ? <Activity className="cad-upload-action__progress" size={13} /> : scan.cadFiles?.length ? <Check className="cad-upload-action__check" size={14} strokeWidth={3} /> : null}
              </label>
              <button className="icon-button icon-button--small upload-cancel-button" onClick={() => removeScan(scan.id)} aria-label={`${scan.name} 업로드 취소`} title="업로드 취소"><X size={15} /></button>
            </div>
          </div>)}
        </div>
        <button className="primary-button primary-button--wide" onClick={analyzeAll} disabled={!backendOnline || analyzingCount > 0 || !scans.some((scan) => scan.status === 'ready' || scan.status === 'error')}><Play size={17} fill="currentColor" /> {analyzingCount ? `${analyzingCount}개 이미지 분석 중` : backendOnline === false ? '로컬 엔진 서버 연결 필요' : '3D 스캔 데이터 분석'}<ArrowRight size={18} /></button>
      </div>
    </div>
  </section>;
}

function engineSummary(engine: Engine, result: AnalysisResult) {
  if (result.errors[engine]) return { stat: '실패', detail: result.errors[engine] || '엔진 오류' };
  if (engine === 'label') return { stat: `${result.stats.labelsRemoved}개`, detail: '검출된 라벨 제거 및 주변 색상 복원 완료' };
  if (engine === 'deviation') {
    const detected = result.stats.detectedCandidates ?? result.stats.pointsDetected;
    const connected = result.stats.validCandidates ?? result.stats.pointsDetected;
    return {
      stat: `${result.stats.pointsDetected}개`,
      detail: `라벨 후보 ${detected}개 · 스캔 연결 ${connected}개 · Qwen 실제 판독 ${result.stats.qwenReads}개`,
    };
  }
  return { stat: `${result.stats.zeroRegions}개`, detail: `부품 면적의 ${(result.stats.zeroRatio * 100).toFixed(1)}% · 실제 검출 결과` };
}

/* 방향 판정은 상하좌우가 대칭인 부품에서는 갈리지 않는다. 그래서 근거 수치와 반전
   버튼을 함께 두고, 사람이 확정한 방향만 품번에 저장해 다음 스캔부터 다시 묻지 않는다. */
function AlignmentBar({ alignment, partNumber, source, transferred, total, busy, confirmed, onFlip, onConfirm }: { alignment: AlignmentInfo; partNumber: string | null; source: string | null; transferred: number; total: number; busy: boolean; confirmed: boolean; onFlip?: (flipX?: boolean, flipY?: boolean, rotation?: number) => void; onConfirm?: () => void }) {
  return null;
  /* 정렬은 엔진이 자동 결정하며 일반 사용자 화면에는 조작 옵션을 노출하지 않는다. */
  const trusted = alignment.confident;
  return <div className={`alignment-bar ${trusted ? '' : 'alignment-bar--check'}`}>
    <span className="alignment-bar__state">{trusted ? <><ShieldCheck size={14} /> 자동 판정 신뢰 가능</> : <><MoveRight size={14} /> 방향 확인 필요</>}</span>
    <span className="alignment-bar__facts"><b>{partNumber || '품번 미확인'}</b><small>{source || '제품데이터 없음'}</small><small>외형 {(alignment.outlineIou * 100).toFixed(1)}% · 구멍 {(alignment.holeIou * 100).toFixed(1)}% · 2위와 격차 {alignment.margin.toFixed(3)}</small><small>전사 {transferred}/{total}개</small></span>
    {onFlip && <span className="alignment-bar__actions">{/* 분석 결과는 화면에 남아 있으므로, 엔진이 바뀌면 좌표만 다시 받아 온다. Qwen 판독은 다시 하지 않는다. */}<button type="button" disabled={busy} onClick={() => onFlip?.()} title="정렬만 다시 계산합니다. 방향은 자동 판정과 확정 저장분을 따릅니다">정렬 다시 계산</button><button type="button" disabled={busy} onClick={() => onFlip?.(alignment.flipX, alignment.flipY, alignment.rotation === 90 ? 0 : 90)}>90° 회전</button><button type="button" disabled={busy} onClick={() => onFlip?.(!alignment.flipX, alignment.flipY, alignment.rotation ?? 0)}>좌우 뒤집기</button><button type="button" disabled={busy} onClick={() => onFlip?.(alignment.flipX, !alignment.flipY, alignment.rotation ?? 0)}>상하 뒤집기</button>{onConfirm && <button type="button" className="primary" disabled={busy || confirmed || !partNumber} onClick={onConfirm}>{confirmed ? <><Check size={13} /> 품번에 저장됨</> : '이 방향으로 확정'}</button>}</span>}
  </div>;
}

function Results({ scan, engine, setEngine, onScanData, onService, hiddenPointIds, onPointToggle, keyPointsOnly, onKeyPointsOnlyChange, onRealign, onConfirmAlignment }: { scan: ScanItem; engine: Engine; setEngine: (engine: Engine) => void; onScanData: () => void; onService: () => void; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; keyPointsOnly: boolean; onKeyPointsOnlyChange: (value: boolean) => void; onRealign?: (flipX?: boolean, flipY?: boolean) => Promise<void>; onConfirmAlignment?: () => Promise<void> }) {
  /* 편차 뷰는 세 가지로 본다: 스캔 위, 제품데이터 위, 그리고 정렬 확인용 실루엣 겹침. */
  const [frame, setFrame] = useState<'scan' | 'product' | 'overlay'>('scan');
  const [busy, setBusy] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [hiddenZeroLineIds, setHiddenZeroLineIds] = useState<Set<string>>(new Set());
  useEffect(() => { setHiddenZeroLineIds(new Set()); }, [scan.id]);
  const result = scan.result!;
  const keyPointIds = new Set(result.keySelection?.ids ?? result.points.filter((point) => point.keyReasons?.length).map((point) => point.id));
  const hasKeySelection = result.keySelection !== undefined;
  const showKeyPointsOnly = keyPointsOnly && hasKeySelection;
  const displayedPoints = engine === 'deviation' && showKeyPointsOnly ? result.points.filter((point) => keyPointIds.has(point.id)) : result.points;
  const visibleLabelIds = new Set(displayedPoints.filter((point) => !hiddenPointIds.has(point.id)).map((point) => point.id));
  const alignment = result.alignment;
  const productReady = engine === 'deviation' && Boolean(result.productImage && alignment);
  const showFrame = productReady ? frame : 'scan';
  /* 제품데이터 뷰에서는 같은 포인트의 좌표만 제품 기준으로 바꿔 넘긴다. */
  const productPoints = result.points.filter((point) => point.xProduct !== undefined && point.yProduct !== undefined).map((point) => ({ ...point, x: point.xProduct!, y: point.yProduct! }));
  const displayedProductPoints = productPoints.filter((point) => !showKeyPointsOnly || keyPointIds.has(point.id));
  const zeroLines = (result.zeroLines ?? []).filter((line) => Array.isArray(line.points) && line.points.length >= 2);
  const visibleZeroLines = zeroLines.filter((line) => !hiddenZeroLineIds.has(String(line.id)));
  const hasZeroLineControls = engine === 'zero' && zeroLines.length > 0;
  const hasInspectionPanel = engine === 'label' || engine === 'deviation' || hasZeroLineControls;
  const toggleZeroLine = (id: number | string) => setHiddenZeroLineIds((current) => {
    const next = new Set(current); const key = String(id);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });
  const showReviewedCase1Overlay = engine === 'zero' && result.zeroCase === 1 && showFrame === 'scan' && Boolean(result.zeroOverlay);
  const image = showFrame === 'product' ? result.productImage : showFrame === 'overlay' ? result.alignmentOverlay : showReviewedCase1Overlay ? result.zeroOverlay : engine === 'zero' && hasZeroLineControls ? result.cleanImage || scan.url : engine === 'zero' ? result.zeroOverlay : result.cleanImage || scan.url;
  const frameWidth = showFrame === 'scan' || !alignment ? result.source.width : alignment.productSize[0];
  const frameHeight = showFrame === 'scan' || !alignment ? result.source.height : alignment.productSize[1];
  const toggleLabel = onPointToggle;
  const runRealign = async (flipX?: boolean, flipY?: boolean) => {
    if (!onRealign || busy) return;
    setBusy(true); setConfirmed(false);
    try { await onRealign(flipX, flipY); } catch { /* 연결 실패는 전역 오류창으로 전파하지 않는다. */ } finally { setBusy(false); }
  };
  const runConfirm = async () => {
    if (!onConfirmAlignment || busy) return;
    setBusy(true);
    try { await onConfirmAlignment(); setConfirmed(true); } finally { setBusy(false); }
  };
  return <section className={`page page--results page--results-${engine}`}><div className="page-heading page-heading--compact"><div><h2>3D 스캔 데이터 분석</h2></div><button className="primary-button" onClick={onService}>보정 시트 만들기 <ArrowRight size={17} /></button></div>
    <AnalysisTabs active={engine} result={result} scanReady onSelect={(step) => step === 'scan' ? onScanData() : setEngine(step)} />
    <div className={`results-layout ${hasInspectionPanel ? '' : 'results-layout--viewer-only'}`}>
      <div className="viewer-card card">
        <div className="viewer-toolbar">
          <div><b>{scan.name}</b></div>
          <div>
            {productReady && <div className="frame-toggles">{([['scan', '스캔 위'], ['product', '제품데이터 위'], ['overlay', '정렬 확인']] as const).map(([key, label]) => <button key={key} type="button" className={showFrame === key ? 'active' : ''} onClick={() => setFrame(key)}>{label}</button>)}</div>}
            {engine === 'deviation' && <div className={`point-filter-switch ${showKeyPointsOnly ? 'active' : ''}`}>
              <span>주요 포인트만 표시</span>
              <button type="button" role="switch" aria-checked={showKeyPointsOnly} aria-label="주요 포인트만 표시" disabled={!hasKeySelection} onClick={() => onKeyPointsOnlyChange(!keyPointsOnly)} title={hasKeySelection ? '국소 극값, 부호 변화, 전체 최대·최소 포인트만 표시합니다.' : '주요 포인트 정보가 없습니다. 이미지를 다시 분석해 주세요.'}><i /></button>
            </div>}
          </div>
        </div>
        {productReady && alignment && <AlignmentBar alignment={alignment} partNumber={result.partNumber} source={result.productSource} transferred={productPoints.length} total={result.points.length} busy={busy} confirmed={confirmed} onFlip={onRealign ? runRealign : undefined} onConfirm={onConfirmAlignment ? runConfirm : undefined} />}
        <div className={`viewer-stage ${engine === 'deviation' ? 'viewer-stage--light' : ''}`}>
          <Heatmap key={`${scan.id}-${engine}-${showFrame}`} imageUrl={image} width={frameWidth} height={frameHeight} lightBackground={engine === 'deviation'} containImage>
            {engine === 'deviation' && showFrame !== 'overlay' && <CorrectionPoints coefficient={-1} points={showFrame === 'product' ? displayedProductPoints : displayedPoints} visibleLabelIds={visibleLabelIds} onLabelToggle={toggleLabel} />}
            {engine === 'zero' && hasZeroLineControls && !showReviewedCase1Overlay && <ZeroLineLayer lines={visibleZeroLines} width={frameWidth} height={frameHeight} />}
          </Heatmap>
        </div>
      </div>
      {engine === 'label' && <aside className="inspection-panel">
        <div className="card label-removal-count">
          <span className="label-removal-count__icon"><Sparkles size={19} /></span>
          <div><span>제거된 라벨 영역</span><strong>{result.stats.labelsRemoved}<small>개</small></strong><p>라벨 제거 및 주변 색상 복원 완료</p></div>
        </div>
      </aside>}
      {engine === 'deviation' && <aside className="inspection-panel">
        <div className="card mini-table"><div className="card-title"><h3>검출 포인트</h3><span>{showKeyPointsOnly ? `주요 ${displayedPoints.length}/${result.points.length}` : `${visibleLabelIds.size}/${result.points.length}`}</span></div>{displayedPoints.map((point) => { const visible = visibleLabelIds.has(point.id); return <div className="point-list-row" key={point.id}><span>{point.id}</span><b className={point.value > 0 ? 'positive' : 'negative'}>{point.value > 0 ? '+' : ''}{point.value.toFixed(3)} mm</b><small>{point.xPx}, {point.yPx}</small><button type="button" className={visible ? 'label-visibility active' : 'label-visibility'} onClick={() => toggleLabel(point.id)} aria-label={`${point.id} 라벨 ${visible ? '숨기기' : '표시하기'}`} title={`라벨 ${visible ? 'OFF' : 'ON'}`}>{visible ? <Eye size={14} /> : <EyeOff size={14} />}</button></div>; })}{!displayedPoints.length && <p className="empty-mini">표시할 포인트가 없습니다.</p>}</div>
      </aside>}
      {hasZeroLineControls && <aside className="inspection-panel">
        <div className="card mini-table zero-line-list"><div className="card-title"><h3>추천 제로라인</h3><span>{visibleZeroLines.length}/{zeroLines.length}</span></div>{zeroLines.map((line, index) => { const visible = !hiddenZeroLineIds.has(String(line.id)); const label = `ZL-${String(index + 1).padStart(2, '0')}`; return <div className="zero-line-list-row" key={String(line.id)}><span><i className="zero-line-swatch" />{label}</span><small>{visible ? '표시' : '숨김'}</small><button type="button" className={visible ? 'label-visibility active' : 'label-visibility'} onClick={() => toggleZeroLine(line.id)} aria-label={`${label} ${visible ? '숨기기' : '표시하기'}`} title={visible ? '제로라인 숨기기' : '제로라인 표시하기'}>{visible ? <Eye size={14} /> : <EyeOff size={14} />}</button></div>; })}</div>
      </aside>}
    </div></section>;
}

function FolderTreeNode({ entry, selectedPath, onOpen }: { entry: FolderEntry; selectedPath: string; onOpen: (path: string) => void }) {
  const [expanded, setExpanded] = useState(false); const [children, setChildren] = useState<FolderEntry[]>([]); const [loaded, setLoaded] = useState(false);
  const toggle = async () => {
    if (!loaded) {
      const response = await fetch(`${API_BASE}/api/folders?path=${encodeURIComponent(entry.path)}`); const data = await response.json() as FolderResponse;
      if (response.ok) { setChildren((data.entries || []).filter((item: FolderEntry) => item.isDirectory)); setLoaded(true); }
    }
    setExpanded((value) => !value); onOpen(entry.path);
  };
  return <div className="tree-node"><button className={`tree-root ${selectedPath === entry.path ? 'selected' : ''}`} onClick={toggle}>{expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}<Folder size={17} fill="currentColor" /> <span>{entry.name}</span></button>{expanded && <div className="tree-children">{children.map((child) => <FolderTreeNode key={child.path} entry={child} selectedPath={selectedPath} onOpen={onOpen} />)}{loaded && !children.length && <span className="tree-empty">하위 폴더 없음</span>}</div>}</div>;
}

function Explorer() {
  const [available, setAvailable] = useState<boolean | null>(null); const [rootName, setRootName] = useState('품번별 폴더'); const [rootEntries, setRootEntries] = useState<FolderEntry[]>([]); const [entries, setEntries] = useState<FolderEntry[]>([]); const [path, setPath] = useState(''); const [query, setQuery] = useState('');
  const openFolder = async (nextPath: string) => {
    const response = await fetch(`${API_BASE}/api/folders?path=${encodeURIComponent(nextPath)}`); const data = await response.json() as FolderResponse;
    if (!response.ok || data.available === false) { setAvailable(false); return; }
    setAvailable(true); setRootName(data.rootName || '품번별 폴더'); setEntries(data.entries || []); setPath(data.path || '');
    if (!nextPath) setRootEntries((data.entries || []).filter((item: FolderEntry) => item.isDirectory));
  };
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/folders?path=`)
      .then((response) => response.json().then((data) => ({ ok: response.ok, data: data as FolderResponse })))
      .then(({ ok, data }) => {
        if (cancelled) return;
        if (!ok || data.available === false) { setAvailable(false); return; }
        const nextEntries = data.entries || [];
        setAvailable(true); setRootName(data.rootName || '품번별 폴더'); setEntries(nextEntries); setPath('');
        setRootEntries(nextEntries.filter((item: FolderEntry) => item.isDirectory));
      })
      .catch(() => { if (!cancelled) setAvailable(false); });
    return () => { cancelled = true; };
  }, []);
  if (available === false) return null;
  const filtered = entries.filter((entry) => entry.name.toLowerCase().includes(query.toLowerCase())); const segments = path ? path.split('/') : [];
  return <div className="explorer card"><div className="explorer__title"><div><FolderOpen size={20} /><b>실시간 품번별 폴더</b></div><span>{available == null ? '연결 확인 중' : '현재 PC 폴더와 연결됨'}</span></div><div className="explorer__bar"><div className="explorer__crumb"><button disabled={!path} onClick={() => openFolder(segments.slice(0, -1).join('/'))}><ArrowLeft size={14} /></button><span><button onClick={() => openFolder('')}>{rootName}</button>{segments.map((segment, index) => <span key={`${segment}-${index}`}><ChevronRight size={13} /><button onClick={() => openFolder(segments.slice(0, index + 1).join('/'))}>{segment}</button></span>)}</span></div><label><ZoomIn size={15} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="현재 폴더 검색" /></label></div><div className="explorer__body"><div className="folder-tree"><button className={`tree-root ${!path ? 'selected' : ''}`} onClick={() => openFolder('')}><ChevronDown size={15} /><FolderOpen size={17} /> <span>{rootName}</span></button><div className="tree-children">{rootEntries.map((entry) => <FolderTreeNode key={entry.path} entry={entry} selectedPath={path} onOpen={openFolder} />)}</div></div><div className="folder-content"><div className="folder-content__head"><span>이름</span><span>수정한 날짜</span><span>크기</span></div>{filtered.map((entry) => <button className="folder-row" key={entry.path} onDoubleClick={() => entry.isDirectory && openFolder(entry.path)} onClick={() => entry.isDirectory && openFolder(entry.path)}><span>{entry.isDirectory ? <Folder size={19} fill="currentColor" /> : <File size={18} />}{entry.name}</span><small>{new Date(entry.modified).toLocaleString('ko-KR')}</small><small>{entry.isDirectory ? '파일 폴더' : formatBytes(entry.size)}</small></button>)}{!filtered.length && <div className="empty-search">이 폴더는 비어 있습니다.</div>}<div className="folder-content__status">{filtered.length}개 항목 <span>·</span> 실시간 로컬 조회</div></div></div></div>;
}

function OrganizerFolderNode({ entry, onAssign }: { entry: FolderEntry; onAssign: (ids: string[], path: string) => void }) {
  const [expanded, setExpanded] = useState(false);
  const [children, setChildren] = useState<FolderEntry[]>([]);
  const [loaded, setLoaded] = useState(false);
  const toggle = async () => {
    if (!loaded) {
      const response = await fetch(`${API_BASE}/api/folders?path=${encodeURIComponent(entry.path)}`);
      const data = await response.json() as FolderResponse;
      if (response.ok) {
        setChildren(data.entries || []);
        setLoaded(true);
      }
    }
    setExpanded((current) => !current);
  };
  const acceptDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.currentTarget.classList.remove('drop-ready');
    try {
      const ids = JSON.parse(event.dataTransfer.getData('text/ajin-file-ids')) as string[];
      if (ids.length) onAssign(ids, entry.path);
    } catch { /* 외부 파일 드롭은 왼쪽 업로드 영역에서 처리한다. */ }
  };
  const folders = children.filter((child) => child.isDirectory);
  const files = children.filter((child) => !child.isDirectory);
  return <div className="organizer-folder-node">
    <button type="button" onClick={toggle} onDragOver={(event) => { event.preventDefault(); event.currentTarget.classList.add('drop-ready'); }} onDragLeave={(event) => event.currentTarget.classList.remove('drop-ready')} onDrop={acceptDrop}>
      {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}<Folder size={16} fill="currentColor" /><span>{entry.name}</span>
    </button>
    {expanded && <div>
      {folders.map((child) => <OrganizerFolderNode key={child.path} entry={child} onAssign={onAssign} />)}
      {files.map((file) => <div className="organizer-folder-file" key={file.path} title={file.name}><File size={13} /><span>{file.name}</span></div>)}
      {loaded && !children.length && <small>비어 있음</small>}
    </div>}
  </div>;
}

function FileOrganizerPage() {
  const [status, setStatus] = useState<FileOrganizerStatus | null>(null);
  const [items, setItems] = useState<FileOrganizerItem[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [rootEntries, setRootEntries] = useState<FolderEntry[]>([]);
  const [rootName, setRootName] = useState('품번별 폴더');
  const [dragging, setDragging] = useState(false);
  const [pendingUploadFiles, setPendingUploadFiles] = useState<File[]>([]);
  const [classifiedUploadNames, setClassifiedUploadNames] = useState<string[]>([]);
  const [classifyingFiles, setClassifyingFiles] = useState(false);
  const [busy, setBusy] = useState(false);
  const [operation, setOperation] = useState<'copy' | 'move'>('copy');
  const [conflict, setConflict] = useState<'rename' | 'skip' | 'overwrite'>('rename');
  const [notice, setNotice] = useState<{ tone: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [reconstructionReport, setReconstructionReport] = useState<{
    tone: 'success' | 'error' | 'info'; text: string;
    issues: { name: string; sourcePath: string; message: string }[];
  } | null>(null);
  const [uploadStorageReport, setUploadStorageReport] = useState<{
    tone: 'success' | 'error' | 'info'; text: string; files: UploadStorageResult[];
  } | null>(null);
  const [showDatabase, setShowDatabase] = useState(false);
  const [databaseUrl, setDatabaseUrl] = useState('');
  const [showFolderOrder, setShowFolderOrder] = useState(false);
  const [folderOrder, setFolderOrder] = useState<FolderAxis[]>(FOLDER_AXES);
  const [axisOptions, setAxisOptions] = useState<FolderAxisOption[]>([]);
  const [savingOrder, setSavingOrder] = useState(false);

  const axisLabel = (axis: FolderAxis) => axisOptions.find((option) => option.id === axis)?.label || FOLDER_AXIS_LABELS[axis];

  /* 구버전 백엔드가 'product' 같은 옛 축 이름을 돌려주면 화면이 빈 칸을 그리게 되므로,
     네 축이 한 번씩 다 들어온 응답만 받아들이고 아니면 기본 순서로 되돌린다. */
  const normalizeFolderOrder = (received: unknown): FolderAxis[] => {
    const axes = Array.isArray(received) ? received as FolderAxis[] : [];
    const valid = axes.length === FOLDER_AXES.length
      && FOLDER_AXES.every((axis) => axes.includes(axis));
    return valid ? axes : FOLDER_AXES;
  };

  const loadFolderOrder = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/folder-order`);
      const data = await response.json() as FolderOrderResponse;
      if (!response.ok) throw new Error(data.error || '폴더 순서를 불러오지 못했습니다.');
      setFolderOrder(normalizeFolderOrder(data.folderOrder));
      const validOptions = Array.isArray(data.axes)
        ? data.axes.filter((option) => FOLDER_AXES.includes(option.id))
        : [];
      setAxisOptions(validOptions.length === FOLDER_AXES.length
        ? validOptions
        : FOLDER_AXES.map((id) => ({ id, label: FOLDER_AXIS_LABELS[id] })));
    } catch { /* 상태 카드/기본 순서로 충분히 안내되므로 조용히 넘어간다. */ }
  }, []);

  const [pathsInfo, setPathsInfo] = useState<OrganizerPathsResponse | null>(null);
  const [selectingRoot, setSelectingRoot] = useState<'source' | 'destination' | null>(null);

  const loadOrganizerPaths = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/paths`);
      const data = await response.json() as OrganizerPathsResponse & { error?: string };
      if (!response.ok) throw new Error(data.error || '경로 설정을 불러오지 못했습니다.');
      setPathsInfo(data);
    } catch { /* 저장소 상태 카드에 이미 현재 경로가 표시되므로 조용히 넘어간다. */ }
  }, []);

  const chooseOrganizerFolder = async (which: 'source' | 'destination', purpose: 'configure' | 'existing' = 'configure') => {
    setSelectingRoot(which);
    setNotice(null);
    setUploadStorageReport(null);
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/select-folder`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ which, purpose }),
      });
      const data = await response.json() as OrganizerPathsResponse & { cancelled?: boolean; error?: string };
      if (!response.ok) throw new Error(data.error || '폴더를 선택하지 못했습니다.');
      if (data.cancelled) return;
      setReconstructionReport(null);
      setPathsInfo(data);
      setNotice({
        tone: 'success',
        text: which === 'source'
          ? '기존 폴더를 선택했습니다.'
          : purpose === 'existing'
            ? '재구성한 폴더를 선택했습니다.'
            : '재구성 폴더를 생성할 위치를 선택했습니다.',
      });
      await loadStatus(true); await loadFolders();
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : '폴더 선택 중 오류가 발생했습니다.' });
    } finally { setSelectingRoot(null); }
  };

  const openInExplorer = async (which: 'source' | 'destination') => {
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/reveal`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ which }),
      });
      const data = await response.json() as { error?: string };
      if (!response.ok) throw new Error(data.error || '탐색기를 열지 못했습니다.');
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : '탐색기를 여는 중 오류가 발생했습니다.' });
    }
  };

  const loadStatus = useCallback(async (checkDb = false) => {
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/status?checkDb=${checkDb ? '1' : '0'}`);
      const data = await response.json() as FileOrganizerStatus & { error?: string };
      if (!response.ok) throw new Error(data.error || '저장소 상태를 확인하지 못했습니다.');
      setStatus(data);
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : '로컬 백엔드에 연결할 수 없습니다.' });
    }
  }, []);

  const loadFolders = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/folders?path=`);
      const data = await response.json() as FolderResponse;
      if (!response.ok || data.available === false) return;
      setRootEntries(data.entries || []);
      setRootName(data.rootName || '품번별 폴더');
    } catch { /* 대상 경로가 아직 준비되지 않은 경우 상태 카드가 안내한다. */ }
  }, []);

  useEffect(() => { void loadStatus(true); void loadFolders(); void loadFolderOrder(); void loadOrganizerPaths(); }, [loadFolders, loadFolderOrder, loadOrganizerPaths, loadStatus]);

  const mergeItems = useCallback((incoming: FileOrganizerItem[]) => {
    setItems((current) => {
      const merged = new Map(current.map((item) => [item.sourcePath, item]));
      incoming.forEach((item) => merged.set(item.sourcePath, item));
      return Array.from(merged.values());
    });
    setSelected((current) => new Set([...current, ...incoming.map((item) => item.id)]));
  }, []);

  const scanSource = async (reconstruct = false) => {
    if (busy || selectingRoot !== null) return;
    setBusy(true); setNotice(null);
    let scannedItems: FileOrganizerItem[] = [];
    if (reconstruct) setReconstructionReport({ tone: 'info', text: '기존 폴더를 분석하고 있습니다.', issues: [] });
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/scan`);
      const data = await response.json() as { items?: FileOrganizerItem[]; error?: string };
      if (!response.ok) throw new Error(data.error || '원본 폴더를 스캔하지 못했습니다.');
      scannedItems = data.items || [];
      mergeItems(scannedItems);
      if (!reconstruct) {
        setNotice({ tone: 'info', text: `원본 폴더에서 ${scannedItems.length}개 파일을 분석했습니다.` });
        return;
      }
      if (!scannedItems.length) {
        setReconstructionReport({ tone: 'info', text: '기존 폴더에 재구성할 파일이 없습니다.', issues: [] });
        return;
      }
      setReconstructionReport({ tone: 'info', text: `${scannedItems.length}개 파일 분석 완료 · 재구성 폴더를 생성하고 있습니다.`, issues: [] });
      const executeResponse = await fetch(`${API_BASE}/api/file-organizer/execute`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          operation: 'copy', conflict,
          items: scannedItems.map((item) => ({ sourcePath: item.sourcePath, targetDir: item.targetDir })),
        }),
      });
      const executed = await executeResponse.json() as { results?: { source: string; destination?: string; status: string; message: string }[]; error?: string };
      if (!executeResponse.ok) throw new Error(executed.error || '재구성 폴더를 생성하지 못했습니다.');
      const results = executed.results || [];
      const successful = new Set(results.filter((result) => result.status === 'success').map((result) => result.source));
      const bySource = new Map(results.map((result) => [result.source, result]));
      const issues = scannedItems.flatMap((item) => {
        const result = bySource.get(item.sourcePath);
        const messages: string[] = [];
        if (!result) messages.push('처리 결과를 확인하지 못했습니다.');
        else if (result.status !== 'success') messages.push(
          `${result.status === 'skipped' ? '건너뜀' : '처리 실패'}: ${result.message || '파일을 복사하지 못했습니다.'}`,
        );
        const missing = [!item.itemNo && '품번', !item.customer && '차종', !item.categoryKey && '자료 유형'].filter(Boolean);
        if (!item.targetDir || missing.length) {
          messages.push(`분류 위치 확인 필요${missing.length ? ` (${missing.join(', ')} 인식 불가)` : ' (저장 위치를 찾지 못함)'}.`);
          if (result?.status === 'success') messages.push(`복사는 완료되었습니다. 저장 위치: ${result.destination || item.targetDir || '_미분류'}`);
        }
        return messages.length ? [{ name: item.name, sourcePath: item.sourcePath, message: messages.join(' ') }] : [];
      });
      const completedIds = new Set([...items, ...scannedItems].filter((item) => successful.has(item.sourcePath)).map((item) => item.id));
      setItems((current) => current.filter((item) => !successful.has(item.sourcePath)));
      setSelected((current) => new Set([...current].filter((id) => !completedIds.has(id))));
      setReconstructionReport({
        tone: issues.length ? 'error' : 'success',
        text: issues.length ? `${successful.size}개 파일 복사 완료 · ${issues.length}개 파일은 처리 결과 또는 분류 위치를 확인해 주세요.` : '정상적으로 처리가 되었습니다',
        issues,
      });
      await loadStatus(true); await loadFolders();
    } catch (error) {
      const text = error instanceof Error ? error.message : '폴더 분석·재구성 중 오류가 발생했습니다.';
      if (reconstruct) setReconstructionReport({
        tone: 'error', text,
        issues: scannedItems.map((item) => ({ name: item.name, sourcePath: item.sourcePath, message: '처리 완료 여부를 확인하지 못했습니다.' })),
      });
      else setNotice({ tone: 'error', text });
    } finally { setBusy(false); }
  };

  const uploadFiles = async (files: FileList | File[]) => {
    if (!files.length) return;
    const selectedFiles = Array.from(files);
    setClassifyingFiles(true);
    setBusy(true); setNotice(null);
    setUploadStorageReport({ tone: 'info', text: '파일을 분류하고 두 폴더에 저장하고 있습니다.', files: [] });
    try {
      const form = new FormData();
      selectedFiles.forEach((file) => form.append('files', file, file.name));
      form.append('organize', 'both');
      form.append('conflict', conflict);
      const response = await fetch(`${API_BASE}/api/file-organizer/upload`, { method: 'POST', body: form });
      const data = await response.json() as { items?: FileOrganizerItem[]; storageResults?: UploadStorageResult[]; error?: string };
      if (!response.ok) throw new Error(data.error || '파일을 등록하지 못했습니다.');
      const classifiedItems = data.items || [];
      const storageResults = data.storageResults || [];
      if (!storageResults.length) throw new Error('저장 결과를 받지 못했습니다. 백엔드를 다시 실행한 뒤 시도해 주세요.');
      const failedNames = new Set(storageResults
        .filter((result) => result.existing.status === 'error' || result.reconstructed.status === 'error')
        .map((result) => result.name));
      const completedNames = storageResults.filter((result) => !failedNames.has(result.name)).map((result) => result.name);
      const failedCount = failedNames.size;
      if (failedCount) mergeItems(classifiedItems.filter((item) => failedNames.has(item.name)));
      setClassifiedUploadNames(completedNames);
      setPendingUploadFiles(selectedFiles.filter((file) => failedNames.has(file.name)));
      setUploadStorageReport({
        tone: failedCount ? 'error' : 'success',
        text: failedCount
          ? `${storageResults.length - failedCount}개 파일 저장 완료 · ${failedCount}개 파일은 아래 결과를 확인해 주세요.`
          : `${storageResults.length}개 파일이 정상적으로 분류·저장되었습니다.`,
        files: storageResults,
      });
      setNotice({
        tone: failedCount ? 'error' : 'success',
        text: failedCount ? '일부 파일을 저장하지 못했습니다.' : `${storageResults.length}개 파일의 분류와 저장이 끝났습니다.`,
      });
      await loadStatus(true); await loadFolders();
    } catch (error) {
      const text = error instanceof Error ? error.message : '파일 등록 중 오류가 발생했습니다.';
      setUploadStorageReport({ tone: 'error', text, files: [] });
      setNotice({ tone: 'error', text });
    } finally { setClassifyingFiles(false); setBusy(false); }
  };

  const queueUploadFiles = (files: FileList | File[]) => {
    const incoming = Array.from(files);
    if (!incoming.length) return;
    setClassifiedUploadNames([]);
    setUploadStorageReport(null);
    setPendingUploadFiles((current) => {
      const queued = new Map(current.map((file) => [`${file.name}\0${file.size}\0${file.lastModified}`, file]));
      incoming.forEach((file) => queued.set(`${file.name}\0${file.size}\0${file.lastModified}`, file));
      return [...queued.values()];
    });
    setNotice(null);
  };

  const assignTarget = (ids: string[], targetDir: string) => {
    const idSet = new Set(ids);
    setItems((current) => current.map((item) => idSet.has(item.id) ? { ...item, targetDir, targetPath: `${targetDir}/${item.name}`.replace(/^\//, '') } : item));
    setNotice({ tone: 'info', text: `${ids.length}개 파일의 대상 폴더를 수동 지정했습니다.` });
  };

  const toggleSelected = (id: string) => setSelected((current) => {
    const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next;
  });
  const activeItems = items.filter((item) => selected.has(item.id));

  const removeItem = (item: FileOrganizerItem) => {
    setItems((current) => current.filter((entry) => entry.id !== item.id));
    setSelected((current) => { const next = new Set(current); next.delete(item.id); return next; });
    if (item.sourceKind === 'upload') {
      void fetch(`${API_BASE}/api/file-organizer/discard`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sourcePath: item.sourcePath }),
      }).catch(() => { /* 대기열에서는 이미 지웠으니, 임시 파일 정리 실패는 조용히 넘어간다. */ });
    }
  };

  const execute = async () => {
    if (!activeItems.length) { setNotice({ tone: 'error', text: '실행할 파일을 하나 이상 선택해 주세요.' }); return; }
    const action = operation === 'copy' ? '복사' : '이동';
    if (!window.confirm(`선택한 ${activeItems.length}개 파일을 ${action}할까요?`)) return;
    setBusy(true); setNotice(null);
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/execute`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operation, conflict, items: activeItems.map((item) => ({ sourcePath: item.sourcePath, targetDir: item.targetDir })) }),
      });
      const data = await response.json() as { results?: { source: string; status: string; message: string }[]; databaseNote?: string; error?: string };
      if (!response.ok) throw new Error(data.error || '파일 정리를 실행하지 못했습니다.');
      const successful = new Set((data.results || []).filter((result) => result.status === 'success').map((result) => result.source));
      const failed = (data.results || []).filter((result) => result.status === 'error').length;
      setItems((current) => current.filter((item) => !successful.has(item.sourcePath)));
      setSelected((current) => new Set([...current].filter((id) => items.some((item) => item.id === id && !successful.has(item.sourcePath)))));
      setNotice({ tone: failed ? 'error' : 'success', text: `${successful.size}개 ${action} 완료${failed ? ` · ${failed}개 오류` : ''} · ${data.databaseNote || '감사 로그 저장'}` });
      await loadStatus(true); await loadFolders();
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : '파일 정리 중 오류가 발생했습니다.' });
    } finally { setBusy(false); }
  };

  const connectDatabase = async () => {
    if (!databaseUrl.trim()) { setNotice({ tone: 'error', text: 'MariaDB 연결 URL을 입력해 주세요.' }); return; }
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/database`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ databaseUrl }) });
      const data = await response.json() as FileDatabaseStatus & { error?: string };
      if (!response.ok) throw new Error(data.error || 'MariaDB에 연결하지 못했습니다.');
      setNotice({ tone: 'success', text: `${data.label} 연결과 테이블 초기화를 완료했습니다.` });
      setShowDatabase(false); await loadStatus(true);
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'MariaDB 연결 중 오류가 발생했습니다.' });
    } finally { setBusy(false); }
  };

  const moveAxis = (index: number, direction: -1 | 1) => {
    setFolderOrder((current) => {
      const target = index + direction;
      if (target < 0 || target >= current.length) return current;
      const next = [...current];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  };

  const saveFolderOrder = async () => {
    if (!window.confirm('실제 정리 대상 폴더 안의 파일들을 새 순서로 지금 바로 옮깁니다. 계속할까요?')) return;
    setSavingOrder(true);
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/folder-order`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folderOrder }),
      });
      const data = await response.json() as FolderOrderResponse & { migration?: { moved: number; skipped: number; structureMoved?: number; errors: string[] } };
      if (!response.ok) throw new Error(data.error || '폴더 순서를 저장하지 못했습니다.');
      setFolderOrder(normalizeFolderOrder(data.folderOrder));
      const migration = data.migration;
      /* 빈 폴더뿐인 트리에서는 옮긴 파일이 0개라도 폴더 배치는 실제로 바뀐다.
         그때 "제자리입니다"라고만 알리면 화면과 디스크가 어긋나 보인다. */
      const errorNote = migration?.errors.length ? ` · 오류 ${migration.errors.length}건` : '';
      const summary = migration
        ? migration.moved > 0
          ? `파일 ${migration.moved}개를 새 구조로 옮겼습니다${errorNote}.`
          : (migration.structureMoved ?? 0) > 0
            ? `옮길 파일은 없었고, 빈 폴더 ${migration.structureMoved}개를 새 순서로 재배치했습니다${errorNote}.`
            : '모든 파일이 이미 제자리입니다 — 옮길 파일이 없습니다.'
        : '폴더 순서를 저장했습니다.';
      setNotice({ tone: migration?.errors.length ? 'error' : 'success', text: summary });
      setShowFolderOrder(false);
      await loadFolders();
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : '폴더 순서 저장 중 오류가 발생했습니다.' });
    } finally { setSavingOrder(false); }
  };

  const db = status?.database;
  const examplePath = folderOrder.map((axis) => {
    if (axis === 'item') return '67312';
    if (axis === 'vehicle') return 'JM';
    if (axis === 'category') return '02. 금형도면';
    return '01. 구조도';
  }).join(' / ');
  const hasRebuiltFolder = rootEntries.some((entry) => entry.isDirectory);
  return <section className="page page--file-organizer">
    <div className="page-heading page-heading--compact file-organizer-heading">
      <div><h2>폴더 구조 변경</h2></div>
      <div className="organizer-settings-actions"><button type="button" onClick={() => setShowFolderOrder((current) => !current)}><Layers3 size={14} /> 폴더 구조 변경</button><button type="button" className={db?.connected ? 'connected' : db?.connected === false ? 'error' : ''} onClick={() => setShowDatabase((current) => !current)}><Database size={14} /> {db?.label || 'DB 확인'}</button></div>
    </div>
    {notice && <div className={`file-organizer-notice ${notice.tone}`}>{notice.tone === 'success' ? <CheckCircle2 size={16} /> : notice.tone === 'error' ? <AlertTriangle size={16} /> : <CircleHelp size={16} />}<span>{notice.text}</span><button onClick={() => setNotice(null)} aria-label="알림 닫기"><X size={14} /></button></div>}
    <section className="organizer-flow-section" aria-label="폴더 구조 변경 작업">
      {showDatabase && <div className="card file-database-settings"><div><Database size={20} /><span><b>MariaDB 연결</b><small>파일은 로컬/NAS에, 태그와 작업 이력은 MariaDB에 저장됩니다.</small></span></div><input value={databaseUrl} onChange={(event) => setDatabaseUrl(event.target.value)} placeholder="mysql://사용자:비밀번호@서버:3306/file_organizer" /><button type="button" onClick={() => setDatabaseUrl('mysql://file_demo:file_demo_password@127.0.0.1:3307/file_organizer?charset=utf8mb4&connect_timeout=5')}>데모 설정</button><button type="button" className="primary-button" onClick={connectDatabase} disabled={busy}>연결 테스트·저장</button></div>}
      <div className="organizer-flow-lanes">
        <article className="organizer-flow-lane organizer-flow-lane--existing">
          <div className="organizer-flow-lane__label"><b>01</b><span><strong>폴더 재구성</strong></span></div>
          <div className="organizer-flow-track">
            <div className="organizer-flow-node organizer-flow-node--source">
              <span className="organizer-flow-node__icon"><Server size={22} /></span>
              <span className="organizer-flow-node__text"><strong>기존 폴더</strong><code title={status?.sourceRoot}>{status?.sourceRoot || '경로 확인 중'}</code></span>
              <button type="button" className="organizer-flow-open organizer-flow-open--labeled organizer-flow-open--select" onClick={() => void chooseOrganizerFolder('source')} disabled={busy || pathsInfo?.sourceLocked || selectingRoot !== null} title="정리할 기존 폴더 선택·변경"><FolderOpen size={15} /><span>{selectingRoot === 'source' ? '선택 중…' : '폴더 선택'}</span></button>
            </div>
            <div className="organizer-flow-arrow organizer-flow-arrow--copy" aria-hidden="true"><i /></div>
            <div className="organizer-flow-node organizer-flow-node--result">
              <span className="organizer-flow-node__icon"><HardDrive size={22} /></span>
              <span className="organizer-flow-node__text"><strong>재구성 폴더</strong><code title={status?.destinationRoot}>{status?.destinationRoot || '경로 확인 중'}</code></span>
              <span className="organizer-flow-node__actions"><button type="button" className="organizer-flow-open organizer-flow-open--labeled organizer-flow-open--settings" onClick={() => void chooseOrganizerFolder('destination')} disabled={busy || pathsInfo?.destinationLocked || selectingRoot !== null} title="재구성 폴더 경로 설정"><Settings2 size={14} /><span>{selectingRoot === 'destination' ? '선택 중…' : '경로 설정'}</span></button></span>
            </div>
          </div>
          <div className="organizer-flow-lane-footer">
            <div className="organizer-flow-status-stack">
              <div className="organizer-flow-preserve-note"><ShieldCheck size={16} /><span>기존 폴더는 유지됩니다.</span></div>
              <div className={`organizer-rebuilt-status ${hasRebuiltFolder ? 'is-ready' : 'is-empty'}`}>
                {hasRebuiltFolder ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}
                <span>{hasRebuiltFolder ? '재구성한 폴더가 이미 존재한다면 진행하지 않아도 됩니다.' : '재구성된 폴더가 없습니다.'}</span>
              </div>
            </div>
            <div className="organizer-flow-lane-actions"><button type="button" className="organizer-flow-action organizer-flow-action--existing" onClick={() => void scanSource(true)} disabled={busy || selectingRoot !== null} title="기존 폴더를 분석한 뒤 재구성 폴더에 자동으로 복사합니다"><RefreshCw size={16} /> {busy ? '처리 중…' : '폴더 재구성'}<ArrowRight size={16} /></button></div>
          </div>
        </article>

        {reconstructionReport && <div className={`organizer-reconstruction-report ${reconstructionReport.tone}`} role="status" aria-live="polite">
          <div className="organizer-reconstruction-report__heading">
            {reconstructionReport.tone === 'success' ? <CheckCircle2 size={20} /> : reconstructionReport.tone === 'error' ? <AlertTriangle size={20} /> : <CircleHelp size={20} />}
            <strong>{reconstructionReport.text}</strong>
          </div>
          {reconstructionReport.issues.length > 0 && <ul>{reconstructionReport.issues.map((issue) => <li key={issue.sourcePath}>
            <b>{issue.name}</b><span>{issue.message}</span><small>{issue.sourcePath}</small>
          </li>)}</ul>}
        </div>}

        {showFolderOrder && <div className="card file-order-settings">
          <div className="file-order-settings__intro"><ListFilter size={20} /><span><b>폴더 구조 순서</b><small>차종·품번·자료 유형·세부 폴더의 구조 순서를 정합니다.</small></span></div>
          <ol className="file-order-axis-list">{folderOrder.map((axis, index) => <li key={axis}><span className="file-order-axis-index">{index + 1}</span><span className="file-order-axis-label">{axisLabel(axis)}</span><span className="file-order-axis-buttons"><button type="button" onClick={() => moveAxis(index, -1)} disabled={index === 0} aria-label="위로"><ChevronDown size={14} style={{ transform: 'rotate(180deg)' }} /></button><button type="button" onClick={() => moveAxis(index, 1)} disabled={index === folderOrder.length - 1} aria-label="아래로"><ChevronDown size={14} /></button></span></li>)}</ol>
          <button type="button" className="primary-button" onClick={saveFolderOrder} disabled={savingOrder}>{savingOrder ? '저장 중…' : '저장'}</button>
        </div>}

        <article className={`organizer-flow-lane organizer-flow-lane--upload ${dragging ? 'is-dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event: DragEvent<HTMLElement>) => { event.preventDefault(); setDragging(false); if (!busy) queueUploadFiles(event.dataTransfer.files); }}>
          <div className="organizer-flow-lane__label"><b>02</b><span><strong>파일 추가</strong><small>새로운 파일을 선택한 뒤 파일 분류를 눌러 저장 위치를 분석합니다.</small></span></div>
          <div className="organizer-upload-flow">
            <label className="organizer-upload-source" aria-disabled={busy}>
              <input type="file" multiple disabled={busy} onChange={(event) => { if (event.target.files) queueUploadFiles(event.target.files); event.currentTarget.value = ''; }} />
              <span className="organizer-flow-node__icon"><UploadCloud size={24} /></span>
              <span className="organizer-upload-source__content">
                <span><strong>{classifyingFiles ? '파일 분류 중…' : pendingUploadFiles.length ? `${pendingUploadFiles.length}개 파일 선택됨` : classifiedUploadNames.length ? '파일 분류 완료' : '파일 업로드'}</strong><em>{pendingUploadFiles.length ? '파일을 더 추가하려면 이 영역을 다시 눌러주세요.' : classifiedUploadNames.length ? `${classifiedUploadNames.length}개 파일을 분류했습니다.` : '여러 파일을 한 번에 올릴 수 있어요'}</em></span>
                {(pendingUploadFiles.length > 0 || classifiedUploadNames.length > 0) && <span className={`organizer-upload-file-list ${classifiedUploadNames.length ? 'is-complete' : ''}`}>
                  {(pendingUploadFiles.length ? pendingUploadFiles.map((file) => file.name) : classifiedUploadNames).map((name, index) => <span className="organizer-upload-file" key={`${name}-${index}`}><File size={14} /><b title={name}>{name}</b>{classifiedUploadNames.length > 0 && <CheckCircle2 size={14} />}</span>)}
                </span>}
              </span>
            </label>
            <div className="organizer-split-arrow" aria-hidden="true"><i /><b /><em /></div>
            <div className="organizer-flow-branches">
              <div className="organizer-flow-node organizer-flow-node--source organizer-flow-node--branch">
                <span className="organizer-flow-node__icon"><Server size={20} /></span>
                <span className="organizer-flow-node__text"><strong>기존 폴더</strong><code title={status?.sourceRoot}>{status?.sourceRoot || '경로 확인 중'}</code></span>
                <button type="button" className="organizer-flow-open organizer-flow-open--labeled organizer-flow-open--select" onClick={() => void chooseOrganizerFolder('source')} disabled={busy || pathsInfo?.sourceLocked || selectingRoot !== null} title="새 파일을 함께 저장할 기존 폴더 선택"><FolderOpen size={14} /><span>{selectingRoot === 'source' ? '선택 중…' : '폴더 선택'}</span></button>
              </div>
              <div className="organizer-flow-node organizer-flow-node--result organizer-flow-node--branch">
                <span className="organizer-flow-node__icon"><HardDrive size={20} /></span>
                <span className="organizer-flow-node__text"><strong>재구성 폴더</strong><code title={status?.destinationRoot}>{status?.destinationRoot || '경로 확인 중'}</code></span>
                <button type="button" className="organizer-flow-open organizer-flow-open--labeled organizer-flow-open--select" onClick={() => void chooseOrganizerFolder('destination', 'existing')} disabled={busy || pathsInfo?.destinationLocked || selectingRoot !== null} title="01에서 재구성한 폴더 선택"><FolderOpen size={14} /><span>{selectingRoot === 'destination' ? '선택 중…' : '폴더 선택'}</span></button>
              </div>
            </div>
          </div>
          <div className="organizer-upload-classify">
            <span>{pendingUploadFiles.length ? `선택한 ${pendingUploadFiles.length}개 파일을 분류할 준비가 되었습니다.` : classifiedUploadNames.length ? `${classifiedUploadNames.length}개 파일의 분류가 완료되었습니다.` : '분류할 파일을 먼저 선택해 주세요.'}</span>
            <button type="button" onClick={() => void uploadFiles(pendingUploadFiles)} disabled={busy || pendingUploadFiles.length === 0 || selectingRoot !== null}><ListFilter size={16} /> {classifyingFiles ? '분류 중…' : '파일 분류'}</button>
          </div>
        </article>
        {uploadStorageReport && <div className={`organizer-reconstruction-report organizer-upload-storage-report ${uploadStorageReport.tone}`} role="status" aria-live="polite">
          <div className="organizer-reconstruction-report__heading">
            {uploadStorageReport.tone === 'success' ? <CheckCircle2 size={19} /> : uploadStorageReport.tone === 'error' ? <AlertTriangle size={19} /> : <RefreshCw size={19} className="spin" />}
            <b>{uploadStorageReport.text}</b>
          </div>
          {uploadStorageReport.files.length > 0 && <ul>{uploadStorageReport.files.map((file, index) => <li key={`${file.sourcePath}-${index}`}>
            <b>{file.name}</b>
            <span className={`organizer-upload-storage-target ${file.existing.status}`}><strong>기존 폴더</strong><code title={file.existing.path}>{file.existing.path || file.existing.message || '저장 위치를 확인하지 못했습니다.'}</code>{file.existing.status === 'error' && file.existing.path && <small>{file.existing.message}</small>}</span>
            <span className={`organizer-upload-storage-target ${file.reconstructed.status}`}><strong>재구성 폴더</strong><code title={file.reconstructed.path}>{file.reconstructed.path || file.reconstructed.message || '저장 위치를 확인하지 못했습니다.'}</code>{file.reconstructed.status === 'error' && file.reconstructed.path && <small>{file.reconstructed.message}</small>}</span>
          </li>)}</ul>}
        </div>}
      </div>
    </section>
    {showFolderOrder && <div className="card file-order-settings">
      <div className="file-order-settings__intro"><ListFilter size={20} /><span><b>폴더 구조 순서</b><small>품번·차종·카테고리·세부 하위폴더의 쌓는 순서를 자유롭게 바꿀 수 있습니다. 저장하면 기존 파일도 새 순서로 옮겨집니다.</small></span></div>
      <ol className="file-order-axis-list">{folderOrder.map((axis, index) => <li key={axis}><span className="file-order-axis-index">{index + 1}</span><span className="file-order-axis-label">{axisLabel(axis)}</span><span className="file-order-axis-buttons"><button type="button" onClick={() => moveAxis(index, -1)} disabled={index === 0} aria-label="위로"><ChevronDown size={14} style={{ transform: 'rotate(180deg)' }} /></button><button type="button" onClick={() => moveAxis(index, 1)} disabled={index === folderOrder.length - 1} aria-label="아래로"><ChevronDown size={14} /></button></span></li>)}</ol>
      <div className="file-order-preview"><small>예시 경로</small><code>{examplePath}</code></div>
      <button type="button" className="primary-button" onClick={saveFolderOrder} disabled={savingOrder}>{savingOrder ? '저장 중…' : '이 순서로 저장'}</button>
    </div>}
    {showDatabase && <div className="card file-database-settings"><div><Database size={20} /><span><b>MariaDB 연결</b><small>파일은 로컬/NAS에, 태그와 작업 이력은 MariaDB에 저장됩니다.</small></span></div><input value={databaseUrl} onChange={(event) => setDatabaseUrl(event.target.value)} placeholder="mysql://사용자:비밀번호@서버:3306/file_organizer" /><button type="button" onClick={() => setDatabaseUrl('mysql://file_demo:file_demo_password@127.0.0.1:3307/file_organizer?charset=utf8mb4&connect_timeout=5')}>데모 설정</button><button type="button" className="primary-button" onClick={connectDatabase} disabled={busy}>연결 테스트·저장</button></div>}
    <div className="file-storage-strip">
      <div><Server size={18} /><span><small>원본 폴더</small><b title={status?.sourceRoot}>{status?.sourceRoot || '확인 중'}</b></span><em className={status?.sourceAvailable ? 'ok' : ''}>{status?.sourceAvailable ? '연결됨' : '경로 없음'}</em><button type="button" className="file-storage-action" onClick={() => void openInExplorer('source')} title="탐색기에서 열기" aria-label="원본 폴더 탐색기에서 열기"><FolderOpen size={14} /></button></div>
      <ChevronRight size={17} />
      <div><HardDrive size={18} /><span><small>정리 대상 · 추후 NAS</small><b title={status?.destinationRoot}>{status?.destinationRoot || '확인 중'}</b></span><em className={status?.destinationAvailable ? 'ok' : ''}>{status?.destinationAvailable ? '연결됨' : '경로 없음'}</em><button type="button" className="file-storage-action" onClick={() => void openInExplorer('destination')} title="탐색기에서 열기" aria-label="정리 대상 폴더 탐색기에서 열기"><FolderOpen size={14} /></button></div>
      <div className="file-storage-metrics"><span>카탈로그 <b>{db?.catalogCount || 0}</b></span><span>작업 이력 <b>{db?.operationCount || 0}</b></span></div>
    </div>
  </section>;
}

function SheetTitleBlock({ values, onChange, fonts, onFontChange, fontSizes, onFontSizeChange }: { values: SheetTitleValues; onChange: (field: SheetTitleField, value: string) => void; fonts: SheetTitleFonts; onFontChange: (field: SheetTitleField, fontFamily: string) => void; fontSizes: SheetTitleFontSizes; onFontSizeChange: (field: SheetTitleField, size: number) => void }) {
  const editableText = (field: SheetTitleField, label: string, heading = false) => {
    const size = fontSizes[field] ?? TITLE_DEFAULT_FONT_SIZE[field];
    return <>
      <input
        type="text"
        className={`sheet-title-block__input${heading ? ' sheet-title-block__input--heading' : ''}`}
        style={{ fontFamily: fonts[field] || undefined, fontSize: fontSizes[field] ? `${fontSizes[field]}px` : undefined }}
        value={values[field]}
        onChange={(event) => onChange(field, event.target.value)}
        aria-label={`${label} 수정`}
        title={`${label} - 클릭하여 수정`}
        autoComplete="off"
        spellCheck={false}
      />
      {/* 셀 안에 포커스가 남아있는 동안(:focus-within)만 뜨는 작은 도구막대 — 텍스트를 선택/편집하는 동안 엑셀처럼 옆에서 바로 글꼴·크기를 바꾼다. */}
      <div className="cell-font-picker">
        <select value={fonts[field]} onChange={(event) => onFontChange(field, event.target.value)} onPointerDown={(event) => event.stopPropagation()} aria-label={`${label} 글꼴 선택`}>
          {FONT_FAMILY_OPTIONS.map((option) => <option key={option.label} value={option.value} style={{ fontFamily: option.value || undefined }}>{option.label}</option>)}
        </select>
        <span className="cell-font-picker__divider" />
        <button type="button" onClick={() => onFontSizeChange(field, clamp(size - TITLE_FONT_SIZE_STEP, TITLE_FONT_SIZE_MIN, TITLE_FONT_SIZE_MAX))} disabled={size <= TITLE_FONT_SIZE_MIN} aria-label={`${label} 글자 작게`} title="글자 작게">A<small>−</small></button>
        <span className="cell-font-picker__size" aria-live="polite">{size}</span>
        <button type="button" onClick={() => onFontSizeChange(field, clamp(size + TITLE_FONT_SIZE_STEP, TITLE_FONT_SIZE_MIN, TITLE_FONT_SIZE_MAX))} disabled={size >= TITLE_FONT_SIZE_MAX} aria-label={`${label} 글자 크게`} title="글자 크게">A<small>+</small></button>
      </div>
    </>;
  };
  return <section className="sheet-title-block" aria-label="보정 적용 내용">
    <div className="sheet-title-block__heading"><strong>{editableText('heading', '보정 시트 제목', true)}</strong></div>
    <div className="sheet-title-block__label">{editableText('managementLabel', '관리 NO 항목명')}</div><div className="sheet-title-block__value">{editableText('managementNo', '관리 NO 값')}</div>
    <div className="sheet-title-block__label">{editableText('partNameLabel', 'PART NAME 항목명')}</div><div className="sheet-title-block__value">{editableText('partName', 'PART NAME 값')}</div>
    <div className="sheet-title-block__label">{editableText('processLabel', '공정 항목명')}</div><div className="sheet-title-block__value">{editableText('process', '공정 값')}</div>
    <div className="sheet-title-block__label">{editableText('partNoLabel', 'PART NO 항목명')}</div><div className="sheet-title-block__value">{editableText('partNo', 'PART NO 값')}</div>
    <div className="sheet-title-block__label">{editableText('materialLabel', '원소재 항목명')}</div><div className="sheet-title-block__value">{editableText('material', '원소재 값')}</div>
    <div className="sheet-title-block__label">{editableText('appliedDateLabel', '적용일자 항목명')}</div><div className="sheet-title-block__value">{editableText('appliedDate', '적용일자 값')}</div>
  </section>;
}

function formatHistoryValue(value: number | null) {
  if (value == null) return '자동';
  return `${value > 0 ? '+' : ''}${value.toFixed(1)} mm`;
}

function formatHistoryTime(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value.replace('T', ' ');
  return new Intl.DateTimeFormat('ko-KR', {
    month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  }).format(parsed);
}

function CorrectionHistoryPanel({ partNo, entries, loading, pendingPointIds, deletingEntryIds, error, onReload, onRestore, onDelete }: {
  partNo: string;
  entries: CorrectionHistoryEntry[];
  loading: boolean;
  pendingPointIds: Set<string>;
  deletingEntryIds: Set<number>;
  error: string | null;
  onReload: () => void;
  onRestore: (entry: CorrectionHistoryEntry) => void;
  onDelete: (entry: CorrectionHistoryEntry) => void;
}) {
  return <div className="card correction-history">
    <div className="card-title"><div><h3>보정 이력</h3></div><button type="button" className="correction-history__reload" onClick={onReload} disabled={loading} aria-label="이력 새로고침" title="이력 새로고침">↻</button></div>
    {error && <p className="correction-history__message correction-history__message--error" role="alert">{error}</p>}
    {entries.length === 0 ? <p className="correction-history__empty">{loading ? '불러오는 중…' : '기록된 수정 이력이 없습니다.'}</p> : <ul className="correction-history__list">{entries.map((entry) => {
      const busy = pendingPointIds.has(entry.pointId) || deletingEntryIds.has(entry.id);
      return <li key={entry.id} className="correction-history__item">
        <span className="correction-history__point">{entry.pointId}</span>
        <div className="correction-history__change"><span>{formatHistoryValue(entry.oldValue)}</span><i>→</i><span>{formatHistoryValue(entry.newValue)}</span></div>
        <div className="correction-history__meta"><span>{entry.worker || '이름 미입력'} · {formatHistoryTime(entry.createdAt)}</span></div>
        <div className="correction-history__actions">
          <button type="button" onClick={() => onRestore(entry)} disabled={busy} title="이 포인트를 엔진이 계산한 원래 값으로 되돌립니다">원래 값 복원</button>
          <button type="button" className="correction-history__delete" onClick={() => onDelete(entry)} disabled={busy} title="이 기록만 삭제합니다 (포인트 값은 바뀌지 않음)">이력 삭제</button>
        </div>
      </li>;
    })}</ul>}
  </div>;
}

function ServicePreview({ scan, hiddenPointIds, onPointToggle, keyPointsOnly, onKeyPointsOnlyChange, pointOverrides, onOverrideChange, onClearAllOverrides, annotations = [], setAnnotations, sheetTitle, onSheetTitleChange, sheetTitleFonts, onSheetTitleFontChange, sheetTitleFontSizes, onSheetTitleFontSizeChange, worker, onWorkerChange, coefficient, onCoefficientChange, zeroEdits, onZeroEditsChange, sheetTransformByScan, setSheetTransformByScan, sheetLayoutsByScan, setSheetLayoutsByScan, detailRegionsByScan, setDetailRegionsByScan, frontLabelPositionsByScan, setFrontLabelPositionsByScan, detailLabelPositionsByScan, setDetailLabelPositionsByScan, addedPointsByScan, setAddedPointsByScan }: { scan: ScanItem; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; keyPointsOnly: boolean; onKeyPointsOnlyChange: (value: boolean) => void; pointOverrides: Record<string, number>; onOverrideChange: (id: string, value: number | null) => void; onClearAllOverrides: () => void; annotations: Annotation[]; setAnnotations: (updater: (current: Annotation[]) => Annotation[]) => void; sheetTitle: SheetTitleValues; onSheetTitleChange: (field: SheetTitleField, value: string) => void; sheetTitleFonts: SheetTitleFonts; onSheetTitleFontChange: (field: SheetTitleField, fontFamily: string) => void; sheetTitleFontSizes: SheetTitleFontSizes; onSheetTitleFontSizeChange: (field: SheetTitleField, size: number) => void; worker: string; onWorkerChange: (value: string) => void; coefficient: number; onCoefficientChange: (value: number) => void; zeroEdits: ZeroEdit[]; onZeroEditsChange: (edits: ZeroEdit[]) => void; sheetTransformByScan: Record<string, SheetImageTransform>; setSheetTransformByScan: React.Dispatch<React.SetStateAction<Record<string, SheetImageTransform>>>; sheetLayoutsByScan: Record<string, SheetLayout[]>; setSheetLayoutsByScan: React.Dispatch<React.SetStateAction<Record<string, SheetLayout[]>>>; detailRegionsByScan: Record<string, DetailRegion[]>; setDetailRegionsByScan: React.Dispatch<React.SetStateAction<Record<string, DetailRegion[]>>>; frontLabelPositionsByScan: Record<string, Record<string, { x: number; y: number }>>; setFrontLabelPositionsByScan: React.Dispatch<React.SetStateAction<Record<string, Record<string, { x: number; y: number }>>>>; detailLabelPositionsByScan: Record<string, Record<string, Record<string, { x: number; y: number }>>>; setDetailLabelPositionsByScan: React.Dispatch<React.SetStateAction<Record<string, Record<string, Record<string, { x: number; y: number }>>>>>; addedPointsByScan: Record<string, PointResult[]>; setAddedPointsByScan: React.Dispatch<React.SetStateAction<Record<string, PointResult[]>>> }) {
  const result = scan.result!; const points = result.points; const [showPoints, setShowPoints] = useState(true); const [showZero, setShowZero] = useState(true);
  const [zeroPanel, setZeroPanel] = useState(false);
  const [zeroPointAddMode, setZeroPointAddMode] = useState(false);
  const [zeroPointDeleteMode, setZeroPointDeleteMode] = useState(false);
  const [draftZeroEdits, setDraftZeroEdits] = useState<ZeroEdit[]>(zeroEdits);
  useEffect(() => { setDraftZeroEdits(zeroEdits); }, [scan.id, zeroEdits]);
  /* 보정치 수치 라벨(+1.5 등) 글꼴 — "선택하면 자유롭게" 가 아니라 시트 전체 한 번에 바뀌는 값이라 여기 하나로 둔다. */
  const [pointLabelFont, setPointLabelFont] = useState(DEFAULT_POINT_LABEL_FONT);
  /* 보정시트에 들어가는 그림은 편차 히트맵이 아니라 깨끗한 제품데이터다.
     정렬된 제품데이터가 있으면 그쪽을 기본으로 쓰고, 없을 때만 스캔으로 물러선다. */
  const alignment = result.alignment;
  const productReady = Boolean(result.productImage && alignment);
  const [useProduct, setUseProduct] = useState(true);
  const onProduct = productReady && useProduct;
  /* 미리 렌더된 zeroOverlay 는 스캔 좌표계 래스터라 제품데이터 위에는 못 얹지만,
     벡터 폴리라인(result.zeroLines) 이 있으면 alignment 로 좌표를 옮겨 SVG 로 그려 준다. */
  const hasZeroVector = Boolean(result.zeroLines && result.zeroLines.length);
  const zeroReady = hasZeroVector || (Boolean(result.zeroOverlay) && !onProduct);
  const frameWidth = onProduct ? alignment!.productSize[0] : result.source.width;
  const frameHeight = onProduct ? alignment!.productSize[1] : result.source.height;
  const sheetImageSource = onProduct
    ? result.productImage!
    : showZero && result.zeroOverlay && !hasZeroVector ? result.zeroOverlay : result.cleanImage || scan.url;
  /* 회전/반전 상태도 위(sheetLayoutsByScan 등)와 같은 이유로 Home 에서
     scan.id 로 갈라 물려받는다 -- 탭을 옮겨도 유지되면서, 다른 파트는
     scan.id 가 다르니 자동으로 항등 변환(IDENTITY_SHEET_TRANSFORM)부터
     시작한다. */
  const sheetTransform = sheetTransformByScan[scan.id] ?? IDENTITY_SHEET_TRANSFORM;
  const setSheetTransform = useCallback((updater: SheetImageTransform | ((current: SheetImageTransform) => SheetImageTransform)) => {
    setSheetTransformByScan((current) => {
      const value = current[scan.id] ?? IDENTITY_SHEET_TRANSFORM;
      const next = typeof updater === 'function' ? (updater as (current: SheetImageTransform) => SheetImageTransform)(value) : updater;
      return { ...current, [scan.id]: next };
    });
  }, [scan.id, setSheetTransformByScan]);
  const renderedSheetImage = useTransformedSheetImage(sheetImageSource, sheetTransform);
  const activeSheetTransform = renderedSheetImage.transform;
  const sheetQuarterTurn = activeSheetTransform.rotation === 90 || activeSheetTransform.rotation === 270;
  const sheetFrameWidth = sheetQuarterTurn ? frameHeight : frameWidth;
  const sheetFrameHeight = sheetQuarterTurn ? frameWidth : frameHeight;
  const rotateSheet = () => setSheetTransform((current) => ({
    ...current,
    rotation: ((current.rotation + 90) % 360) as SheetRotation,
  }));
  const flipSheetHorizontal = () => setSheetTransform((current) => (
    current.rotation === 90 || current.rotation === 270
      ? { ...current, flipY: !current.flipY }
      : { ...current, flipX: !current.flipX }
  ));
  const flipSheetVertical = () => setSheetTransform((current) => (
    current.rotation === 90 || current.rotation === 270
      ? { ...current, flipX: !current.flipX }
      : { ...current, flipY: !current.flipY }
  ));
  const requestedQuarterTurn = sheetTransform.rotation === 90 || sheetTransform.rotation === 270;
  const sheetHorizontalFlipped = requestedQuarterTurn ? sheetTransform.flipY : sheetTransform.flipX;
  const sheetVerticalFlipped = requestedQuarterTurn ? sheetTransform.flipX : sheetTransform.flipY;
  const [tool, setTool] = useState<AnnotationTool>('select'); const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null); const [showAnnotations, setShowAnnotations] = useState(true); const [detailMode, setDetailMode] = useState(false); const [labelAreaMode, setLabelAreaMode] = useState<'hide' | 'show' | null>(null);
  /* 엑셀 내보내기가 실제 UI 배치(정면도·디테일 뷰의 캔버스 % 좌표)를
     알아야 주석·창 위치를 시트에 그대로 옮길 수 있다. SheetCanvas 안에
     있는 layouts 상태를 콜백으로 위로 끌어올린다.

     scan.id 로 갈라 두는 이유: "다른 파트로 바뀌면 이전 배치를 지운다"를
     "바뀌는 순간 useState 를 {} 로 리셋"하는 방식으로 짰다가 실제로
     라벨이 새로 마운트된 SheetCanvas 로 넘어가는 걸 fiber 단위로 추적해
     보니, 리렌더 타이밍상 새로 마운트되는 자식이 리셋 *전* 값을 초기
     시드로 붙잡아 버리고, 그 자식의 lift-up 이펙트가 그 옛 값을 다시
     부모로 밀어 올려 리셋을 덮어써 버렸다(React 의 "prop 변경 시 렌더링
     중 setState" 패턴은 이 중첩 key-리마운트 상황에서 보장되지 않았다).
     리셋 타이밍에 의존하는 대신, 애초에 파트별로 갈라 저장하면 다른
     파트의 값이 존재할 수 없어 새는 경로 자체가 없다. 이제 이 dict 자체도
     Home 에서 물려받는다(탭 전환 시 언마운트돼도 유지되도록). */
  const sheetLayouts = sheetLayoutsByScan[scan.id] ?? [];
  const setSheetLayouts = useCallback((next: SheetLayout[]) => {
    setSheetLayoutsByScan((current) => ({ ...current, [scan.id]: next }));
  }, [scan.id]);
  /* 엔진 결과는 그대로 두고 작업자가 찍은 포인트만 따로 얹는다. 이 목록도
     scan.id 로 갈라 Home 에서 물려받는다 -- 안 그러면 탭을 옮겼다 왔을 때
     점이 사라지고, 그 점에 물려 있던 라벨 위치(위 frontLabelPositionsByScan)
     도 "화면에 없는 점" 취급으로 같이 지워졌다. */
  const addedPoints = addedPointsByScan[scan.id] ?? [];
  const setAddedPoints = useCallback((updater: PointResult[] | ((current: PointResult[]) => PointResult[])) => {
    setAddedPointsByScan((current) => {
      const value = current[scan.id] ?? [];
      const next = typeof updater === 'function' ? (updater as (current: PointResult[]) => PointResult[])(value) : updater;
      return { ...current, [scan.id]: next };
    });
  }, [scan.id, setAddedPointsByScan]);
  const [addPointMode, setAddPointMode] = useState(false);
  const [sampling, setSampling] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const editedZeroLinePixels = useMemo<[number, number][][]>(() => (result.zeroLines || []).map((line, lineIndex) => {
    const edit = draftZeroEdits.find((item) => item.index === lineIndex);
    if (edit?.hidden) return [];
    const source = Array.isArray(edit?.vertices) && edit.vertices.length >= 2 ? edit.vertices : (line.points || []);
    return source
      .filter((point) => Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]))
      .map(([x, y], pointIndex) => {
        const point = edit?.points?.[String(pointIndex)];
        return [x + (edit?.dx || 0) + (point?.dx || 0), y + (edit?.dy || 0) + (point?.dy || 0)] as [number, number];
      });
  }), [result.zeroLines, draftZeroEdits]);
  const keyPointIds = useMemo(() => new Set(result.keySelection?.ids
    ?? points.filter((point) => point.keyReasons?.length).map((point) => point.id)), [result.keySelection, points]);
  const hasKeySelection = result.keySelection !== undefined;
  const sheetSourcePoints = [...points, ...addedPoints]
    .filter((point) => !pointTouchesZeroLine(point, editedZeroLinePixels, result.source.width, result.source.height))
    .filter((point) => !keyPointsOnly || !hasKeySelection || keyPointIds.has(point.id));
  /* 제품데이터 위에 올릴 때는 같은 포인트의 좌표만 제품 기준으로 바꿔 넘긴다.
     전사되지 않은 포인트는 제품데이터 밖으로 나간 것이라 시트에서 뺀다.
     라벨 위치 계산(아래)이 각 점의 현재 화면 좌표를 알아야 해서, 이 계산을
     그보다 앞으로 옮겨 뒀다(원래는 이 함수 뒤쪽에 있었다). */
  const sheetPoints = sheetSourcePoints.flatMap((point) => {
    if (!onProduct) return [point];
    if (point.xProduct === undefined || point.yProduct === undefined) return [];
    return [{ ...point, x: point.xProduct, y: point.yProduct }];
  }).map((point) => {
    const [x, y] = transformSheetPoint(activeSheetTransform, point.x, point.y);
    return { ...point, x, y };
  });
  /* 엑셀 내보내기. sheetLayouts 와 같은 이유로 scan.id 로 갈라 저장하고, 이제 이 dict도 Home 에서 물려받는다. */
  const detailRegions = detailRegionsByScan[scan.id] ?? [];
  const setDetailRegions = useCallback((next: DetailRegion[]) => {
    setDetailRegionsByScan((current) => ({ ...current, [scan.id]: next }));
  }, [scan.id]);
  /* 각 레이아웃(정면도 + Detail들)이 화면에 실제로 그린 라벨 위치. 레이어
     대비 0~100% 로, point.x/y 와 같은 규칙이라 엑셀 쪽이 그대로 쓸 수 있다.
     레이아웃 id 로 한 번, scan.id 로 한 번 더 갈라 둔다 -- Detail 뷰마다
     라벨 배치가 다르고(레이아웃 id), 다른 파트의 라벨 위치가 존재할 수도
     없어야 한다(scan.id, 위 sheetLayouts 주석 참고).

     정면도(front) 라벨은 "점에서 얼마나 떨어져 있는지"를 실제 화면
     픽셀 벡터로 저장한다. 처음엔 위치(%) 를 원본 기준으로 저장하고 회전
     때만 transformSheetPoint 로 돌렸는데, 그러면 위치 자체는 점을 잘
     따라가도 지시선 "길이" 가 회전마다 미묘하게 달라졌다 -- 정면도 창은
     회전할 때마다(위 sheetLayouts 넓이-보존 리핏) 실제 픽셀 크기가
     바뀌는데, %는 그 레이어 크기에 상대적이라 같은 % 오프셋도 가로 축이냐
     세로 축이냐에 따라 실제 픽셀 거리가 달라졌던 것(종이 자체의 가로:세로
     비율로 가정한 값으로 보정해 봤지만, 실제 레이어는 제목 표시줄 같은
     테두리 때문에 그 비율과 딱 안 맞아 여전히 어긋났다). 그래서 이론상의
     비율 대신 CorrectionPoints 가 실측한 레이어 픽셀 크기(frontLayerSizePx,
     아래)를 직접 받아 그 실제 픽셀 단위로 벡터를 저장한다 -- 진짜 화면
     픽셀끼리는 가로세로가 항상 같은 척도라, rotateVector(길이를 보존하는
     회전/반전)를 그대로 적용해도 지시선 길이가 절대 안 변한다. 이 두
     dict(frontLabelPositionsByScan/detailLabelPositionsByScan)도 이제
     Home 에서 물려받는다. frontLayerSizePx는 실측값이라 리마운트되면
     다시 재는 게 맞아 여기(로컬)에 그대로 둔다. */
  const [frontLayerSizePx, setFrontLayerSizePx] = useState({ width: 1, height: 1 });
  const handleLayerSizeChange = useCallback((layoutId: string, size: { width: number; height: number }) => {
    if (layoutId !== 'front' || !size.width || !size.height) return;
    setFrontLayerSizePx((current) => (current.width === size.width && current.height === size.height ? current : size));
  }, []);
  /* CorrectionPoints 가 넘기는 위치는 라벨 박스의 "왼쪽 위 모서리" 다.
     점→라벨 벡터를 그 모서리 기준으로 회전시키면 벡터 자체의 길이는
     보존돼도, 사람 눈에 보이는 지시선(점 → 라벨 "가운데")의 길이는 여전히
     달라진다 -- 라벨 박스 자체는 글자라 절대 안 돌아가는데, 모서리 기준
     벡터를 돌리면 그 안 돌아가는 가로/세로 폭(라벨 크기)이 방향마다 다른
     비중으로 섞여 들어간다. 그래서 점→모서리가 아니라 점→"가운데"를
     회전시키고, 가운데에서 모서리로 뺄 때는 회전과 무관하게 항상 같은
     라벨 크기(라벨은 안 돌아가므로)를 그대로 빼준다. 라벨 크기 계산은
     CorrectionPoints 의 getLabelWidth/labelHeight 와 반드시 같아야 한다. */
  const getFrontLabelWidthPx = useCallback((point: PointResult) => {
    const display = pointOverrides[point.id] !== undefined ? pointOverrides[point.id]! : -(point.value * coefficient);
    const text = `${display > 0 ? '+' : ''}${display.toFixed(1)}`;
    return Math.max(24, text.length * 5.2 + 8);
  }, [pointOverrides, coefficient]);
  const FRONT_LABEL_HEIGHT_PX = 17;
  /* 정면도 창은 회전마다(위 넓이-보존 리핏) 가로/세로 폭이 서로 다른 비율로
     바뀐다 -- 넓이(w*h)는 보존되지만 가로:세로 비율 자체는 방향마다 다르다.
     stored 벡터를 그냥 실측 px 로 회전시키면, 회전 전/후 레이어의 "축척"이
     가로·세로마다 다르게 늘어나 있어 rotateVector 의 등거리 가정이 깨진다
     (예: 가로로 넓적한 레이어 → 세로로 넓적한 레이어). 넓이가 보존되니
     sqrt(w*h) 는 회전과 무관하게 항상 같다 -- 이 값을 공통 척도로 삼아 실측
     px 를 등방(가로세로 같은 척도) 좌표로 바꾼 다음에 회전시키고, 표시할 때
     다시 그 순간의 실측 가로/세로로 풀어내야 지시선 길이가 정확히 보존된다. */
  const storedFrontLabelOffsets = frontLabelPositionsByScan[scan.id] ?? {};
  const displayedFrontLabelPositions = useMemo(() => {
    const next: Record<string, { x: number; y: number }> = {};
    const layerScale = Math.sqrt(frontLayerSizePx.width * frontLayerSizePx.height) || 1;
    for (const point of sheetPoints) {
      const stored = storedFrontLabelOffsets[point.id];
      if (!stored) continue;
      const [isoDx, isoDy] = rotateVector(activeSheetTransform, stored.x, stored.y);
      const pxDx = isoDx * (frontLayerSizePx.width / layerScale);
      const pxDy = isoDy * (frontLayerSizePx.height / layerScale);
      const labelWidthPx = getFrontLabelWidthPx(point);
      const cornerPxX = pxDx - labelWidthPx / 2;
      const cornerPxY = pxDy - FRONT_LABEL_HEIGHT_PX / 2;
      const offsetLayerPctX = cornerPxX / frontLayerSizePx.width * 100;
      const offsetLayerPctY = cornerPxY / frontLayerSizePx.height * 100;
      next[point.id] = { x: point.x + offsetLayerPctX, y: point.y + offsetLayerPctY };
    }
    return next;
  }, [storedFrontLabelOffsets, activeSheetTransform, sheetPoints, frontLayerSizePx, getFrontLabelWidthPx]);
  const labelPositionsByLayout: Record<string, Record<string, { x: number; y: number }>> = {
    ...(detailLabelPositionsByScan[scan.id] ?? {}),
    front: displayedFrontLabelPositions,
  };
  const handleLabelPositionsChange = useCallback((layoutId: string, positions: Record<string, { x: number; y: number }>, reportedLayerSize?: { width: number; height: number }) => {
    if (layoutId === 'front') {
      const canonical: Record<string, { x: number; y: number }> = {};
      /* frontLayerSizePx(부모 state)는 ServicePreview 가 탭 전환으로
         통째로 리마운트되면 한 렌더 늦게(기본값 1×1에서) 갱신된다 -- 그
         사이에 CorrectionPoints 가 자기 실측값으로 이미 옳게 라벨을
         앉혀 놓고 이 콜백으로 위치를 보고하면, 부모가 그 늦은(틀린) 값을
         써서 되돌려 계산해 canonical 을 오염시켰다(탭을 옮겼다 오면
         지시선이 다른 곳에 붙는 증상). 이 콜백이 매번 받는 layerSize는
         CorrectionPoints 자신이 그 순간 실측한 값이라 항상 맞으니, 있으면
         그걸 쓰고 없을 때만 부모 state 로 물러선다. */
      const layerSizePx = reportedLayerSize && reportedLayerSize.width && reportedLayerSize.height ? reportedLayerSize : frontLayerSizePx;
      const layerScale = Math.sqrt(layerSizePx.width * layerSizePx.height) || 1;
      for (const [id, position] of Object.entries(positions)) {
        const point = sheetPoints.find((item) => item.id === id);
        if (!point) continue;
        const labelWidthPx = getFrontLabelWidthPx(point);
        const cornerOffsetLayerPctX = position.x - point.x;
        const cornerOffsetLayerPctY = position.y - point.y;
        const cornerPxX = cornerOffsetLayerPctX / 100 * layerSizePx.width;
        const cornerPxY = cornerOffsetLayerPctY / 100 * layerSizePx.height;
        const pxDx = cornerPxX + labelWidthPx / 2;
        const pxDy = cornerPxY + FRONT_LABEL_HEIGHT_PX / 2;
        const isoDx = pxDx * (layerScale / layerSizePx.width);
        const isoDy = pxDy * (layerScale / layerSizePx.height);
        const [canonicalDx, canonicalDy] = unrotateVector(activeSheetTransform, isoDx, isoDy);
        canonical[id] = { x: canonicalDx, y: canonicalDy };
      }
      setFrontLabelPositionsByScan((current) => {
        const previous = current[scan.id];
        if (previous && JSON.stringify(previous) === JSON.stringify(canonical)) return current;
        return { ...current, [scan.id]: canonical };
      });
      return;
    }
    setDetailLabelPositionsByScan((current) => {
      const currentForScan = current[scan.id] ?? {};
      /* CorrectionPoints 는 매번 새 객체를 만들어 올리므로 참조 비교로는
         항상 "달라짐" 이 된다. 값까지 같으면 그대로 두어야, 이 setState 가
         부모를 리렌더 -> 자식 리렌더 -> 다시 새 객체로 이어지는 루프의
         꼬리를 확실히 끊는다(콜백 참조는 이미 위에서 고정해 뒀지만, 그와
         별개로 여기서도 막아 둔다). */
      const previous = currentForScan[layoutId];
      if (previous && JSON.stringify(previous) === JSON.stringify(positions)) return current;
      return { ...current, [scan.id]: { ...currentForScan, [layoutId]: positions } };
    });
  }, [scan.id, activeSheetTransform, sheetPoints, frontLayerSizePx, getFrontLabelWidthPx]);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const removeAddedPoint = (id: string) => setAddedPoints((current) => current.filter((item) => item.id !== id));
  const addPointAt = async (xNorm: number, yNorm: number) => {
    setSampling(true); setSampleError(null);
    try {
      [xNorm, yNorm] = invertSheetPoint(activeSheetTransform, xNorm, yNorm);
      /* 클릭 좌표는 화면에 보이는 이미지 기준이다. 색 역산은 편차 스캔에서만
         가능하므로, 제품데이터를 보고 있으면 변환을 되짚어 스캔 좌표로 보낸다. */
      let sampleX = xNorm; let sampleY = yNorm;
      if (onProduct && alignment) {
        const [productW, productH] = alignment.productSize;
        const [scanW, scanH] = alignment.scanSize;
        const mapped = invertAffinePoint(alignment.matrix, xNorm / 100 * productW, yNorm / 100 * productH);
        if (!mapped) { setSampleError('정렬 정보가 올바르지 않습니다.'); return; }
        sampleX = mapped[0] / scanW * 100;
        sampleY = mapped[1] / scanH * 100;
        if (sampleX < 0 || sampleX > 100 || sampleY < 0 || sampleY > 100) {
          setSampleError('스캔 범위를 벗어난 지점입니다.'); return;
        }
      }
      const form = new FormData();
      form.append('file', scan.file);
      form.append('x', String(sampleX));
      form.append('y', String(sampleY));
      const response = await fetch(`${API_BASE}/api/sample`, { method: 'POST', body: form });
      // 엔진 서버가 옛 코드로 떠 있으면 /api/sample 이 없어 HTML 404 가 온다.
      // 그대로 json() 하면 파싱 예외라 원인이 안 보여서, 상태코드로 알려준다.
      const body = await response.text();
      let data: { error?: string; xPx: number; yPx: number; x: number; y: number; value: number };
      try {
        data = JSON.parse(body) as typeof data;
      } catch {
        data = { error: response.status === 404
          ? '보정 포인트 API를 찾을 수 없습니다. 로컬 엔진 서버를 최신 코드로 다시 시작하세요.'
          : `엔진 서버 응답을 읽을 수 없습니다. (HTTP ${response.status})` } as typeof data;
      }
      if (!response.ok) { setSampleError(data?.error || '편차값을 추정하지 못했습니다.'); return; }
      /* 응답은 스캔 좌표다. 엔진 포인트와 같은 규칙으로 제품 좌표도 함께 담아 둔다. */
      let productCoords: { xProduct?: number; yProduct?: number } = {};
      if (alignment) {
        const [productW, productH] = alignment.productSize;
        const [productX, productY] = mapAffinePoint(alignment.matrix, data.xPx, data.yPx);
        productCoords = {
          xProduct: productX / productW * 100,
          yProduct: productY / productH * 100,
        };
      }
      setAddedPoints((current) => {
        /* 이제 addedPoints 가 탭을 넘나들며 유지되니, 마운트마다 0부터
           다시 세는 ref 대신 지금 목록에 있는 가장 큰 번호 다음 값을
           쓴다 -- 안 그러면 리마운트 후 새 점이 기존 점과 같은 id(M-01
           등)를 다시 받아 충돌한다. */
        const nextSeq = current.reduce((max, item) => {
          const match = /^M-(\d+)$/.exec(item.id);
          return match ? Math.max(max, Number(match[1])) : max;
        }, 0) + 1;
        return [...current, {
          id: `M-${String(nextSeq).padStart(2, '0')}`,
          xPx: data.xPx, yPx: data.yPx, x: data.x, y: data.y, ...productCoords,
          value: data.value, labelColor: 'white', confidence: 'colormap', source: 'colormap',
        }];
      });
    } catch (error) {
      setSampleError(error instanceof Error ? error.message : '엔진 서버에 연결하지 못했습니다.');
    } finally {
      setSampling(false);
    }
  };
  /* 보정치 수동 수정 이력. 백엔드 로컬 DB(SQLite)에서 품번 기준으로 불러온다 —
     스캔 데이터를 외부로 보낼 수 없는 정책이라 로컬 서버 안에서만 오간다. */
  const [history, setHistory] = useState<CorrectionHistoryEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [pendingPointIds, setPendingPointIds] = useState<Set<string>>(() => new Set());
  const pendingPointIdsRef = useRef(new Set<string>());
  const [deletingEntryIds, setDeletingEntryIds] = useState<Set<number>>(() => new Set());
  const deletingEntryIdsRef = useRef(new Set<number>());
  /* 이펙트 안에서 곧바로 setState 를 호출하면 린트가 캐스케이드 렌더 위험으로 잡아내므로,
     자동 로드(마운트/품번 변경)는 로딩 표시 없이 fetchHistory 만 부르고, 새로고침 버튼처럼
     사용자 조작에서 시작하는 경우에만 loadHistory 로 로딩 상태를 켠다. */
  const normalizeHistoryEntry = (entry: CorrectionHistoryEntry): CorrectionHistoryEntry => ({
    ...entry,
    oldMode: entry.oldMode ?? null,
    newMode: entry.newMode ?? (entry.newValue == null ? 'auto' : null),
    coefficient: entry.coefficient ?? null,
    action: entry.action ?? 'edit',
    sourceEntryId: entry.sourceEntryId ?? null,
  });
  const fetchHistory = useCallback(() => {
    const query = new URLSearchParams({ partNo: scan.partNo, scanName: scan.name });
    return fetch(`${API_BASE}/api/corrections?${query}`)
      .then(async (response) => {
        const data = await response.json() as { entries?: CorrectionHistoryEntry[]; error?: string };
        if (!response.ok) throw new Error(data.error || '보정 이력을 불러오지 못했습니다.');
        return data;
      })
      .then((data) => { setHistory((data.entries || []).map(normalizeHistoryEntry)); setHistoryError(null); })
      .catch((error: unknown) => setHistoryError(error instanceof Error ? error.message : '보정 이력을 불러오지 못했습니다.'));
  }, [scan.name, scan.partNo]);
  const loadHistory = () => { setHistoryLoading(true); void fetchHistory().finally(() => setHistoryLoading(false)); };
  useEffect(() => { void fetchHistory(); }, [fetchHistory]);
  const recordCorrection = async ({ pointId, oldValue, newValue, oldMode, newMode, action, sourceEntryId }: {
    pointId: string;
    oldValue: number;
    newValue: number;
    oldMode: CorrectionMode;
    newMode: CorrectionMode;
    action: CorrectionAction;
    sourceEntryId?: number;
  }) => {
    const response = await fetch(`${API_BASE}/api/corrections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ partNo: scan.partNo, scanName: scan.name, pointId, oldValue, newValue, oldMode, newMode, coefficient, action, sourceEntryId, worker }),
    });
    const data = await response.json() as CorrectionHistoryEntry & { error?: string };
    if (!response.ok) throw new Error(data.error || '보정 이력을 저장하지 못했습니다.');
    const saved = normalizeHistoryEntry(data);
    setHistory((current) => [saved, ...current.filter((entry) => entry.id !== saved.id)].sort((a, b) => b.id - a.id).slice(0, 200));
    return saved;
  };
  /* 시트에는 엔진이 찾은 포인트와 작업자가 찍은 포인트를 함께 올린다.
     표시 여부도 합친 목록 기준으로 계산해야 추가한 포인트의 라벨이 숨김 처리되지 않는다. */
  const sheetAddedPoints = sheetPoints.filter((point) => point.source === 'colormap');
  /* 제로 폴리라인도 포인트와 같은 규칙으로 프레임 % 로 옮긴다. 스캔 원본은 픽셀 좌표라
     [scanW, scanH] 로 나눠 %, 제품데이터는 alignment 행렬로 옮긴 뒤 [productW, productH] 로 % 를 낸다.
     알림: 여기서 알고 있는 alignment 는 shear=0 (b=c=0) 인 축정렬 아핀이라 add point 와 같은 형태를 쓴다. */
  const zeroLineSplineSegments = useMemo(() => (result.zeroLines || []).map((line, lineIndex) => {
    const edit = draftZeroEdits.find((item) => item.index === lineIndex);
    if (Array.isArray(edit?.splineSegments)) return edit.splineSegments;
    return edit?.spline ? Array.from({ length: Math.max((line.points || []).length - 1, 0) }, (_, index) => index) : [];
  }), [result.zeroLines, draftZeroEdits]);
  const sheetZeroLines = useMemo<[number, number][][]>(() => {
    if (!showZero || !hasZeroVector) return [];
    const scanW = result.source.width;
    const scanH = result.source.height;
    if (onProduct && alignment) {
      const [productW, productH] = alignment.productSize;
      return editedZeroLinePixels
        .map((line) => line
          .map(([xPx, yPx]) => {
            const [productX, productY] = mapAffinePoint(alignment.matrix, xPx, yPx);
            const productXPct = productX / productW * 100;
            const productYPct = productY / productH * 100;
            return transformSheetPoint(activeSheetTransform, productXPct, productYPct);
          }));
    }
    return editedZeroLinePixels.map((line) => line.map(([xPx, yPx]) => transformSheetPoint(activeSheetTransform, xPx / scanW * 100, yPx / scanH * 100)));
  }, [showZero, hasZeroVector, result.source.width, result.source.height, onProduct, alignment, editedZeroLinePixels, activeSheetTransform]);
  const moveZeroPoint = (lineIndex: number, pointIndex: number, dxPercent: number, dyPercent: number) => {
    [dxPercent, dyPercent] = invertSheetDelta(activeSheetTransform, dxPercent, dyPercent);
    const scanW = result.source.width;
    const scanH = result.source.height;
    let dx = dxPercent * scanW / 100;
    let dy = dyPercent * scanH / 100;
    if (onProduct && alignment) {
      const [productW, productH] = alignment.productSize;
      const mapped = invertAffineDelta(alignment.matrix, dxPercent * productW / 100, dyPercent * productH / 100);
      if (!mapped) return;
      [dx, dy] = mapped;
    }
    setDraftZeroEdits((current) => {
      const previous = current.find((item) => item.index === lineIndex) || { index: lineIndex, dx: 0, dy: 0 };
      if (Array.isArray(previous.vertices) && previous.vertices[pointIndex]) {
        const vertices = previous.vertices.map((point, index) => index === pointIndex ? [point[0] + dx, point[1] + dy] as [number, number] : point);
        const next: ZeroEdit = { ...previous, vertices };
        return [...current.filter((item) => item.index !== lineIndex), next].sort((left, right) => left.index - right.index);
      }
      const key = String(pointIndex);
      const oldPoint = previous.points?.[key] || { dx: 0, dy: 0 };
      const next: ZeroEdit = { ...previous, points: { ...previous.points, [key]: { dx: oldPoint.dx + dx, dy: oldPoint.dy + dy } } };
      return [...current.filter((item) => item.index !== lineIndex), next].sort((left, right) => left.index - right.index);
    });
  };
  const toggleZeroSplineSegment = (lineIndex: number, segmentIndex: number) => setDraftZeroEdits((current) => {
    const previous = current.find((item) => item.index === lineIndex) || { index: lineIndex, dx: 0, dy: 0 };
    const fallbackCount = Math.max(editedZeroLinePixels[lineIndex]?.length - 1, 0);
    const existing = Array.isArray(previous.splineSegments)
      ? previous.splineSegments
      : (previous.spline ? Array.from({ length: fallbackCount }, (_, index) => index) : []);
    const splineSegments = existing.includes(segmentIndex)
      ? existing.filter((index) => index !== segmentIndex)
      : [...existing, segmentIndex].sort((left, right) => left - right);
    const next: ZeroEdit = { ...previous, spline: undefined, splineSegments };
    return [...current.filter((item) => item.index !== lineIndex), next].sort((left, right) => left.index - right.index);
  });
  const addZeroPoint = (lineIndex: number, segmentIndex: number, xPercent: number, yPercent: number) => {
    [xPercent, yPercent] = invertSheetPoint(activeSheetTransform, xPercent, yPercent);
    const scanW = result.source.width;
    const scanH = result.source.height;
    let x = xPercent * scanW / 100;
    let y = yPercent * scanH / 100;
    if (onProduct && alignment) {
      const [productW, productH] = alignment.productSize;
      const mapped = invertAffinePoint(alignment.matrix, xPercent * productW / 100, yPercent * productH / 100);
      if (!mapped) return;
      [x, y] = mapped;
    }
    const visible = editedZeroLinePixels[lineIndex];
    if (!visible || visible.length < 2) return;
    setDraftZeroEdits((current) => {
      const previous = current.find((item) => item.index === lineIndex) || { index: lineIndex, dx: 0, dy: 0 };
      const vertices = visible.map((point) => [...point] as [number, number]);
      vertices.splice(Math.min(segmentIndex + 1, vertices.length), 0, [x, y]);
      const oldSegments = Array.isArray(previous.splineSegments)
        ? previous.splineSegments
        : (previous.spline ? Array.from({ length: Math.max(visible.length - 1, 0) }, (_, index) => index) : []);
      const splineSegments = oldSegments.flatMap((index) => index < segmentIndex ? [index] : index > segmentIndex ? [index + 1] : [index, index + 1]);
      const next: ZeroEdit = { ...previous, dx: 0, dy: 0, points: undefined, vertices, spline: undefined, splineSegments };
      return [...current.filter((item) => item.index !== lineIndex), next].sort((left, right) => left.index - right.index);
    });
  };
  const deleteZeroPoint = (lineIndex: number, pointIndex: number) => {
    const visible = editedZeroLinePixels[lineIndex];
    if (!visible || visible.length < 2) return;
    const first = visible[0];
    const last = visible[visible.length - 1];
    const closed = (first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2 < 1e-8;
    const uniqueVertices = closed ? visible.slice(0, -1) : visible;
    if (uniqueVertices.length <= (closed ? 3 : 2)) return;
    const deleteIndex = closed && pointIndex === visible.length - 1 ? 0 : pointIndex;
    if (deleteIndex < 0 || deleteIndex >= uniqueVertices.length) return;
    setDraftZeroEdits((current) => {
      const previous = current.find((item) => item.index === lineIndex) || { index: lineIndex, dx: 0, dy: 0 };
      const segmentCount = closed ? uniqueVertices.length : uniqueVertices.length - 1;
      const oldSegments = new Set(Array.isArray(previous.splineSegments)
        ? previous.splineSegments
        : (previous.spline ? Array.from({ length: segmentCount }, (_, index) => index) : []));
      const remainingIndices = uniqueVertices.map((_, index) => index).filter((index) => index !== deleteIndex);
      const vertices = remainingIndices.map((index) => [...uniqueVertices[index]] as [number, number]);
      const splineSegments: number[] = [];
      if (closed) {
        for (let index = 0; index < remainingIndices.length; index += 1) {
          const start = remainingIndices[index];
          const end = remainingIndices[(index + 1) % remainingIndices.length];
          const curved = end === (start + 1) % uniqueVertices.length
            ? oldSegments.has(start)
            : oldSegments.has((deleteIndex - 1 + uniqueVertices.length) % uniqueVertices.length) || oldSegments.has(deleteIndex);
          if (curved) splineSegments.push(index);
        }
        vertices.push([...vertices[0]] as [number, number]);
      } else {
        for (let index = 0; index < vertices.length - 1; index += 1) {
          const curved = deleteIndex === 0 ? oldSegments.has(index + 1)
            : deleteIndex === uniqueVertices.length - 1 ? oldSegments.has(index)
              : index < deleteIndex - 1 ? oldSegments.has(index)
                : index === deleteIndex - 1 ? oldSegments.has(deleteIndex - 1) || oldSegments.has(deleteIndex)
                  : oldSegments.has(index + 1);
          if (curved) splineSegments.push(index);
        }
      }
      const next: ZeroEdit = { ...previous, dx: 0, dy: 0, points: undefined, vertices, spline: undefined, splineSegments };
      return [...current.filter((item) => item.index !== lineIndex), next].sort((left, right) => left.index - right.index);
    });
  };
  const visiblePointIds = new Set(sheetPoints.filter((point) => !hiddenPointIds.has(point.id)).map((point) => point.id));
  const createAnnotation = (annotation: Annotation) => setAnnotations((current) => [...current, annotation]);
  const commitAnnotation = (annotation: Annotation) => setAnnotations((current) => current.map((item) => item.id === annotation.id ? annotation : item));
  const deleteAnnotation = (id: string) => setAnnotations((current) => current.filter((item) => item.id !== id));
  const clearAnnotations = () => { setAnnotations(() => []); setSelectedAnnotationId(null); setTool('select'); };
  /* 브라우저 인쇄를 그대로 쓴다. 캔버스로 굽지 않아 글자가 벡터로 남고 추가 의존성도 없다.
     인쇄 대화상자에서 '대상: PDF로 저장'을 고르면 된다. */
  /* 현업 엑셀 양식으로 내보낸다. 보정량은 화면이 들고 있는 최종값을
     그대로 보내야 시트와 엑셀이 어긋나지 않는다. */
  const [excelState, setExcelState] = useState<'idle' | 'saving' | 'error'>('idle');
  const [excelError, setExcelError] = useState<string | null>(null);
  const saveExcel = async () => {
    const analysisId = result.analysisId;
    if (!analysisId) { setExcelError('분석 결과가 없습니다.'); return; }
    setExcelState('saving'); setExcelError(null);
    const corrections: Record<string, number> = {};
    for (const point of sheetPoints) {
      if (!visiblePointIds.has(point.id)) continue;
      corrections[point.id] = displayFor(point);
    }
    try {
      const response = await fetch(`${API_BASE}/api/sheet-excel`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysisId, corrections,
          filename: `${scan.partNo || 'ADC'}_보정시트`,
          /* 머리말은 파일명 규칙에서 읽은 값을 그대로 보낸다.
             화면에 보이는 것과 엑셀이 달라지면 안 된다. */
          meta: {
            partNo: result.naming?.part_no || scan.partNo,
            partName: result.naming?.part_name || '',
            process: result.naming?.process || '',
            controlNo: result.naming?.control_no || '',
            appliedAt: result.naming?.applied_at || '',
            coefficient,
          },
        }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(data.error || '엑셀을 만들지 못했습니다.');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${scan.partNo || 'ADC'}_보정시트.xlsx`;
      link.click();
      URL.revokeObjectURL(url);
      setExcelState('idle');
    } catch (err) {
      setExcelError(String((err as Error).message || err));
      setExcelState('error');
    }
  };

  const sheetRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const savePdf = () => {
    setSelectedAnnotationId(null);
    setTool('select');
    setDetailMode(false);
    setLabelAreaMode(null);
    /* 시트의 조상만 남기고 형제는 인쇄에서 빼야 한다. visibility 로 감추면 자리를
       그대로 차지해 빈 둘째 장이 생긴다. */
    const chain: HTMLElement[] = [];
    for (let node = sheetRef.current?.parentElement; node && node !== document.body; node = node.parentElement) {
      node.classList.add('adc-print-chain');
      chain.push(node);
    }
    document.body.classList.add('adc-printing');
    const cleanup = () => {
      document.body.classList.remove('adc-printing');
      chain.forEach((node) => node.classList.remove('adc-print-chain'));
      window.removeEventListener('afterprint', cleanup);
    };
    window.addEventListener('afterprint', cleanup);
    /* 전환이 끝나 자리가 확정된 뒤에 인쇄해야 중간값이 찍히지 않는다. */
    window.setTimeout(() => window.print(), 80);
  };
  /* 엑셀 저장 — 아진산업이 실제로 쓰는 보정시트_양식.xlsx / 기존 보정내용.xlsx 파일을 직접 열어
     열 너비·행 높이·병합 범위·페이지 나누기를 그대로 뽑아냈다 (30개 열, 전부 폭 4.375 / 행 높이
     13.5pt / 표제란은 6행 병합 3쌍(관리NO·PART NAME, 공정·PART NO, 원소재·적용일자) / 블록 하나당
     정확히 40행, 사이 여백 없이 바로 다음 블록이 시작되고 그 경계에 페이지 나누기가 들어간다).
     표제란은 실제 셀(글꼴·크기 그대로)로, 도면+포인트+주석은 화면 그대로 캡처한 이미지로 그 아래
     40행 안에 맞춰 넣는다. 기존 파일을 골라두면 그 파일 끝에 같은 규칙으로 이어붙인다. */
  const [excelFile, setExcelFile] = useState<File | null>(null);
  const [excelSaving, setExcelSaving] = useState(false);
  const excelInputRef = useRef<HTMLInputElement>(null);
  /* 서버 build_sheet 경로로 저장한다. 라벨은 편집 가능한 텍스트박스로, 지시선은 라벨을
     따라 늘어나는 attached connector 로 나오고, 라벨 위치는 서버가 자동으로 잡는다.
     이전에는 html2canvas 로 프리뷰를 사진 찍어 넣었는데 그 방식은 라벨이 픽셀로 굳어
     Excel 에서 값 수정이 불가능했다. */
  const saveSheetExcel = async () => {
    if (excelSaving) return;
    setExcelSaving(true);
    setExcelError(null);
    try {
      // 제품데이터가 없으면 현재 스캔(라벨 제거본 우선)을 시트 정면도로 쓴다.
      // 이 경우 포인트도 스캔 좌표계로 이미 계산되어 있어 별도 변환이 필요 없다.
      const sheetImageUrl = renderedSheetImage.url;
      if (!sheetImageUrl) throw new Error('시트에 넣을 이미지를 찾을 수 없습니다.');
      const productBlob = await (await fetch(sheetImageUrl)).blob();

      const visiblePointsList = sheetPoints.filter((point) => visiblePointIds.has(point.id));
      const payloadPoints = visiblePointsList.map((point) => ({
        id: point.id,
        text: formatCorrection(displayFor(point)),
        x: point.x,
        y: point.y,
      }));

      const payloadAnnotations = annotations.map((annotation) => ({
        id: annotation.id,
        kind: annotation.kind,
        x: annotation.x,
        y: annotation.y,
        w: annotation.w,
        h: annotation.h,
        text: annotation.text ?? '',
        fontSize: annotation.fontSize ?? null,
        fontFamily: annotation.fontFamily ?? null,
        color: annotation.color ?? DEFAULT_ANNOTATION_COLOR,
      }));

      /* Detail 크롭 영역과 그 label 을 그대로 넘겨 백엔드가 별도 뷰로 자르도록 한다. */
      const frontLayout = sheetLayouts.find((layout) => layout.kind === 'front');
      const detailLayoutById = new Map(sheetLayouts.filter((layout) => layout.kind === 'detail').map((layout) => [layout.regionId, layout]));
      /* UI 가 실제로 그린(그리고 사용자가 드래그해 옮긴) 라벨 위치를 그대로
         엑셀에 넘긴다 — CorrectionPoints 가 매 렌더마다 labelPositionsByLayout
         로 이미 올려 둔 state 라, 저장 시점에 DOM 을 다시 재는 것보다 정확
         하다(border/padding·숨김 탭이면 0 인 rect 같은 문제가 없다). */
      const frontLabels = frontLayout ? (labelPositionsByLayout[frontLayout.id] ?? {}) : {};
      const payloadDetails = detailRegions.map((region) => {
        const layout = detailLayoutById.get(region.id);
        return {
          id: region.id,
          label: region.label,
          x: region.x,
          y: region.y,
          w: region.w,
          h: region.h,
          /* placement: 잘라낸 뷰가 시트 캔버스의 어느 자리에 어느 크기로 놓이는지. */
          placement: layout ? { x: layout.x, y: layout.y, w: layout.w, h: layout.h } : null,
          /* labels: 이 Detail 뷰가 실제로 화면에 그린 라벨 위치. */
          labels: layout ? (labelPositionsByLayout[layout.id] ?? {}) : {},
        };
      });

      const payload = {
        partNumber: scan.partNo,
        title: {
          heading: sheetTitle.heading,
          managementLabel: sheetTitle.managementLabel,
          managementNo: sheetTitle.managementNo,
          partNameLabel: sheetTitle.partNameLabel,
          partName: sheetTitle.partName,
          processLabel: sheetTitle.processLabel,
          process: sheetTitle.process,
          partNoLabel: sheetTitle.partNoLabel,
          partNo: sheetTitle.partNo,
          materialLabel: sheetTitle.materialLabel,
          material: sheetTitle.material,
          appliedDateLabel: sheetTitle.appliedDateLabel,
          appliedDate: sheetTitle.appliedDate,
        },
        titleFonts: {
          heading: extractFontName(sheetTitleFonts.heading),
          management_label: extractFontName(sheetTitleFonts.managementLabel),
          management_no: extractFontName(sheetTitleFonts.managementNo),
          part_name_label: extractFontName(sheetTitleFonts.partNameLabel),
          part_name: extractFontName(sheetTitleFonts.partName),
          process_label: extractFontName(sheetTitleFonts.processLabel),
          process: extractFontName(sheetTitleFonts.process),
          part_no_label: extractFontName(sheetTitleFonts.partNoLabel),
          part_no: extractFontName(sheetTitleFonts.partNo),
          material_label: extractFontName(sheetTitleFonts.materialLabel),
          material: extractFontName(sheetTitleFonts.material),
          applied_date_label: extractFontName(sheetTitleFonts.appliedDateLabel),
          applied_date: extractFontName(sheetTitleFonts.appliedDate),
        },
        titleFontSizes: {
          heading: sheetTitleFontSizes.heading,
          management_label: sheetTitleFontSizes.managementLabel,
          management_no: sheetTitleFontSizes.managementNo,
          part_name_label: sheetTitleFontSizes.partNameLabel,
          part_name: sheetTitleFontSizes.partName,
          process_label: sheetTitleFontSizes.processLabel,
          process: sheetTitleFontSizes.process,
          part_no_label: sheetTitleFontSizes.partNoLabel,
          part_no: sheetTitleFontSizes.partNo,
          material_label: sheetTitleFontSizes.materialLabel,
          material: sheetTitleFontSizes.material,
          applied_date_label: sheetTitleFontSizes.appliedDateLabel,
          applied_date: sheetTitleFontSizes.appliedDate,
        },
        pointFontFamily: extractFontName(pointLabelFont),
        points: payloadPoints,
        annotations: payloadAnnotations,
        details: payloadDetails,
        /* 정면도 picture 를 시트 캔버스 어디에 얼마 크기로 놓을지. UI 와 같은 % 좌표로 넘긴다. */
        frontPlacement: frontLayout ? { x: frontLayout.x, y: frontLayout.y, w: frontLayout.w, h: frontLayout.h } : null,
        /* frontLabels: 정면도가 실제로 화면에 그린 라벨 위치. */
        frontLabels,
        /* 제로라인도 점과 같은 프레임(% 좌표) 기준으로 넘긴다. 백엔드가 파트
           이미지에 굽지 않고 별도 벡터 도형으로 그려서, 엑셀에서 이미지와
           따로 선택/삭제할 수 있다. */
        zeroLines: showZero ? sheetZeroLines : [],
      };

      const form = new FormData();
      form.append('payload', JSON.stringify(payload));
      form.append('product', productBlob, 'product.png');
      if (excelFile) form.append('previous', excelFile, excelFile.name);

      const response = await fetch(`${API_BASE}/api/sheet`, { method: 'POST', body: form });
      if (!response.ok) {
        const errorData = await response.json().catch(() => null) as { error?: string } | null;
        throw new Error(errorData?.error || `서버 오류 (HTTP ${response.status})`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = excelFileName(sheetTitle.managementNo);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      setExcelFile(null);
      if (excelInputRef.current) excelInputRef.current.value = '';
    } catch (error) {
      setExcelError(error instanceof Error ? error.message : '엑셀로 저장하지 못했습니다.');
    } finally {
      setExcelSaving(false);
    }
  };
  /* 새 주석은 늘 기본색으로 그려지고, 색 변경은 주석을 고른 뒤 팔레트를 누르는 동작으로만 일어난다. */
  const selectedColor = selectedAnnotationId ? (annotations.find((item) => item.id === selectedAnnotationId)?.color ?? DEFAULT_ANNOTATION_COLOR) : null;
  const changeColor = (hex: string) => {
    if (!selectedAnnotationId) return;
    setAnnotations((current) => current.map((item) => item.id === selectedAnnotationId ? { ...item, color: hex } : item));
  };
  const displayFor = useCallback((point: PointResult) => pointOverrides[point.id] !== undefined ? pointOverrides[point.id] : -(point.value * coefficient), [coefficient, pointOverrides]);
  const formatCorrection = useCallback((value: number) => `${value > 0 ? '+' : ''}${value.toFixed(1)}`, []);
  const maxCorrection = useMemo(() => points.length ? Math.max(...points.map((point) => Math.abs(displayFor(point)))) : 0, [displayFor, points]);
  const overrideCount = useMemo(() => points.filter((point) => pointOverrides[point.id] !== undefined).length, [points, pointOverrides]);
  const applyCorrection = async (id: string, targetOverride: number | null, action: CorrectionAction, options?: { skipRecord?: boolean }) => {
    const point = sheetPoints.find((item) => item.id === id);
    if (!point) {
      setHistoryError(`${id} 포인트가 현재 시트에 없어 적용할 수 없습니다.`);
      return false;
    }
    if (pendingPointIdsRef.current.has(id)) return false;
    const oldMode: CorrectionMode = pointOverrides[id] === undefined ? 'auto' : 'manual';
    const newMode: CorrectionMode = targetOverride === null ? 'auto' : 'manual';
    const oldValue = displayFor(point);
    const newValue = targetOverride ?? -(point.value * coefficient);
    if (oldMode === newMode && Math.abs(oldValue - newValue) < 0.0001) {
      setHistoryError(null);
      return true;
    }
    pendingPointIdsRef.current.add(id);
    setPendingPointIds(new Set(pendingPointIdsRef.current));
    setHistoryError(null);
    try {
      if (!options?.skipRecord) {
        await recordCorrection({ pointId: id, oldValue, newValue, oldMode, newMode, action });
      }
      onOverrideChange(id, targetOverride);
      return true;
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : '보정값을 적용하지 못했습니다.');
      return false;
    } finally {
      pendingPointIdsRef.current.delete(id);
      setPendingPointIds(new Set(pendingPointIdsRef.current));
    }
  };
  const handleOverrideChange = (id: string, value: number | null) => {
    void applyCorrection(id, value, value === null ? 'reset_auto' : 'edit');
  };
  /* "원래 값 복원" = 이 기록이 무엇이었든 상관없이 해당 포인트를 엔진이 계산한
     자동값으로 되돌린다. 포인트 라벨의 ↺ 초기화 버튼과 동일한 동작이지만, 이력 패널에서
     누른 복원은 그 자체로 새 이력을 남기지 않는다 — 되돌리는 동작까지 기록되면
     이력이 계속 늘어나기만 해서 원래 무엇을 되돌렸는지 추적하기 어려워지기 때문. */
  const restoreHistoryEntry = (entry: CorrectionHistoryEntry) => {
    void applyCorrection(entry.pointId, null, 'reset_auto', { skipRecord: true });
  };
  const deleteHistoryEntry = async (entry: CorrectionHistoryEntry) => {
    if (deletingEntryIdsRef.current.has(entry.id)) return;
    deletingEntryIdsRef.current.add(entry.id);
    setDeletingEntryIds(new Set(deletingEntryIdsRef.current));
    setHistoryError(null);
    try {
      const response = await fetch(`${API_BASE}/api/corrections?id=${entry.id}`, { method: 'DELETE' });
      const data = await response.json() as { error?: string };
      if (!response.ok) throw new Error(data.error || '이력을 삭제하지 못했습니다.');
      setHistory((current) => current.filter((item) => item.id !== entry.id));
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : '이력을 삭제하지 못했습니다.');
    } finally {
      deletingEntryIdsRef.current.delete(entry.id);
      setDeletingEntryIds(new Set(deletingEntryIdsRef.current));
    }
  };
  const handleClearAllOverrides = async () => {
    const results = await Promise.all(Object.keys(pointOverrides).map((id) => applyCorrection(id, null, 'reset_all')));
    if (results.length > 0 && results.every(Boolean)) onClearAllOverrides();
  };
  /* 보정시트에 들어가는 그림 — 정렬된 제품데이터가 있으면 그쪽을 우선한다.
     스캔 모드에서 zeroOverlay(래스터)만 있고 벡터가 없으면 종전대로 오버레이를 그림 자체에 굽는다.
     벡터가 있으면 굳이 굽지 않고 깨끗한 이미지를 쓰고, 위에 SVG 폴리라인으로 얹는다. */
  const baseImage = renderedSheetImage.url;
  const toggleZeroEditor = () => {
    setZeroPanel((current) => {
      const next = !current;
      if (!next) {
        setZeroPointAddMode(false);
        setZeroPointDeleteMode(false);
      }
      return next;
    });
    setShowZero(true);
  };
  return <section className="page page--service">
    <div className="page-heading page-heading--compact"><div><h2>보정 시트 작성</h2></div></div>
    <div className="service-grid"><div className="correction-card card">
      <div className="viewer-toolbar"><div><span className="status status--done"><Check size={13} /> 레이아웃 편집</span><b>{scan.partNo} · 보정 작업 지시도</b></div><div className="layer-toggles"><button className={onProduct ? 'active blue' : ''} onClick={() => setUseProduct(!useProduct)} disabled={!productReady} title={productReady ? '제품데이터 위에 보정치를 올립니다' : '이 품번의 제품데이터가 등록되어 있지 않습니다'}><i /> 제품데이터</button><button className={showPoints ? 'active orange' : ''} onClick={() => setShowPoints(!showPoints)}><i /> 보정치</button><button className={showZero && zeroReady ? 'active green' : ''} onClick={() => setShowZero(!showZero)} disabled={!zeroReady} title={!zeroReady ? '이 스캔에는 제로라인 데이터가 없습니다' : (onProduct && !hasZeroVector ? '제품데이터 위에 겹칠 제로라인 벡터가 없습니다' : '')}><i /> 제로라인</button><button className={showAnnotations ? 'active amber' : ''} onClick={() => { setShowAnnotations(!showAnnotations); setTool('select'); setSelectedAnnotationId(null); }}><i /> 주석</button></div></div>
      <div className="sheet-image-toolbar" role="toolbar" aria-label="보정시트 이미지 방향">
        <b>이미지 방향</b>
        <button type="button" onClick={rotateSheet} title="이미지와 보정 위치를 함께 시계 방향으로 90° 회전">90° 회전</button>
        <button type="button" className={sheetHorizontalFlipped ? 'is-active' : ''} onClick={flipSheetHorizontal} aria-pressed={sheetHorizontalFlipped}>좌우 뒤집기</button>
        <button type="button" className={sheetVerticalFlipped ? 'is-active' : ''} onClick={flipSheetVertical} aria-pressed={sheetVerticalFlipped}>상하 뒤집기</button>
        <button type="button" onClick={() => setSheetTransform(IDENTITY_SHEET_TRANSFORM)} disabled={sheetTransformKey(sheetTransform) === sheetTransformKey(IDENTITY_SHEET_TRANSFORM)}>원래 방향</button>
        <span>{renderedSheetImage.busy ? '이미지 변환 중…' : `${activeSheetTransform.rotation}°`}</span>
        {renderedSheetImage.error && <em>{renderedSheetImage.error}</em>}
      </div>
      <AnnotationToolbar tool={tool} setTool={(next) => { setShowAnnotations(true); setTool(next); setDetailMode(false); setLabelAreaMode(null); if (next !== 'select') setSelectedAnnotationId(null); }} hasAnnotations={annotations.length > 0} onClearAll={clearAnnotations} selectedColor={selectedColor} onColorChange={changeColor} detailMode={detailMode} onDetailMode={() => { setDetailMode(!detailMode); setLabelAreaMode(null); setTool('select'); setSelectedAnnotationId(null); }} labelAreaMode={labelAreaMode} onLabelAreaMode={(mode) => { setLabelAreaMode((current) => current === mode ? null : mode); setDetailMode(false); setAddPointMode(false); setTool('select'); setSelectedAnnotationId(null); }} addPointMode={addPointMode} onAddPointMode={() => { setAddPointMode(!addPointMode); setDetailMode(false); setLabelAreaMode(null); setTool('select'); setSelectedAnnotationId(null); setSampleError(null); }} zeroEditActive={zeroPanel} zeroEditDisabled={!editableZeroLineCount(result)} onZeroEdit={toggleZeroEditor} keyPointsOnly={keyPointsOnly && hasKeySelection} keyPointsDisabled={!hasKeySelection} onKeyPointsOnlyChange={() => onKeyPointsOnlyChange(!keyPointsOnly)} />
      {zeroPanel && <div className="zero-edit zero-edit--compact"><div className="zero-edit__head"><div><b>제로라인 직접 편집</b><span>점을 끌어 이동 · 구간을 더블클릭해 직선/스플라인 전환</span></div><div className="zero-edit__tools"><button type="button" className={zeroPointAddMode ? 'zero-edit__mode is-active' : 'zero-edit__mode'} onClick={() => setZeroPointAddMode((current) => { const next = !current; if (next) setZeroPointDeleteMode(false); return next; })}>{zeroPointAddMode ? '점 추가 종료' : '점 추가'}</button><button type="button" className={zeroPointDeleteMode ? 'zero-edit__mode is-active' : 'zero-edit__mode'} onClick={() => setZeroPointDeleteMode((current) => { const next = !current; if (next) setZeroPointAddMode(false); return next; })}>{zeroPointDeleteMode ? '점 삭제 종료' : '점 삭제'}</button><button type="button" className="zero-edit__apply" onClick={() => onZeroEditsChange(draftZeroEdits)}>3D에 적용</button><button type="button" onClick={() => { setDraftZeroEdits([]); onZeroEditsChange([]); }}>초기화</button></div></div>
        <div className="zero-edit__status"><span>{zeroPointDeleteMode ? '삭제할 꼭짓점을 클릭하세요. 열린 선은 2점, 닫힌 선은 3점을 유지합니다.' : zeroPointAddMode ? '분할할 구간을 한 번 클릭하세요.' : '곡선으로 만들 구간만 더블클릭하세요. 인접 구간은 그대로 유지됩니다.'}</span>{JSON.stringify(draftZeroEdits) !== JSON.stringify(zeroEdits) && <em>3D 미적용 변경 있음</em>}</div>
      </div>}
      <div className="sheet-page" ref={sheetRef}><SheetTitleBlock values={sheetTitle} onChange={onSheetTitleChange} fonts={sheetTitleFonts} onFontChange={onSheetTitleFontChange} fontSizes={sheetTitleFontSizes} onFontSizeChange={onSheetTitleFontSizeChange} /><div className="sheet-stage sheet-stage--light" ref={stageRef}><SheetCanvas key={`${scan.id}-${onProduct ? 'product' : 'scan'}-${sheetTransformKey(activeSheetTransform)}`} scan={scan} imageUrl={baseImage} frameWidth={sheetFrameWidth} frameHeight={sheetFrameHeight} initialRegions={detailRegions} initialLayouts={sheetLayouts} initialLabelPositionsByLayout={labelPositionsByLayout} frontRotationSeed={{ transform: activeSheetTransform, canonicalOffsets: storedFrontLabelOffsets }} onRegionsChange={setDetailRegions} onLayoutsChange={setSheetLayouts} onLabelPositionsChange={handleLabelPositionsChange} onLayerSizeChange={handleLayerSizeChange} points={sheetPoints} coefficient={coefficient} showPoints={showPoints} visiblePointIds={visiblePointIds} onPointToggle={onPointToggle} pointOverrides={pointOverrides} onOverrideChange={handleOverrideChange} labelFontFamily={pointLabelFont} annotations={annotations} showAnnotations={showAnnotations} annotationTool={tool} setAnnotationTool={setTool} selectedAnnotationId={selectedAnnotationId} setSelectedAnnotationId={setSelectedAnnotationId} onAnnotationCommit={commitAnnotation} onAnnotationCreate={createAnnotation} onAnnotationDelete={deleteAnnotation} detailMode={detailMode} setDetailMode={setDetailMode} labelAreaMode={labelAreaMode} setLabelAreaMode={setLabelAreaMode} addPointMode={addPointMode} onAddPointAt={addPointAt} sampling={sampling} sampleError={sampleError} addedPoints={sheetAddedPoints} onRemoveAddedPoint={removeAddedPoint} zeroLines={sheetZeroLines} zeroSplineSegments={zeroLineSplineSegments} showZero={showZero} zeroEditable={zeroPanel} zeroPointAddMode={zeroPointAddMode} zeroPointDeleteMode={zeroPointDeleteMode} onZeroPointMove={moveZeroPoint} onZeroSegmentDoubleClick={toggleZeroSplineSegment} onZeroPointAdd={addZeroPoint} onZeroPointDelete={deleteZeroPoint} /></div></div>
      <div className="sheet-note"><ShieldCheck size={17} /><span><b>상단 표의 모든 글자를 클릭해 수정할 수 있습니다. 레이아웃은 제목 막대와 선택 핸들로 이동·조절합니다.</b>{excelError && <><br /><b className="sheet-note__error">{excelError}</b></>}</span>
        <input ref={excelInputRef} type="file" accept=".xlsx" className="visually-hidden" onChange={(e) => { setExcelFile(e.target.files?.[0] || null); setExcelError(null); }} aria-label="이어붙일 기존 보정 시트 엑셀 파일" />
        <button type="button" className="sheet-print sheet-print--ghost" onClick={() => excelInputRef.current?.click()} title="기존 보정 시트 엑셀 파일을 골라두면 그 아래에 이어붙입니다"><UploadCloud size={14} /> {excelFile ? excelFile.name : '기존 엑셀 불러오기'}</button>
        {excelFile && <button type="button" className="sheet-print__clear" onClick={() => { setExcelFile(null); if (excelInputRef.current) excelInputRef.current.value = ''; }} aria-label="선택한 엑셀 파일 취소" title="선택 취소"><X size={12} /></button>}
        <button type="button" className="sheet-print" onClick={() => void saveSheetExcel()} disabled={excelSaving}><FileSpreadsheet size={14} /> {excelSaving ? '엑셀 저장 중…' : '보정 시트 엑셀 저장'}</button>
        <button type="button" className="sheet-print" onClick={savePdf}><Printer size={14} /> 보정 시트 PDF 저장</button>
      </div>
    </div><aside className="control-panel"><div className="card coefficient-card"><div className="card-title"><div><h3>보정 계수</h3></div><span>{coefficient.toFixed(2)}×</span></div><div className="coefficient-input"><input aria-label="보정 계수 직접 입력" type="number" min="0.5" max="1.5" step="0.01" value={coefficient} onChange={(e) => { const value = e.target.valueAsNumber; if (!Number.isNaN(value)) onCoefficientChange(Math.max(0.5, Math.min(1.5, value))); }} /><span>×</span></div><input aria-label="보정 계수" type="range" min="0.5" max="1.5" step="0.05" value={coefficient} onChange={(e) => onCoefficientChange(Number(e.target.value))} /><div className="range-labels"><span>보수적 0.50</span><span>기준 1.00</span><span>적극적 1.50</span></div><div className="formula"><span>보정치</span><b>= 편차 × {coefficient.toFixed(2)} × (−1)</b></div>{overrideCount > 0 && <p className="coefficient-note">수정된 {overrideCount}개 포인트는 계수 영향을 받지 않습니다.</p>}</div><div className="card correction-summary"><h3>실제 엔진 요약</h3><div><span>보정 포인트</span><b>{visiblePointIds.size}개</b></div>{overrideCount > 0 && <div><span>수정된 포인트</span><b className="blue">{overrideCount}개</b></div>}<div><span>최대 보정량</span><b className="orange">{maxCorrection.toFixed(3)} mm</b></div><div><span>제로라인</span><b className="green">{result.stats.zeroRegions}개 영역</b></div><div><span>처리 품번</span><b>{scan.partNo}</b></div><div><span>작업자</span><input type="text" className="worker-input" value={worker} onChange={(e) => onWorkerChange(e.target.value)} placeholder="이름 입력" aria-label="작업자 이름" /></div><div><span>보정치 글꼴</span><select className="worker-input" value={pointLabelFont} onChange={(e) => setPointLabelFont(e.target.value)} aria-label="보정치 수치 글꼴 선택">{FONT_FAMILY_OPTIONS.map((option) => <option key={option.label} value={option.value} style={{ fontFamily: option.value || undefined }}>{option.label}</option>)}</select></div>{overrideCount > 0 && <button type="button" className="reset-all-overrides" onClick={() => void handleClearAllOverrides()}>모든 수정 취소</button>}</div><CorrectionHistoryPanel partNo={scan.partNo} entries={history} loading={historyLoading} pendingPointIds={pendingPointIds} deletingEntryIds={deletingEntryIds} error={historyError} onReload={loadHistory} onRestore={restoreHistoryEntry} onDelete={(entry) => void deleteHistoryEntry(entry)} /></aside></div>
  </section>;
}

function CadWorkspace({ active, scans, coefficientByScan, hiddenPointIdsByScan, pointOverridesByScan, onOverrideChange, zeroEditsByScan, notesByCad, setNotesByCad, regionsByCad, setRegionsByCad, zonesByPart, setZonesByPart }: {
  active: boolean;
  scans: ScanItem[];
  coefficientByScan: Record<string, number>;
  hiddenPointIdsByScan: Record<string, Set<string>>;
  pointOverridesByScan: Record<string, Record<string, number>>;
  onOverrideChange: (scanId: string, pointId: string, value: number | null) => void;
  zeroEditsByScan: Record<string, ZeroEdit[]>;
  notesByCad: Record<string, CadNote[]>;
  setNotesByCad: React.Dispatch<React.SetStateAction<Record<string, CadNote[]>>>;
  regionsByCad: Record<string, CadRegion[]>;
  setRegionsByCad: React.Dispatch<React.SetStateAction<Record<string, CadRegion[]>>>;
  zonesByPart: Record<string, CadRegion[]>;
  setZonesByPart: React.Dispatch<React.SetStateAction<Record<string, CadRegion[]>>>;
}) {
  type OpenCad = { key: string; mesh: CadMesh; scanId?: string };
  const [opened, setOpened] = useState<OpenCad[]>([]);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const fileStore = useRef<Record<string, File>>({});
  const [loadingCount, setLoadingCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [overlayScanId, setOverlayScanId] = useState('');
  const [overlay, setOverlay] = useState<CadOverlay | null>(null);
  const [overlayBusy, setOverlayBusy] = useState(false);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [shapeMatchWarning, setShapeMatchWarning] = useState<string | null>(null);
  const overlayCache = useRef<Record<string, CadOverlay>>({});
  /* 시트에 담아둔 3D 화면들. 어느 시점에서 찍었는지 함께 들고 있는다 —
     엑셀 쪽마다 그 이름을 적어 두지 않으면 나중에 보는 사람이 방향을
     못 가린다. 고유 열쇠는 앞 장을 지울 때 뒤 장이 다시 그려지지 않게 한다. */
  const [shots, setShots] = useState<
    { id: string; url: string; label: string }[]>([]);
  const [sheetBusy, setSheetBusy] = useState(false);
  const [adjustByCad, setAdjustByCad] = useState<Record<string, FitAdjust>>({});
  const [showAlign, setShowAlign] = useState(false);
  const importedReferenceFiles = useRef<Set<string>>(new Set());

  const requestCadMesh = useCallback(async (file: File, registeredPartNumber?: string, analysisId?: string) => {
    const form = new FormData();
    if (registeredPartNumber) {
      form.append('source', 'registered');
      form.append('partNumber', registeredPartNumber);
      if (analysisId) form.append('analysisId', analysisId);
    } else {
      form.append('file', file, file.name);
    }
    let response = await fetch(`${API_BASE}/api/cad`, { method: 'POST', body: form });
    /* 등록 직후의 서버 재시작이나 이전 작업 파일처럼 라이브러리에 원본이
       남아 있지 않은 경우에만 기존 직접 업로드 경로로 안전하게 되돌아간다. */
    if (response.status === 404 && registeredPartNumber) {
      const fallback = new FormData();
      fallback.append('file', file, file.name);
      response = await fetch(`${API_BASE}/api/cad`, { method: 'POST', body: fallback });
    }
    return response;
  }, []);

  const uploadCad = async (file: File, scanId?: string, registeredPartNumber?: string, analysisId?: string) => {
    setLoadingCount((count) => count + 1);
    setError(null);
    try {
      const response = await requestCadMesh(file, registeredPartNumber, analysisId);
      const data = await response.json() as CadMesh & { error?: string };
      if (!response.ok) throw new Error(data.error || `${file.name} 파일을 읽지 못했습니다.`);
      const key = data.cadId || `${file.name}-${Date.now()}-${Math.random()}`;
      fileStore.current[data.summary.name] = file;
      setOpened((current) => [...current.filter((item) => item.mesh.summary.name !== data.summary.name), { key, mesh: data, scanId }]);
      setActiveKey(key);
    } catch (err) {
      setError(String((err as Error).message || err));
    } finally {
      setLoadingCount((count) => Math.max(0, count - 1));
    }
  };

  const uploadMany = async (files: FileList | File[]) => {
    for (const file of Array.from(files)) await uploadCad(file);
  };

  useEffect(() => {
    let cancelled = false;
    const importRegisteredCads = async () => {
      for (const scan of scans) {
        for (const file of scan.cadFiles || []) {
          if (cancelled) return;
          const analysisId = scan.result?.analysisId;
          const identity = `registered-step-v3:${scan.id}:${analysisId || 'pending'}:${file.name}:${file.size}:${file.lastModified}`;
          if (importedReferenceFiles.current.has(identity)) continue;
          importedReferenceFiles.current.add(identity);
          const registeredPartNumber = partNoFromName(scan.partNo ?? '')
            || partNoFromName(file.name) || undefined;
          await uploadCad(file, scan.id, registeredPartNumber,
                          analysisId ?? undefined);
        }
      }
    };
    void importRegisteredCads();
    return () => { cancelled = true; };
  }, [scans]);

  const removeCad = (key: string) => {
    setOpened((current) => {
      const next = current.filter((item) => item.key !== key);
      if (activeKey === key) setActiveKey(next[0]?.key || null);
      return next;
    });
  };

  const selected = opened.find((item) => item.key === activeKey) || opened[0] || null;
  // 브랜치 원본과 같이 cadId가 아니라 파일명으로 저장한다. cadId는 파일을
  // 다시 열 때마다 달라지지만 파일명은 세션 복원 뒤에도 유지된다.
  const cadStateKey = selected?.mesh.summary.name || '';
  const zoneKey = partOfCad(selected?.mesh);
  const standardZones = zonesByPart[zoneKey] || [];
  const adjust = adjustByCad[cadStateKey] || NO_ADJUST;
  const analysed = useMemo(() => scans.filter((scan) => scan.result?.analysisId), [scans]);
  const overlayScan = analysed.find((scan) => scan.id === overlayScanId);
  const sheetValues = useMemo(() => {
    if (!overlayScan?.result) return null;
    const coefficient = coefficientByScan[overlayScan.id] ?? 1;
    const overrides = pointOverridesByScan[overlayScan.id] || {};
    const hidden = hiddenPointIdsByScan[overlayScan.id] || new Set<string>();
    const values: Record<string, number> = {};
    for (const point of overlayScan.result.points) {
      if (hidden.has(point.id)) continue;
      values[point.id] = overrides[point.id] !== undefined
        ? overrides[point.id]
        : -(point.value * coefficient);
    }
    return values;
  }, [overlayScan, coefficientByScan, pointOverridesByScan, hiddenPointIdsByScan]);

  const reopenCad = useCallback(async (name: string): Promise<CadMesh | null> => {
    const file = fileStore.current[name];
    if (!file) return null;
    const linkedScanId = opened.find((item) => item.mesh.summary.name === name)?.scanId;
    const linkedScan = scans.find((item) => item.id === linkedScanId);
    const registeredPartNumber = linkedScan
      ? partNoFromName(linkedScan.partNo ?? '')
        || partNoFromName(file.name) || undefined
      : undefined;
    const response = await requestCadMesh(
      file, registeredPartNumber, linkedScan?.result?.analysisId ?? undefined);
    const data = await response.json() as CadMesh & { error?: string };
    if (!response.ok) return null;
    const key = data.cadId || name;
    setOpened((current) => [...current.filter((item) => item.mesh.summary.name !== name), { key, mesh: data, scanId: linkedScanId }]);
    setActiveKey(key);
    return data;
  }, [opened, requestCadMesh, scans]);

  const requestOverlay = useCallback(async (scanId: string, cad = selected?.mesh, moved?: FitAdjust, retried = false): Promise<void> => {
    setOverlayScanId(scanId);
    setOverlayError(null);
    setShapeMatchWarning(null);
    const scan = scans.find((item) => item.id === scanId);
    const cadId = cad?.cadId;
    const analysisId = scan?.result?.analysisId;
    if (!cadId || !analysisId) {
      setOverlay(null);
      if (scanId) setOverlayError('CAD 또는 스캔 분석 결과를 찾을 수 없습니다.');
      return;
    }
    const placed = moved || adjustByCad[cad.summary.name] || NO_ADJUST;
    const sameAsAuto = placed.angle === 0 && placed.dx === 0 && placed.dy === 0 && placed.scale === 1;
    const appliedZeroEdits = zeroEditsByScan[scanId] || [];
    // surface-v2 invalidates browser-memory overlays produced while the
    // optional trimesh ray backend silently returned a 0% hit rate.
    const cacheKey = `surface-v2:${cadId}:${analysisId}${sameAsAuto ? '' : `:${JSON.stringify(placed)}`}${appliedZeroEdits.length ? `:${JSON.stringify(appliedZeroEdits)}` : ''}`;
    const cached = overlayCache.current[cacheKey];
    if (cached) {
      setOverlay(cached);
      setShapeMatchWarning(cached.fit.reliable === false || (cached.fit.hit_rate ?? 1) < 0.5
        ? `등록한 CAD와 스캔 형상의 정합률이 낮습니다 (${Math.round((cached.fit.hit_rate ?? 0) * 100)}%). 파일을 확인하거나 수동 정합을 사용하세요.`
        : null);
      return;
    }
    setOverlayBusy(true);
    setOverlay(null);
    try {
      const response = await fetch(`${API_BASE}/api/cad-overlay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cadId, analysisId, fitAdjust: sameAsAuto ? undefined : placed, zeroEdits: appliedZeroEdits.length ? appliedZeroEdits : undefined }),
      });
      const data = await response.json() as CadOverlay & { error?: string };
      if (!response.ok) throw new Error(data.error || '3D 표시 결과를 만들지 못했습니다.');
      overlayCache.current[cacheKey] = data;
      setOverlay(data);
      setShapeMatchWarning(data.fit.reliable === false || (data.fit.hit_rate ?? 1) < 0.5
        ? `등록한 CAD와 스캔 형상의 정합률이 낮습니다 (${Math.round((data.fit.hit_rate ?? 0) * 100)}%). 파일을 확인하거나 수동 정합을 사용하세요.`
        : null);
    } catch (err) {
      const message = String((err as Error).message || err);
      if (!retried && message.includes('만료')) {
        const fresh = await reopenCad(cad.summary.name);
        if (fresh) {
          await requestOverlay(scanId, fresh, placed, true);
          return;
        }
      }
      setOverlayError(message);
    } finally {
      setOverlayBusy(false);
    }
  }, [scans, selected, adjustByCad, reopenCad, zeroEditsByScan]);

  const appliedZeroEdits = overlayScanId ? JSON.stringify(zeroEditsByScan[overlayScanId] || []) : '';
  useEffect(() => {
    if (!overlayScanId || !selected) return;
    void requestOverlay(overlayScanId, selected.mesh);
    // requestOverlay changes when CAD state changes; this effect is specifically for applied zero-line edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appliedZeroEdits]);

  const nudge = (patch: Partial<FitAdjust>) => {
    if (!selected || !cadStateKey || !overlayScanId) return;
    const moved = { ...adjust, ...patch };
    setAdjustByCad((current) => ({ ...current, [cadStateKey]: moved }));
    void requestOverlay(overlayScanId, selected.mesh, moved);
  };

  const saveCadTable = () => {
    if (!overlay || !sheetValues) return;
    const rows = ['ID,X(mm),Y(mm),Z(mm),보정치(mm),공정'];
    for (const point of overlay.points) {
      const value = sheetValues[point.id];
      if (value === undefined) continue;
      const cad = point.cad || point.position;
      rows.push([point.id, ...cad.map((item) => item.toFixed(3)), value.toFixed(3), value >= 0 ? '용접' : 'CNC 가공'].join(','));
    }
    const blob = new Blob(['\ufeff' + rows.join('\r\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = `${selected?.mesh.summary.name || 'part'}_CAD_보정표.csv`; link.click();
    URL.revokeObjectURL(url);
  };

  const makeCadSheet = async () => {
    if (!overlayScan?.result?.analysisId || !sheetValues) return;
    setSheetBusy(true); setOverlayError(null);
    try {
      const naming = overlayScan.result.naming;
      const response = await fetch(`${API_BASE}/api/sheet-excel`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysisId: overlayScan.result.analysisId,
          corrections: sheetValues,
          images: shots.map((shot) => shot.url),
          imageLabels: shots.map((shot) => shot.label),
          filename: `${overlayScan.partNo || 'ADC'}_보정시트_3D`,
          meta: {
            partNo: naming?.part_no || overlayScan.partNo,
            partName: naming?.part_name || '', process: naming?.process || '',
            controlNo: naming?.control_no || '', appliedAt: naming?.applied_at || '',
            coefficient: coefficientByScan[overlayScan.id] ?? 1,
          },
        }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(data.error || '3D 보정시트를 만들지 못했습니다.');
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a');
      link.href = url; link.download = `${overlayScan.partNo || 'ADC'}_보정시트_3D.xlsx`; link.click();
      URL.revokeObjectURL(url);
    } catch (err) { setOverlayError(String((err as Error).message || err)); }
    finally { setSheetBusy(false); }
  };

  useEffect(() => {
    if (!selected) return;
    setOverlay(null);
    setOverlayError(null);
    if (selected.mesh.summary.source_format === 'morph') {
      setOverlayScanId('');
      return;
    }
    const linked = selected.scanId && analysed.find((scan) => scan.id === selected.scanId);
    const chosen = analysed.find((scan) => scan.id === overlayScanId);
    if (linked) void requestOverlay(linked.id, selected.mesh);
    else if (chosen) void requestOverlay(chosen.id, selected.mesh);
    else if (analysed.length === 1) void requestOverlay(analysed[0].id, selected.mesh);
  }, [selected?.key]); // CAD 탭을 바꾸면 해당 CAD 좌표로 다시 투영한다.

  return <section className="page page--workspace">
    <div className="page-heading"><div><h2>3D CAD 뷰어</h2></div></div>
    <div className="card upload-panel">
      <label className="dropzone">
        <input type="file" multiple accept=".catpart,.step,.stp,.stl,.ply,.obj,.off,.glb,.gltf,.3mf" onChange={(event: ChangeEvent<HTMLInputElement>) => {
          if (event.target.files?.length) void uploadMany(event.target.files);
          event.currentTarget.value = '';
        }} />
        <span className="dropzone__icon"><Layers3 size={29} /></span>
        <b>{loadingCount ? `CAD ${loadingCount}개 읽는 중…` : 'CAD 파일을 선택하세요'}</b>
        <span>CATPart · STEP · STL · PLY · OBJ · GLB · 3MF · 여러 파일 동시 선택 가능</span>
      </label>
      {error && <p className="sheet-note__error">{error}</p>}
    </div>
    {opened.length > 0 && <div className="cad-file-tabs" role="tablist" aria-label="열린 CAD 파일">
      {opened.map((item) => <div key={item.key} className={`cad-file-tab ${selected?.key === item.key ? 'active' : ''}`}>
        <button type="button" role="tab" aria-selected={selected?.key === item.key} onClick={() => setActiveKey(item.key)}>{item.mesh.summary.name}</button>
        <button type="button" aria-label={`${item.mesh.summary.name} 닫기`} onClick={() => removeCad(item.key)}><X size={12} /></button>
      </div>)}
    </div>}
    {selected && <>
      {selected.mesh.note && <div className="cad-overlay-bar">
        <span className="cad-overlay-bar__note">{selected.mesh.note}</span>
        <span className="count-chip">{selected.mesh.summary.source_format.toUpperCase()}</span>
      </div>}
      <div className="cad-overlay-bar">
        <label htmlFor="cad-overlay-scan">스캔 결과 표시</label>
        <select id="cad-overlay-scan" value={overlayScanId} onChange={(event) => {
          const scanId = event.target.value;
          if (scanId) void requestOverlay(scanId, selected.mesh);
          else { setOverlayScanId(''); setOverlay(null); setOverlayError(null); }
        }}>
          <option value="">선택 안 함</option>
          {analysed.map((scan) => <option key={scan.id} value={scan.id}>{scan.partNo || scan.name}</option>)}
        </select>
        {overlayBusy && <span className="cad-overlay-bar__note">제로라인·보정치를 3D 표면에 올리는 중…</span>}
        {overlay && <span className="cad-overlay-bar__note">
          제로라인 {overlay.zeroLines.length + (overlay.zeroAreas?.length || 0)}개 · 보정 포인트 {Object.keys(sheetValues || {}).length}개 ·{' '}
          {/* 검사 원본에서 온 포인트는 맞춘 것이 아니다 — 얹힘 대신
              표면까지 실제 거리를 적는다. 지어낸 지표를 보이지 않는다. */}
          {overlay.source === 'workspace'
            ? (overlay.surfaceGap
                ? `표면까지 ${overlay.surfaceGap.median.toFixed(2)}mm · 정합 없음`
                : '검사 원본 좌표 · 정합 없음')
            : `형상 얹힘 ${Math.round((overlay.fit.hit_rate || 0) * 100)}%`}
        </span>}
        {overlay && <button type="button" className="tool-button" onClick={() => setShowAlign((current) => !current)}>정렬 맞추기</button>}
        {overlay && sheetValues && <button type="button" className="tool-button" onClick={() => void makeCadSheet()} disabled={sheetBusy}>{sheetBusy ? '시트 만드는 중…' : `보정시트 만들기${shots.length ? ` (${shots.length}장)` : ''}`}</button>}
        {shots.length > 0 && <button type="button" className="tool-button"
          onClick={() => setShots([])}>담은 화면 비우기</button>}
        {overlay && sheetValues && <button type="button" className="tool-button" onClick={saveCadTable}>CAD 보정표</button>}
        {overlayError && <span className="cad-overlay-bar__err">{overlayError}</span>}
        {shapeMatchWarning && <span className="cad-overlay-bar__err">{shapeMatchWarning}</span>}
        {!analysed.length && <span className="cad-overlay-bar__err">먼저 엔진 결과에서 스캔 분석을 완료하세요.</span>}
      </div>
      {showAlign && overlay && <div className="cad-align-panel">
        <b>수동 정합</b>
        <label>회전 <input type="number" step="0.25" value={adjust.angle} onChange={(event) => nudge({ angle: Number(event.target.value) })} />°</label>
        <label>X 이동 <input type="number" step="1" value={adjust.dx} onChange={(event) => nudge({ dx: Number(event.target.value) })} />px</label>
        <label>Y 이동 <input type="number" step="1" value={adjust.dy} onChange={(event) => nudge({ dy: Number(event.target.value) })} />px</label>
        <label>배율 <input type="number" min="0.5" max="1.5" step="0.005" value={adjust.scale} onChange={(event) => nudge({ scale: Number(event.target.value) })} /></label>
        <button type="button" className="tool-button" onClick={() => nudge(NO_ADJUST)}>자동값 복원</button>
      </div>}
      {/* 담은 화면을 눈으로 확인하고 한 장씩 뺀다. 숫자만 보이면 잘못
          담았을 때 전부 비우고 처음부터 다시 찍는 수밖에 없다.
          순서가 곧 엑셀의 쪽 순서다(1쪽은 늘 스캔 전체도). */}
      {shots.length > 0 && <div className="shot-strip">
        <span className="shot-strip__head">
          시트에 담은 화면 {shots.length}장 · 1쪽은 스캔 전체도이고 아래
          순서대로 2쪽부터 붙습니다
        </span>
        <div className="shot-strip__row">
          {shots.map((shot, order) => (
            <figure key={shot.id} className="shot-card">
              <img src={shot.url} alt={`${shot.label} 화면`} />
              <figcaption>{order + 2}쪽 · {shot.label}</figcaption>
              <button type="button" aria-label={`${shot.label} 화면 빼기`}
                title="이 화면만 뺍니다"
                onClick={() => setShots((current) =>
                  current.filter((_, index) => index !== order))}>×</button>
            </figure>
          ))}
        </div>
      </div>}
      <div className="card cad-viewer" style={{ height: 760 }}>
        <CadViewer
          active={active}
          mesh={selected.mesh}
          showHoles
          overlay={overlay}
          sheetValues={sheetValues}
          onCapture={(url, label) => setShots((current) => [...current,
            // 지울 때 뒤 장들의 열쇠가 바뀌지 않게 고유한 값을 준다.
            { id: `S-${Date.now().toString(36)}-${current.length}`, url, label }])}
          onCorrectionChange={overlayScanId ? (pointId, value) => onOverrideChange(overlayScanId, pointId, value) : undefined}
          notes={notesByCad[cadStateKey] || []}
          onNotesChange={(notes) => cadStateKey && setNotesByCad((current) => ({ ...current, [cadStateKey]: notes }))}
          regions={[...standardZones, ...(regionsByCad[cadStateKey] || [])]}
          onRegionsChange={(regions) => cadStateKey && setRegionsByCad((current) => ({ ...current, [cadStateKey]: regions.filter((region) => !region.standard) }))}
        />
      </div>
      {zoneKey && <div className="cad-zone-book card">
        <div className="card-title"><div><h3>표준 공정 구역</h3></div>
          <button type="button" className="tool-button" onClick={() => setZonesByPart((current) => ({ ...current, [zoneKey]: [...(current[zoneKey] || []), { id: `S-${Date.now().toString(36)}`, standard: true, die: '하형', work: '용접', note: '', box: { min: [0, 0, 0], max: [100, 100, 100] } }] }))}>구역 추가</button>
        </div>
        {standardZones.map((zone, index) => <div className="cad-zone-row" key={zone.id}>
          <b>{CIRCLED[index] || index + 1}</b>
          <select value={zone.die} onChange={(event) => setZonesByPart((current) => ({ ...current, [zoneKey]: (current[zoneKey] || []).map((item) => item.id === zone.id ? { ...item, die: event.target.value as CadRegion['die'] } : item) }))}>{DIE_CHOICES.map((value) => <option key={value}>{value}</option>)}</select>
          <select value={zone.work} onChange={(event) => setZonesByPart((current) => ({ ...current, [zoneKey]: (current[zoneKey] || []).map((item) => item.id === zone.id ? { ...item, work: event.target.value as CadRegion['work'] } : item) }))}>{WORK_CHOICES.map((value) => <option key={value}>{value}</option>)}</select>
          <span>시작 {zone.box?.min.join(', ')}</span><span>끝 {zone.box?.max.join(', ')}</span>
          <input value={zone.note || ''} placeholder="메모" onChange={(event) => setZonesByPart((current) => ({ ...current, [zoneKey]: (current[zoneKey] || []).map((item) => item.id === zone.id ? { ...item, note: event.target.value } : item) }))} />
          <button type="button" onClick={() => setZonesByPart((current) => ({ ...current, [zoneKey]: (current[zoneKey] || []).filter((item) => item.id !== zone.id) }))}><X size={13} /></button>
        </div>)}
      </div>}
    </>}
  </section>;
}

export default function Home() {
  const [view, setView] = useState<View>('overview'); const [scans, setScans] = useState<ScanItem[]>([]); const [activeId, setActiveId] = useState<string>(); const [backendOnline, setBackendOnline] = useState<boolean | null>(null); const [hiddenPointIdsByScan, setHiddenPointIdsByScan] = useState<Record<string, Set<string>>>({}); const [pointOverridesByScan, setPointOverridesByScan] = useState<Record<string, Record<string, number>>>({}); const [coefficientByScan, setCoefficientByScan] = useState<Record<string, number>>({}); const [annotationsByScan, setAnnotationsByScan] = useState<Record<string, Annotation[]>>({}); const [sheetTitlesByScan, setSheetTitlesByScan] = useState<Record<string, SheetTitleValues>>({});
  const [keyPointsOnlyByScan, setKeyPointsOnlyByScan] = useState<Record<string, boolean>>({});
  const [resultEngine, setResultEngine] = useState<Engine>('label');
  const [sheetTitleFontsByScan, setSheetTitleFontsByScan] = useState<Record<string, SheetTitleFonts>>({});
  const [sheetTitleFontSizesByScan, setSheetTitleFontSizesByScan] = useState<Record<string, SheetTitleFontSizes>>({});
  const [notesByCad, setNotesByCad] = useState<Record<string, CadNote[]>>({});
  const [regionsByCad, setRegionsByCad] = useState<Record<string, CadRegion[]>>({});
  const [zonesByPart, setZonesByPart] = useState<Record<string, CadRegion[]>>({});
  const [zeroEditsByScan, setZeroEditsByScan] = useState<Record<string, ZeroEdit[]>>({});
  /* 보정시트 탭에서 회전/라벨위치/창 크기 조절은 전부 ServicePreview 안의
     로컬 state 였다 -- WORKSPACE 메뉴를 "엔진 결과" 등 다른 탭으로 옮기면
     view !== 'service' 라 ServicePreview 가 통째로 언마운트되고, 그 안의
     state 는 전부 사라졌다가 되돌아오면 초기화된 채로 새로 마운트됐다.
     다른 byScan state(hiddenPointIdsByScan 등)와 같은 자리(Home)로 끌어
     올려 탭을 옮겨도 유지되게 한다. scan.id 로 갈라 두는 건 그대로라
     다른 파트에는 영향이 없다. */
  const [sheetTransformByScan, setSheetTransformByScan] = useState<Record<string, SheetImageTransform>>({});
  const [sheetLayoutsByScan, setSheetLayoutsByScan] = useState<Record<string, SheetLayout[]>>({});
  const [detailRegionsByScan, setDetailRegionsByScan] = useState<Record<string, DetailRegion[]>>({});
  const [frontLabelPositionsByScan, setFrontLabelPositionsByScan] = useState<Record<string, Record<string, { x: number; y: number }>>>({});
  const [detailLabelPositionsByScan, setDetailLabelPositionsByScan] = useState<Record<string, Record<string, Record<string, { x: number; y: number }>>>>({});
  /* 작업자가 직접 찍은 포인트(addedPoints)도 안 올려두면, 탭을 옮겼다
     돌아왔을 때 점 자체가 사라지고 -- 그 점에 붙어 있던 라벨 위치도
     "지금 화면에 없는 점" 이라 라벨 위치 재계산 과정에서 같이 지워진다
     (라벨 위치 지속이 이 점에 얹혀 있는 셈이라 같이 끌어올려야 한다). */
  const [addedPointsByScan, setAddedPointsByScan] = useState<Record<string, PointResult[]>>({});
  const sessionRef = useRef<SessionSnapshot>(emptySession());
  const [sessionLoaded, setSessionLoaded] = useState(false);
  /* 화면이 새로 붙는 순간에는 물려 있는 분석 요청이 있을 수 없다.
   *
   * 분석 중에 화면이 다시 그려지면(개발 중 파일 수정, 새로고침, 작업
   * 불러오기) 돌던 반복문이 딸린 화면째 버려진다. 그런데 상태는 '분석 중'
   * 인 채로 남고, 그 상태에서는 분석 버튼이 잠긴다 — 서버는 놀고 있는데
   * 사람은 30분을 기다려도 아무 일이 안 일어나고 다시 누를 수도 없다.
   * 붙는 순간 '오류' 로 돌려 이유를 보이고 다시 누를 수 있게 한다. */
  useEffect(() => {
    setScans((current) => current.some((scan) => scan.status === 'analyzing')
      ? current.map((scan) => scan.status === 'analyzing'
        ? { ...scan, status: 'error' as const,
            error: '화면이 다시 그려지며 분석이 끊겼습니다 — 다시 눌러 주세요.' }
        : scan)
      : current);
  }, []);
  /* 작업자 이름은 보정 이력에 남기는 용도라 브라우저에 저장해 다음에도 다시 입력하지 않게 한다. */
  const [worker, setWorker] = useState(() => (typeof window === 'undefined' ? '' : window.localStorage.getItem('adc-worker-name') || ''));
  useEffect(() => { if (typeof window !== 'undefined') window.localStorage.setItem('adc-worker-name', worker); }, [worker]);
  useEffect(() => {
    const saved = loadSession() || emptySession();
    sessionRef.current = saved;
    const notes: Record<string, CadNote[]> = {};
    const regions: Record<string, CadRegion[]> = {};
    for (const [name, entry] of Object.entries(saved.byCad || {})) {
      if (entry.notes) notes[name] = entry.notes as CadNote[];
      if (entry.regions) regions[name] = entry.regions as CadRegion[];
    }
    setNotesByCad(notes);
    setRegionsByCad(regions);
    setSessionLoaded(true);
  }, []);
  useEffect(() => { fetch(`${API_BASE}/api/health`).then((response) => response.json() as Promise<HealthResponse>).then((data) => setBackendOnline(Boolean(data.ok))).catch(() => setBackendOnline(false)); }, []);
  const resolvedActiveId = activeId || scans[0]?.id;
  const activeScan = scans.find((scan) => scan.id === resolvedActiveId); const completedScan = activeScan?.result ? activeScan : scans.find((scan) => scan.result); const hasResult = Boolean(completedScan?.result);
  const hiddenPointIds = completedScan ? hiddenPointIdsByScan[completedScan.id] || new Set<string>() : new Set<string>();
  const pointOverrides = completedScan ? pointOverridesByScan[completedScan.id] || {} : {};
  const coefficient = completedScan ? coefficientByScan[completedScan.id] ?? 1 : 1;
  const keyPointsOnly = completedScan ? keyPointsOnlyByScan[completedScan.id] ?? false : false;
  const sheetTitle = completedScan ? sheetTitlesByScan[completedScan.id] || createDefaultSheetTitleValues(completedScan) : undefined;
  const sheetTitleFonts = completedScan ? sheetTitleFontsByScan[completedScan.id] || DEFAULT_TITLE_FONTS : DEFAULT_TITLE_FONTS;
  const sheetTitleFontSizes = completedScan ? sheetTitleFontSizesByScan[completedScan.id] || {} : {};
  const togglePoint = (id: string) => completedScan && setHiddenPointIdsByScan((current) => { const next = new Set(current[completedScan.id] || []); if (next.has(id)) next.delete(id); else next.add(id); return { ...current, [completedScan.id]: next }; });
  const setPointOverride = (id: string, value: number | null) => completedScan && setPointOverridesByScan((current) => { const next = { ...(current[completedScan.id] || {}) }; if (value === null) delete next[id]; else next[id] = value; return { ...current, [completedScan.id]: next }; });
  const setPointOverrideFor = (scanId: string, id: string, value: number | null) => setPointOverridesByScan((current) => { const next = { ...(current[scanId] || {}) }; if (value === null) delete next[id]; else next[id] = value; return { ...current, [scanId]: next }; });
  const setCoefficient = (value: number) => completedScan && setCoefficientByScan((current) => ({ ...current, [completedScan.id]: value }));
  const clearAllOverrides = () => completedScan && setPointOverridesByScan((current) => ({ ...current, [completedScan.id]: {} }));
  const annotations = completedScan ? annotationsByScan[completedScan.id] || [] : [];
  const setAnnotations = (updater: (current: Annotation[]) => Annotation[]) => completedScan && setAnnotationsByScan((current) => ({ ...current, [completedScan.id]: updater(current[completedScan.id] || []) }));
  const setSheetTitleField = (field: SheetTitleField, value: string) => {
    if (!completedScan) return;
    const targetScan = completedScan;
    setSheetTitlesByScan((current) => ({ ...current, [targetScan.id]: { ...(current[targetScan.id] || createDefaultSheetTitleValues(targetScan)), [field]: value } }));
  };
  const setSheetTitleFontField = (field: SheetTitleField, fontFamily: string) => {
    if (!completedScan) return;
    const targetScan = completedScan;
    setSheetTitleFontsByScan((current) => ({ ...current, [targetScan.id]: { ...(current[targetScan.id] || DEFAULT_TITLE_FONTS), [field]: fontFamily } }));
  };
  const setSheetTitleFontSizeField = (field: SheetTitleField, size: number) => {
    if (!completedScan) return;
    const targetScan = completedScan;
    setSheetTitleFontSizesByScan((current) => ({ ...current, [targetScan.id]: { ...(current[targetScan.id] || {}), [field]: size } }));
  };
  useEffect(() => {
    if (!sessionLoaded) return;
    for (const scan of scans) {
      const saved = sessionRef.current.byPart[scan.partNo];
      if (!saved) continue;
      if (saved.coefficient !== undefined && coefficientByScan[scan.id] === undefined) setCoefficientByScan((current) => ({ ...current, [scan.id]: saved.coefficient! }));
      if (saved.overrides && pointOverridesByScan[scan.id] === undefined) setPointOverridesByScan((current) => ({ ...current, [scan.id]: { ...saved.overrides } }));
      if (saved.hidden && hiddenPointIdsByScan[scan.id] === undefined) {
        const keyIds = new Set(scan.result?.keySelection?.ids || []);
        const legacy = scan.result?.keySelection
          ? scan.result.points.filter((point) => !keyIds.has(point.id)).map((point) => point.id).sort()
          : [];
        const stored = [...saved.hidden].sort();
        const wasLegacyKeyFilter = legacy.length === stored.length && legacy.every((id, index) => id === stored[index]);
        setHiddenPointIdsByScan((current) => ({ ...current, [scan.id]: new Set(wasLegacyKeyFilter ? [] : saved.hidden) }));
      }
      if (saved.head && sheetTitlesByScan[scan.id] === undefined) setSheetTitlesByScan((current) => ({ ...current, [scan.id]: saved.head as SheetTitleValues }));
      if (saved.zones && zonesByPart[scan.partNo] === undefined) setZonesByPart((current) => ({ ...current, [scan.partNo]: saved.zones as CadRegion[] }));
      if (saved.zeroEdits && zeroEditsByScan[scan.id] === undefined) setZeroEditsByScan((current) => ({ ...current, [scan.id]: saved.zeroEdits as ZeroEdit[] }));
    }
  }, [sessionLoaded, scans, coefficientByScan, pointOverridesByScan, hiddenPointIdsByScan, sheetTitlesByScan, zonesByPart, zeroEditsByScan]);
  /* 방향만 다시 계산한다. Qwen 판독은 그대로 두고 좌표만 옮겨 받는다. */
  const realign = async (flipX?: boolean, flipY?: boolean, rotation?: number) => {
    if (!completedScan?.result) return;
    const target = completedScan;
    const form = new FormData();
    form.append('file', target.file, target.name);
    if (target.productFile) form.append('product', target.productFile, target.productFile.name);
    /* 반전을 지정하지 않으면 서버가 자동 판정과 확정 저장분을 따른다. 단순 재계산이 그 경우다. */
    if (flipX !== undefined) form.append('flipX', String(flipX));
    if (flipY !== undefined) form.append('flipY', String(flipY));
    if (rotation !== undefined) form.append('rotation', String(rotation));
    form.append('points', JSON.stringify(target.result!.points.map((point) => ({ id: point.id, xPx: point.xPx, yPx: point.yPx }))));
    const response = await fetch(`${API_BASE}/api/realign`, { method: 'POST', body: form });
    const data = await response.json() as { alignment?: AlignmentInfo; alignmentOverlay?: string; productImage?: string; productSource?: string; points?: { id: string; xProduct: number; yProduct: number }[]; warnings?: string[]; error?: string };
    if (!response.ok || !data.alignment) throw new Error(data.error || '정렬을 다시 계산하지 못했습니다.');
    const moved = new Map((data.points || []).map((point) => [point.id, point]));
    setScans((current) => current.map((scan) => scan.id !== target.id || !scan.result ? scan : { ...scan, result: {
      ...scan.result,
      alignment: data.alignment!,
      alignmentOverlay: data.alignmentOverlay ?? scan.result.alignmentOverlay,
      productImage: data.productImage ?? scan.result.productImage,
      productSource: data.productSource ?? scan.result.productSource,
      points: scan.result.points.map((point) => { const next = moved.get(point.id); return next ? { ...point, xProduct: next.xProduct, yProduct: next.yProduct } : { ...point, xProduct: undefined, yProduct: undefined }; }),
      stats: { ...scan.result.stats, pointsTransferred: moved.size },
      warningsByEngine: { ...scan.result.warningsByEngine, product: data.warnings || [] },
    } }));
  };
  const confirmAlignment = async () => {
    const alignment = completedScan?.result?.alignment;
    const partNumber = completedScan?.result?.partNumber;
    if (!alignment || !partNumber) return;
    const response = await fetch(`${API_BASE}/api/alignment`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ partNumber, alignment }) });
    if (!response.ok) throw new Error('정렬을 저장하지 못했습니다.');
  };
  const openResults = (id: string) => { setActiveId(id); setResultEngine('label'); setView('results'); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const openEngine = (engine: Engine) => { if (!hasResult) return; setResultEngine(engine); setView('results'); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const selectView = (next: View) => {
    if ((next === 'service' || next === 'results') && !hasResult) return;
    setView(next);
    window.scrollTo({ top: 0, behavior: 'instant' });
  };
  const buildSessionSnapshot = (): SessionSnapshot => {
    const snapshot = emptySession();
    for (const scan of scans) {
      const overrides = pointOverridesByScan[scan.id];
      const hidden = hiddenPointIdsByScan[scan.id];
      const head = sheetTitlesByScan[scan.id];
      const coefficient = coefficientByScan[scan.id];
      const zones = zonesByPart[scan.partNo];
      const zeroEdits = zeroEditsByScan[scan.id];
      if (!overrides && !hidden && !head && coefficient === undefined && !zones?.length && !zeroEdits?.length) continue;
      snapshot.byPart[scan.partNo] = {
        coefficient,
        overrides: overrides ? { ...overrides } : undefined,
        hidden: hidden ? [...hidden] : undefined,
        head: head ? { ...head } : undefined,
        zones: zones?.length ? zones : undefined,
        zeroEdits: zeroEdits?.length ? zeroEdits : undefined,
      };
    }
    for (const name of new Set([...Object.keys(notesByCad), ...Object.keys(regionsByCad)])) {
      const notes = notesByCad[name] || [];
      const regions = regionsByCad[name] || [];
      if (!notes.length && !regions.length) continue;
      snapshot.byCad[name] = { notes: notes.length ? notes : undefined, regions: regions.length ? regions : undefined };
    }
    return snapshot;
  };
  useEffect(() => {
    if (!sessionLoaded) return;
    const snapshot = buildSessionSnapshot();
    sessionRef.current = snapshot;
    saveSession(snapshot);
  }, [sessionLoaded, scans, coefficientByScan, pointOverridesByScan, hiddenPointIdsByScan, sheetTitlesByScan, notesByCad, regionsByCad, zonesByPart, zeroEditsByScan]);
  const saveWorkFile = () => {
    const snapshot = buildSessionSnapshot();
    saveSession(snapshot);
    downloadSession(snapshot);
  };
  const loadWorkFile = async (file: File) => {
    const snapshot = await readSessionFile(file);
    if (!snapshot) return;
    sessionRef.current = snapshot;
    saveSession(snapshot);
    const notes: Record<string, CadNote[]> = {};
    const regions: Record<string, CadRegion[]> = {};
    for (const [name, entry] of Object.entries(snapshot.byCad || {})) {
      if (entry.notes) notes[name] = entry.notes as CadNote[];
      if (entry.regions) regions[name] = entry.regions as CadRegion[];
    }
    setNotesByCad(notes);
    setRegionsByCad(regions);
    for (const scan of scans) {
      const saved = snapshot.byPart[scan.partNo];
      if (!saved) continue;
      if (saved.overrides) setPointOverridesByScan((current) => ({ ...current, [scan.id]: { ...saved.overrides } }));
      if (saved.hidden) setHiddenPointIdsByScan((current) => ({ ...current, [scan.id]: new Set(saved.hidden) }));
      if (saved.head) setSheetTitlesByScan((current) => ({ ...current, [scan.id]: saved.head as SheetTitleValues }));
      if (saved.coefficient !== undefined) setCoefficientByScan((current) => ({ ...current, [scan.id]: saved.coefficient! }));
      if (saved.zones) setZonesByPart((current) => ({ ...current, [scan.partNo]: saved.zones as CadRegion[] }));
      if (saved.zeroEdits) setZeroEditsByScan((current) => ({ ...current, [scan.id]: saved.zeroEdits as ZeroEdit[] }));
    }
  };
  const resetWork = () => {
    clearSession();
    setPointOverridesByScan({});
    setHiddenPointIdsByScan({});
    setSheetTitlesByScan({});
    setCoefficientByScan({});
    setNotesByCad({});
    setRegionsByCad({});
    setZonesByPart({});
    setZeroEditsByScan({});
    sessionRef.current = emptySession();
  };
  return <main className={`studio-shell${view === 'overview' ? ' studio-shell--overview' : ''}`}>
    <WorkspaceNavigation view={view} onSelect={selectView} hasResult={hasResult} scans={scans} activeId={resolvedActiveId} onScanChange={setActiveId} />
    <div className="app-main">
      {view === 'overview' && <WorkspaceHub onSelect={selectView} hasResult={hasResult} scanCount={scans.length} backendOnline={backendOnline} />}
      {view === 'workspace' && <Workspace scans={scans} selectedScan={activeScan || scans[0]} setScans={setScans} result={completedScan?.result} onOpenResults={openResults} onOpenEngine={openEngine} backendOnline={backendOnline} />}
      {view === 'results' && completedScan?.result && <Results scan={completedScan} engine={resultEngine} setEngine={setResultEngine} onScanData={() => setView('workspace')} onService={() => setView('service')} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} keyPointsOnly={keyPointsOnly} onKeyPointsOnlyChange={(value) => setKeyPointsOnlyByScan((current) => ({ ...current, [completedScan.id]: value }))} onRealign={realign} onConfirmAlignment={confirmAlignment} />}
      {view === 'service' && completedScan?.result && sheetTitle && <ServicePreview scan={completedScan} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} keyPointsOnly={keyPointsOnly} onKeyPointsOnlyChange={(value) => setKeyPointsOnlyByScan((current) => ({ ...current, [completedScan.id]: value }))} pointOverrides={pointOverrides} onOverrideChange={setPointOverride} onClearAllOverrides={clearAllOverrides} annotations={annotations} setAnnotations={setAnnotations} sheetTitle={sheetTitle} onSheetTitleChange={setSheetTitleField} sheetTitleFonts={sheetTitleFonts} onSheetTitleFontChange={setSheetTitleFontField} sheetTitleFontSizes={sheetTitleFontSizes} onSheetTitleFontSizeChange={setSheetTitleFontSizeField} worker={worker} onWorkerChange={setWorker} coefficient={coefficient} onCoefficientChange={setCoefficient} zeroEdits={zeroEditsByScan[completedScan.id] || EMPTY_ZERO_EDITS} onZeroEditsChange={(edits) => setZeroEditsByScan((current) => ({ ...current, [completedScan.id]: edits }))} sheetTransformByScan={sheetTransformByScan} setSheetTransformByScan={setSheetTransformByScan} sheetLayoutsByScan={sheetLayoutsByScan} setSheetLayoutsByScan={setSheetLayoutsByScan} detailRegionsByScan={detailRegionsByScan} setDetailRegionsByScan={setDetailRegionsByScan} frontLabelPositionsByScan={frontLabelPositionsByScan} setFrontLabelPositionsByScan={setFrontLabelPositionsByScan} detailLabelPositionsByScan={detailLabelPositionsByScan} setDetailLabelPositionsByScan={setDetailLabelPositionsByScan} addedPointsByScan={addedPointsByScan} setAddedPointsByScan={setAddedPointsByScan} />}
      {view === 'files' && <FileOrganizerPage />}
      <div style={{ display: view === 'cad' ? 'block' : 'none' }}>
        <CadWorkspace active={view === 'cad'} scans={scans} coefficientByScan={coefficientByScan} hiddenPointIdsByScan={hiddenPointIdsByScan} pointOverridesByScan={pointOverridesByScan} onOverrideChange={setPointOverrideFor} zeroEditsByScan={zeroEditsByScan} notesByCad={notesByCad} setNotesByCad={setNotesByCad} regionsByCad={regionsByCad} setRegionsByCad={setRegionsByCad} zonesByPart={zonesByPart} setZonesByPart={setZonesByPart} />
      </div>
    </div>
  </main>;
}
