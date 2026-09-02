'use client';

/* Blob/data URLs are local engine outputs and cannot use Next's remote image loader. */
/* eslint-disable @next/next/no-img-element */

import {
  Activity, AlertTriangle, ArrowLeft, ArrowRight, ArrowUpRight, BarChart3, Check, CheckCircle2, ChevronDown, ChevronRight,
  Circle, CircleHelp, Copy, Crosshair, Database, Eye, EyeOff, File, FileSpreadsheet, Files, Folder, FolderOpen, Gauge, Grid2X2, HardDrive, Image as ImageIcon,
  Layers3, ListFilter, Maximize2, MousePointer2, Move, MoveRight, PanelLeftClose, Play, RefreshCw, Settings2,
  Printer, Server, ShieldCheck, Sparkles, Square, Trash2, Type, UploadCloud, X, ZoomIn, ZoomOut,
} from 'lucide-react';
import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { clearSession, downloadSession, emptySession, loadSession, readSessionFile, saveSession, type SessionSnapshot } from './session-store';
import { CIRCLED, DIE_CHOICES, WORK_CHOICES, CadViewer, type CadMesh, type CadMorph, type CadNote, type CadOverlay, type CadRegion, type CadSection } from './cad-viewer';

const API_BASE = 'http://127.0.0.1:8000';

type View = 'workspace' | 'results' | 'service' | 'files' | 'cad';
type Engine = 'label' | 'deviation' | 'zero';
type ScanStatus = 'ready' | 'analyzing' | 'done' | 'error';
/* source 가 'colormap' 이면 작업자가 찍은 추정 포인트다. 라벨을 읽어 얻은 실측값과
   섞이지 않도록 화면에서도 구분해 보여준다. */
/* xProduct/yProduct 는 같은 포인트를 제품데이터 이미지 기준 %로 다시 적은 값이다.
   정렬에 실패했거나 제품데이터가 없으면 비어 있다. */
/* keyReasons 는 보정시트에 기본으로 올릴 이유다: peak(국소 극값), sign_change(부호 반전), extreme(전체 최대·최소). */
type PointResult = { id: string; xPx: number; yPx: number; x: number; y: number; value: number; labelColor: string; confidence: string; source?: 'colormap'; xProduct?: number; yProduct?: number; keyReasons?: string[] };
type KeySelection = { ids: string[]; total: number; selected: number; peaks: number; signChanges: number; extremes: number };
/* 스캔을 제품데이터 위로 옮기는 변환. margin 은 1위와 2위 방향의 점수 차이고,
   대칭 부품은 이 값이 0에 가까워 사람이 방향을 정해 줘야 한다. */
type AlignmentInfo = { matrix: number[]; flipX: boolean; flipY: boolean; outlineIou: number; holeIou: number; bandIou: number; score: number; margin: number; confident: boolean; overridden: boolean; scanSize: number[]; productSize: number[]; candidates?: { flipX: boolean; flipY: boolean; score: number }[]; warnings: string[] };
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
  /* 보정시트에 적을 만한 포인트만 골라낸 것. 스캔에는 백여 개가 찍히지만
     현업 시트에 적히는 건 열몇 개다. */
  keyPoints?: { point_id: string; x_px: number; y_px: number; value: number; score: number; reason: string }[];
  /* 현업 파이프라인(lab_pipeline)이 만든 제로라인. 허용범위 밖 영역을
     윤곽 위 제로포인트 둘로 닫은 선이라 근거가 가장 분명하다. */
  labZeroLines?: [number, number][][];
  /* 제로 **영역**(67XX6). 서버가 이미 네모로 다듬어 보낸다 —
     시트와 3D 가 같은 도형을 그리려고 한 군데서 만든다. */
  labZeroAreas?: [number, number][][];
  labZeroRegions?: { label: string; area: number; status: string;
                     zeroPoints: string[]; attempts: number; coverage: number }[];
  keyPointsRejected?: { id: string; value: number }[];
  source: { name: string; width: number; height: number };
  partNumber: string | null;
  cleanImage: string | null;
  productImage: string | null;
  productSource: string | null;
  alignment: AlignmentInfo | null;
  alignmentOverlay: string | null;
  keySelection?: KeySelection;
  zeroOverlay: string | null;
  zeroMask: string | null;
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
type ScanItem = { id: string; name: string; partNo: string; size: string; url: string; file: File; status: ScanStatus; tone: number; result?: AnalysisResult; error?: string; productFile?: File; productUrl?: string };
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
type HealthResponse = { ok?: boolean; folderAvailable?: boolean };
type FileDatabaseStatus = { configured: boolean; label: string; connected: boolean | null; catalogCount: number; operationCount: number; version?: string; error?: string };
type FileOrganizerStatus = { sourceRoot: string; destinationRoot: string; sourceAvailable: boolean; destinationAvailable: boolean; database: FileDatabaseStatus };
type FileOrganizerItem = { id: string; name: string; sourcePath: string; sourceKind: 'source' | 'upload'; size: number; modified: string; customer: string; itemNo: string; family: string; productName: string; process: string; categoryKey: string; categoryLabel: string; confidence: number; reasons: string[]; targetDir: string; targetPath: string; matchedProductFolder: string; detailPath: string };
type FolderAxis = 'family' | 'category' | 'product';
type FolderAxisOption = { id: FolderAxis; label: string };
type FolderMigration = { moved: number; skipped: number; errors: string[] };
type FolderOrderResponse = { folderOrder: FolderAxis[]; axes: FolderAxisOption[]; migration?: FolderMigration; error?: string };
type OrganizerPathsResponse = { sourceRoot: string; destinationRoot: string; sourceLocked: boolean; destinationLocked: boolean };
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

const engineMeta: Record<Engine, { name: string; short: string; color: string }> = {
  label: { name: '라벨 제거 · 복원', short: 'label_removal', color: '#7058e8' },
  deviation: { name: '편차 포인트 추출', short: 'deviation_extraction', color: '#ee6b3c' },
  zero: { name: '제로라인 검출', short: 'zero_line_detection', color: '#17a58b' },
};

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

function Heatmap({ imageUrl, width, height, children, lightBackground = false }: { imageUrl?: string | null; width: number; height: number; children?: React.ReactNode; lightBackground?: boolean }) {
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
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
    {imageUrl ? <div className="heatmap__media" style={{ aspectRatio: `${width} / ${height}` }}>
      <div className="heatmap__transform" style={{ transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${scale})` }}>
        <img src={imageUrl} alt="엔진이 처리한 3D 스캔 편차 이미지" />
        {children}
      </div>
    </div> : <div className="heatmap__empty"><ImageIcon size={34} /><span>분석 결과 이미지가 없습니다.</span></div>}
  </div>;
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

function AnnotationToolbar({ tool, setTool, hasAnnotations, onClearAll, selectedColor, onColorChange, detailMode, onDetailMode, labelAreaMode, onLabelAreaMode, addPointMode, onAddPointMode }: { tool: AnnotationTool; setTool: (tool: AnnotationTool) => void; hasAnnotations: boolean; onClearAll: () => void; selectedColor: string | null; onColorChange: (hex: string) => void; detailMode?: boolean; onDetailMode?: () => void; labelAreaMode?: 'hide' | 'show' | null; onLabelAreaMode?: (mode: 'hide' | 'show') => void; addPointMode?: boolean; onAddPointMode?: () => void }) {
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
        {selectedAnnotation.kind === 'text' && <div className="annotation-fontsize" style={{ left: `${Math.min(selectedAnnotation.x, selectedAnnotation.x + selectedAnnotation.w)}%`, top: `${Math.min(selectedAnnotation.y, selectedAnnotation.y + selectedAnnotation.h) + Math.abs(selectedAnnotation.h)}%` }}
          onPointerDown={(event) => { event.stopPropagation(); event.preventDefault(); }}>
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

function CorrectionPoints({ coefficient, points, labels = true, visibleLabelIds, onLabelToggle, overrides, onOverrideChange, labelFontFamily }: { coefficient: number; points: PointResult[]; labels?: boolean; visibleLabelIds?: Set<string>; onLabelToggle?: (id: string) => void; overrides?: Record<string, number>; onOverrideChange?: (id: string, value: number | null) => void; labelFontFamily?: string }) {
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
  }, [layerSize, points, labelHeight, displayFor, formatCorrection, getLabelWidth]);
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
    const labelClasses = ['measure-point__label'];
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

function SheetCanvas({ scan, imageUrl, frameWidth, frameHeight, onRegionsChange, onLayoutsChange, points, coefficient, showPoints, visiblePointIds, onPointToggle, pointOverrides, onOverrideChange, labelFontFamily, annotations, showAnnotations, annotationTool, setAnnotationTool, selectedAnnotationId, setSelectedAnnotationId, onAnnotationCommit, onAnnotationCreate, onAnnotationDelete, detailMode, setDetailMode, labelAreaMode, setLabelAreaMode, addPointMode, onAddPointAt, sampling, sampleError, addedPoints, onRemoveAddedPoint }: { scan: ScanItem; imageUrl: string; frameWidth: number; frameHeight: number; onRegionsChange?: (regions: DetailRegion[]) => void; onLayoutsChange?: (layouts: SheetLayout[]) => void; points: PointResult[]; coefficient: number; showPoints: boolean; visiblePointIds: Set<string>; onPointToggle: (id: string) => void; pointOverrides: Record<string, number>; onOverrideChange: (id: string, value: number | null) => void; labelFontFamily?: string; annotations: Annotation[]; showAnnotations: boolean; annotationTool: AnnotationTool; setAnnotationTool: (tool: AnnotationTool) => void; selectedAnnotationId: string | null; setSelectedAnnotationId: (id: string | null) => void; onAnnotationCommit: (annotation: Annotation) => void; onAnnotationCreate: (annotation: Annotation) => void; onAnnotationDelete: (id: string) => void; detailMode: boolean; setDetailMode: (value: boolean) => void; labelAreaMode: 'hide' | 'show' | null; setLabelAreaMode: (value: 'hide' | 'show' | null) => void; addPointMode: boolean; onAddPointAt: (x: number, y: number) => void; sampling: boolean; sampleError: string | null; addedPoints: PointResult[]; onRemoveAddedPoint: (id: string) => void }) {
  /* 정렬 합성 이미지는 스캔 원본과 크기가 다를 수 있어 프레임 치수를 직접 받는다. */
  const sourceAspect = frameWidth / frameHeight;
  const initialFrontSize = fitAspectSize(sourceAspect, 62, 64);
  const [regions, setRegions] = useState<DetailRegion[]>([]);
  /* 엑셀 내보내기가 Detail 영역을 알아야 해서 위로 올려 준다. */
  useEffect(() => { onRegionsChange?.(regions); }, [regions, onRegionsChange]);
  const [layouts, setLayouts] = useState<SheetLayout[]>([{ id: 'front', kind: 'front', x: 4, y: 7, ...initialFrontSize }]);
  useEffect(() => { onLayoutsChange?.(layouts); }, [layouts, onLayoutsChange]);
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
      const layoutVisiblePointIds = layout.kind === 'front' ? visiblePointIds : new Set(detailPoints.filter((point) => !hiddenDetailPointIds[layout.id]?.has(point.id)).map((point) => point.id));
      const toggleLayoutPoint = layout.kind === 'front' ? onPointToggle : (id: string) => setHiddenDetailPointIds((current) => { const hidden = new Set(current[layout.id] || []); if (hidden.has(id)) hidden.delete(id); else hidden.add(id); return { ...current, [layout.id]: hidden }; });
      const applyAreaPoints = (ids: string[], mode: 'hide' | 'show') => {
        if (layout.kind === 'front') ids.filter((id) => mode === 'hide' ? layoutVisiblePointIds.has(id) : !layoutVisiblePointIds.has(id)).forEach(onPointToggle);
        else setHiddenDetailPointIds((current) => { const hidden = new Set(current[layout.id] || []); ids.forEach((id) => mode === 'hide' ? hidden.add(id) : hidden.delete(id)); return { ...current, [layout.id]: hidden }; });
      };
      return <SheetLayoutFrame key={layout.id} layout={layout} imageAspect={imageAspect} selected={selectedLayoutId === layout.id} onSelect={() => setSelectedLayoutId(layout.id)} onChange={updateLayout} onDelete={region ? () => deleteDetail(region.id) : undefined} title={title}>
        {region ? <div className="detail-crop"><div className="layout-image-clip"><img src={imageUrl} alt={`${region.label} 확대 정면도`} style={{ width: `${10000 / region.w}%`, height: `${10000 / region.h}%`, left: `${-region.x / region.w * 100}%`, top: `${-region.y / region.h * 100}%` }} /></div>{showPoints && <CorrectionPoints coefficient={coefficient} points={detailPoints} visibleLabelIds={layoutVisiblePointIds} onLabelToggle={toggleLayoutPoint} overrides={pointOverrides} onOverrideChange={onOverrideChange} labelFontFamily={labelFontFamily} />}</div>
          : <div className="front-view-layout"><img src={imageUrl} alt="스캔 데이터에서 추출한 정면도" />{addPointMode && layout.kind === 'front' && <><div className="add-point-catcher" onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); const rect = event.currentTarget.getBoundingClientRect(); if (!rect.width || !rect.height) return; onAddPointAt((event.clientX - rect.left) / rect.width * 100, (event.clientY - rect.top) / rect.height * 100); }} />
          {/* 지우기는 거리 판정 대신 포인트 위 전용 버튼으로 받는다. 점이 작아 손으로 정확히 겨누기 어렵다. */}
          {addedPoints.map((added) => <button key={added.id} type="button" className="add-point-remove" style={{ left: `${added.x}%`, top: `${added.y}%` }}
            onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); onRemoveAddedPoint(added.id); }}
            aria-label={`${added.id} 추가 포인트 삭제`} title="이 추가 포인트 삭제"><X size={9} /></button>)}</>}{showPoints && <CorrectionPoints coefficient={coefficient} points={points} visibleLabelIds={layoutVisiblePointIds} onLabelToggle={toggleLayoutPoint} overrides={pointOverrides} onOverrideChange={onOverrideChange} labelFontFamily={labelFontFamily} />}<DetailRegionLayer regions={regions} active={detailMode} selectedId={selectedRegionId} onSelect={setSelectedRegionId} onCreate={createDetail} onChange={updateDetailRegion} onDelete={deleteDetail} /></div>}
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
    { id: 'workspace' as const, label: '분석 작업실', icon: Grid2X2 },
    { id: 'results' as const, label: '엔진 결과', icon: BarChart3 },
    { id: 'service' as const, label: 'ADC 보정 시트', icon: Layers3 },
    { id: 'files' as const, label: '품번 파일 정리', icon: Files },
    { id: 'cad' as const, label: '3D CAD 뷰어', icon: Layers3 },
  ];
  return <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
    <div className="brand"><img className="brand__logo" src="/ajin-industrial-logo.png" alt="아진산업" /></div>
    <nav className="sidebar__nav" aria-label="주 메뉴"><span className="sidebar__eyebrow">ADC WORKSPACE</span>{items.map((item) => { const Icon = item.icon; const disabled = (item.id === 'results' || item.id === 'service') && !hasResult; return <button key={item.id} disabled={disabled} onClick={() => !disabled && setView(item.id)} className={view === item.id ? 'active' : ''}><Icon size={19} /><span>{item.label}</span></button>; })}</nav>
    <div className="sidebar__guide"><CircleHelp size={19} /><div><b>실제 엔진 연결</b><span>모든 처리는 이 PC 안에서 실행</span></div><ChevronRight size={16} /></div>
    <button className="sidebar__collapse" onClick={() => setCollapsed(!collapsed)} aria-label="사이드바 접기"><PanelLeftClose size={18} /><span>메뉴 접기</span></button>
  </aside>;
}

function Header({ scans, activeId, setActiveId, onSaveFile, onLoadFile, onReset, note }: { scans: ScanItem[]; activeId?: string; setActiveId: (id: string) => void; onSaveFile: () => void; onLoadFile: (file: File) => void; onReset: () => void; note?: string | null }) {
  const fileRef = useRef<HTMLInputElement>(null);
  return <header className="topbar"><div><span className="topbar__context">AJIN INDUSTRIAL · DIE ENGINEERING</span><h1>ADC <small>Ajin Die Compensation</small></h1></div><div className="topbar__actions">
    <label className="item-select"><span>현재 품번</span><select value={activeId || ''} disabled={!scans.length} onChange={(e) => setActiveId(e.target.value)}><option value="">등록된 이미지 없음</option>{scans.map((scan) => <option value={scan.id} key={scan.id}>{scan.partNo} · {scan.name}</option>)}</select></label>
    {/* 작업 내용은 자동으로 이 PC 에 남는다. 파일로 빼두면 보관하거나
        다른 사람에게 넘길 수 있다. */}
    <div className="topbar__session">
      {note && <span className="topbar__session-note">{note}</span>}
      <button type="button" className="tool-button" onClick={onSaveFile}
        title="고친 보정값과 설정을 파일로 내려받습니다">작업 저장</button>
      <button type="button" className="tool-button"
        onClick={() => fileRef.current?.click()}
        title="저장해 둔 작업 파일을 불러옵니다">작업 불러오기</button>
      <button type="button" className="tool-button" onClick={onReset}
        title="이 PC 에 남은 작업 내용을 지웁니다">작업 비우기</button>
      <input ref={fileRef} type="file" hidden accept="application/json,.json"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onLoadFile(file);
          event.target.value = '';
        }} />
    </div>
    <button className="icon-button" aria-label="설정"><Settings2 size={19} /></button><div className="profile"><span>KJ</span><div><b>금형생산팀</b><small>관리자</small></div></div>
  </div></header>;
}

/* 파일명에서 품번을 뽑는다. 백엔드 file_naming.py 와 같은 규칙이며
   짧은 형태(64XX2)를 준다 — 컬러바 표와 제로라인 라이브러리의 열쇠다.

   순수 숫자 여섯 자리는 **날짜**라 품번으로 보지 않는다. NC 데이터가
   `260825_JDZ_DASH LWR_OP10_...ZIP` 처럼 날짜를 앞에 달고 오는데,
   예전 규칙은 이걸 품번이라고 집어냈다. */
function partNoFromName(name: string): string | null {
  for (const token of name.toUpperCase().split(/[_\s]+/)) {
    const head = token.split(/[-/]/)[0];
    if (!/^[0-9]{2}[A-Z0-9]{2,4}$/.test(head)) continue;
    if (/^[0-9]+$/.test(head) && !/[-/]/.test(token)) continue;   // 날짜
    return head;
  }
  return null;
}

/* 컬러바 범위가 등록된 품번. 백엔드 PRODUCT_COLORBAR_MM 과 같은 표이며
   분석 응답의 knownParts 로도 확인할 수 있다. */
const KNOWN_PARTS = ['64XX2', '67XX6', '71XX2'];

function Workspace({ scans, setScans, onOpenResults, backendOnline }: { scans: ScanItem[]; setScans: React.Dispatch<React.SetStateAction<ScanItem[]>>; onOpenResults: (id: string) => void; backendOnline: boolean | null }) {
  const [dragging, setDragging] = useState(false);
  const analyzingCount = scans.filter((scan) => scan.status === 'analyzing').length;
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
  const attachProduct = (id: string, file: File) => setScans((current) => current.map((scan) => {
    if (scan.id !== id) return scan;
    if (scan.productUrl) URL.revokeObjectURL(scan.productUrl);
    return { ...scan, productFile: file, productUrl: URL.createObjectURL(file), status: scan.status === 'done' ? 'ready' : scan.status };
  }));
  const detachProduct = (id: string) => setScans((current) => current.map((scan) => {
    if (scan.id !== id) return scan;
    if (scan.productUrl) URL.revokeObjectURL(scan.productUrl);
    return { ...scan, productFile: undefined, productUrl: undefined };
  }));
  return <section className="page page--workspace">
    <div className="page-heading"><div><h2>스캔 이미지를 한 번에 분석하세요</h2><p>업로드한 이미지는 이 PC의 세 엔진으로 처리되며 외부 서버로 전송되지 않습니다.</p></div><div className="step-pills"><span className="done"><Check size={14} /> 1. 이미지 등록</span><span className={analyzingCount ? 'active' : ''}>2. 엔진 분석</span><span>3. 보정 시트</span></div></div>
    <div className="workspace-grid"><div className="upload-panel card"><div className="card-title"><div><h3>스캔 이미지 등록</h3><p>PNG, JPG, WEBP · 여러 파일 동시 선택 가능</p></div><span className="count-chip">{scans.length}개 등록</span></div>
      <label className={`dropzone ${dragging ? 'dropzone--active' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(e: DragEvent<HTMLLabelElement>) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}><input type="file" multiple accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff" onChange={(e: ChangeEvent<HTMLInputElement>) => e.target.files && addFiles(e.target.files)} /><span className="dropzone__icon"><UploadCloud size={29} /></span><b>스캔 이미지를 여기에 놓으세요</b><span>또는 클릭하여 파일 선택</span><em>여러 품번의 이미지를 동시에 올릴 수 있습니다</em></label>
      <div className="file-list"><div className="file-list__head"><span>등록된 이미지</span><button><ListFilter size={15} /> 상태순</button></div>{!scans.length && <div className="empty-file-list">아직 등록된 이미지가 없습니다.</div>}{scans.map((scan) => <div className="file-row" key={scan.id}><div className={`file-thumb tone-${scan.tone}`}><img src={scan.url} alt="" /></div><div className="file-row__name"><b>{scan.name}</b><span>{scan.partNo} · {scan.error || scan.size}</span><span className="product-slot">{scan.productFile ? <><ImageIcon size={12} /> 제품데이터 {scan.productFile.name}<button type="button" onClick={() => detachProduct(scan.id)} aria-label="제품데이터 해제">해제</button></> : <label><input type="file" accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff" onChange={(event: ChangeEvent<HTMLInputElement>) => event.target.files?.[0] && attachProduct(scan.id, event.target.files[0])} /><UploadCloud size={12} /> 제품데이터 직접 지정 (없으면 품번으로 자동)</label>}</span></div><span className={`status status--${scan.status}`}>{scan.status === 'done' ? <><Check size={13} /> 분석 완료</> : scan.status === 'analyzing' ? <><Activity size={13} /> 분석 중</> : scan.status === 'error' ? '오류' : '대기'}</span>{scan.status === 'done' ? <button className="text-button" onClick={() => onOpenResults(scan.id)}>결과 보기 <ArrowRight size={14} /></button> : <button className="icon-button icon-button--small" onClick={() => removeScan(scan.id)} aria-label={`${scan.name} 삭제`}><X size={15} /></button>}</div>)}</div>
      <button className="primary-button primary-button--wide" onClick={analyzeAll} disabled={!backendOnline || analyzingCount > 0 || !scans.some((scan) => scan.status === 'ready' || scan.status === 'error')}><Play size={17} fill="currentColor" /> {analyzingCount ? `${analyzingCount}개 이미지 분석 중` : backendOnline === false ? '로컬 엔진 서버 연결 필요' : '대기 이미지 전체 분석 시작'}<ArrowRight size={18} /></button>
    </div><aside className="engine-panel"><div className="card engine-overview"><div className="card-title"><div><h3>분석 엔진</h3><p>실제 연결 상태</p></div><span className={`live-dot ${backendOnline === false ? 'offline' : ''}`}>{backendOnline == null ? '확인 중' : backendOnline ? '연결됨' : '연결 안 됨'}</span></div>{(Object.keys(engineMeta) as Engine[]).map((key, index) => { const meta = engineMeta[key]; return <div className="engine-row" key={key}><span className="engine-row__number" style={{ background: `${meta.color}16`, color: meta.color }}>0{index + 1}</span><div><b>{meta.name}</b><span>{meta.short}</span></div>{backendOnline ? <ShieldCheck size={18} color={meta.color} /> : <X size={18} color="#a2aab4" />}</div>; })}</div><div className="tip-card"><span><Gauge size={20} /></span><div><b>편차값 판독 방식</b><p>Qwen2.5-VL-3B를 RTX GPU에서 실행하며, 모델은 인터넷 없이 로컬 파일만 사용합니다.</p></div></div></aside></div>
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
function AlignmentBar({ alignment, partNumber, source, transferred, total, busy, confirmed, onFlip, onConfirm }: { alignment: AlignmentInfo; partNumber: string | null; source: string | null; transferred: number; total: number; busy: boolean; confirmed: boolean; onFlip?: (flipX?: boolean, flipY?: boolean) => void; onConfirm?: () => void }) {
  const trusted = alignment.confident;
  return <div className={`alignment-bar ${trusted ? '' : 'alignment-bar--check'}`}>
    <span className="alignment-bar__state">{trusted ? <><ShieldCheck size={14} /> 자동 판정 신뢰 가능</> : <><MoveRight size={14} /> 방향 확인 필요</>}</span>
    <span className="alignment-bar__facts"><b>{partNumber || '품번 미확인'}</b><small>{source || '제품데이터 없음'}</small><small>외형 {(alignment.outlineIou * 100).toFixed(1)}% · 구멍 {(alignment.holeIou * 100).toFixed(1)}% · 2위와 격차 {alignment.margin.toFixed(3)}</small><small>전사 {transferred}/{total}개</small></span>
    {onFlip && <span className="alignment-bar__actions">{/* 분석 결과는 화면에 남아 있으므로, 엔진이 바뀌면 좌표만 다시 받아 온다. Qwen 판독은 다시 하지 않는다. */}<button type="button" disabled={busy} onClick={() => onFlip()} title="정렬만 다시 계산합니다. 방향은 자동 판정과 확정 저장분을 따릅니다">정렬 다시 계산</button><button type="button" disabled={busy} onClick={() => onFlip(!alignment.flipX, alignment.flipY)}>좌우 뒤집기</button><button type="button" disabled={busy} onClick={() => onFlip(alignment.flipX, !alignment.flipY)}>상하 뒤집기</button>{onConfirm && <button type="button" className="primary" disabled={busy || confirmed || !partNumber} onClick={onConfirm}>{confirmed ? <><Check size={13} /> 품번에 저장됨</> : '이 방향으로 확정'}</button>}</span>}
  </div>;
}

function Results({ scan, onService, hiddenPointIds, onPointToggle, onAllPointsToggle, onRealign, onConfirmAlignment }: { scan: ScanItem; onService: () => void; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; onAllPointsToggle: (visible: boolean) => void; onRealign?: (flipX?: boolean, flipY?: boolean) => Promise<void>; onConfirmAlignment?: () => Promise<void> }) {
  const [engine, setEngine] = useState<Engine>('label');
  /* 편차 뷰는 세 가지로 본다: 스캔 위, 제품데이터 위, 그리고 정렬 확인용 실루엣 겹침. */
  const [frame, setFrame] = useState<'scan' | 'product' | 'overlay'>('scan');
  const [busy, setBusy] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const result = scan.result!; const meta = engineMeta[engine]; const summary = engineSummary(engine, result);
  const engineWarnings = result.warningsByEngine?.[engine] ?? (engine === 'zero' ? result.warnings : []);
  const visibleLabelIds = new Set(result.points.filter((point) => !hiddenPointIds.has(point.id)).map((point) => point.id));
  const alignment = result.alignment;
  const productReady = engine === 'deviation' && Boolean(result.productImage && alignment);
  const showFrame = productReady ? frame : 'scan';
  /* 제품데이터 뷰에서는 같은 포인트의 좌표만 제품 기준으로 바꿔 넘긴다. */
  const productPoints = result.points.filter((point) => point.xProduct !== undefined && point.yProduct !== undefined).map((point) => ({ ...point, x: point.xProduct!, y: point.yProduct! }));
  const image = showFrame === 'product' ? result.productImage : showFrame === 'overlay' ? result.alignmentOverlay : engine === 'zero' ? result.zeroOverlay : result.cleanImage || scan.url;
  const frameWidth = showFrame === 'scan' || !alignment ? result.source.width : alignment.productSize[0];
  const frameHeight = showFrame === 'scan' || !alignment ? result.source.height : alignment.productSize[1];
  const toggleLabel = onPointToggle;
  const allLabelsVisible = result.points.length > 0 && visibleLabelIds.size === result.points.length;
  const runRealign = async (flipX?: boolean, flipY?: boolean) => {
    if (!onRealign || busy) return;
    setBusy(true); setConfirmed(false);
    try { await onRealign(flipX, flipY); } finally { setBusy(false); }
  };
  const runConfirm = async () => {
    if (!onConfirmAlignment || busy) return;
    setBusy(true);
    try { await onConfirmAlignment(); setConfirmed(true); } finally { setBusy(false); }
  };
  return <section className="page page--results"><div className="page-heading page-heading--compact"><div><span className="breadcrumb">분석 작업실 <ChevronRight size={14} /> {scan.partNo}</span><h2>엔진별 실제 분석 결과</h2><p>{scan.name} · {result.source.width} × {result.source.height}px</p></div><button className="primary-button" onClick={onService}>보정 시트 만들기 <ArrowRight size={17} /></button></div>
    <div className="result-tabs" role="tablist">{(Object.keys(engineMeta) as Engine[]).map((key, index) => { const item = engineMeta[key]; const failed = Boolean(result.errors[key]); return <button role="tab" aria-selected={engine === key} className={engine === key ? 'active' : ''} onClick={() => setEngine(key)} key={key}><span style={{ color: failed ? '#bd4650' : item.color }}>0{index + 1}</span><div><b>{item.name}</b><small>{failed ? '실행 오류' : item.short}</small></div>{!failed && <Check size={17} />}</button>; })}</div>
    <div className="results-layout"><div className="viewer-card card"><div className="viewer-toolbar"><div><span className={`status ${result.errors[engine] ? 'status--error' : 'status--done'}`}>{result.errors[engine] ? <><X size={13} /> 실행 실패</> : <><Check size={13} /> 실제 분석 완료</>}</span><b>{meta.name}</b></div><div>{productReady && <div className="frame-toggles">{([['scan', '스캔 위'], ['product', '제품데이터 위'], ['overlay', '정렬 확인']] as const).map(([key, label]) => <button key={key} type="button" className={showFrame === key ? 'active' : ''} onClick={() => setFrame(key)}>{label}</button>)}</div>}{engine === 'deviation' && <button className="tool-button" onClick={() => onAllPointsToggle(!allLabelsVisible)}>{allLabelsVisible ? <EyeOff size={14} /> : <Eye size={14} />} 라벨 전체 {allLabelsVisible ? 'OFF' : 'ON'}</button>}</div></div>{productReady && alignment && <AlignmentBar alignment={alignment} partNumber={result.partNumber} source={result.productSource} transferred={productPoints.length} total={result.points.length} busy={busy} confirmed={confirmed} onFlip={onRealign ? runRealign : undefined} onConfirm={onConfirmAlignment ? runConfirm : undefined} />}<div className={`viewer-stage ${engine === 'deviation' ? 'viewer-stage--light' : ''}`}><Heatmap key={`${scan.id}-${engine}-${showFrame}`} imageUrl={image} width={frameWidth} height={frameHeight} lightBackground={engine === 'deviation'}>{engine === 'deviation' && showFrame !== 'overlay' && <CorrectionPoints coefficient={-1} points={showFrame === 'product' ? productPoints : result.points} visibleLabelIds={visibleLabelIds} onLabelToggle={toggleLabel} />}</Heatmap></div><div className="viewer-legend"><span><i className="legend-dot" style={{ background: meta.color }} /> 현재 표시: {meta.name}</span><span>{engine === 'deviation' ? '라벨이나 포인트 점을 누르면 개별 표시를 켜고 끌 수 있습니다.' : '표시된 값과 위치는 업로드 이미지의 실제 엔진 결과입니다.'}</span></div></div>
      <aside className="inspection-panel"><div className="score-card card"><span className="score-card__icon" style={{ color: meta.color, background: `${meta.color}12` }}>{engine === 'label' ? <Sparkles /> : engine === 'deviation' ? <Activity /> : <Gauge />}</span><span>핵심 결과</span><strong style={{ color: result.errors[engine] ? '#bd4650' : meta.color }}>{summary.stat}</strong><p>{summary.detail}</p></div><div className="card plain-summary"><h3>쉽게 보는 결과</h3><div className="summary-line"><Check size={16} /><div><b>처리 방식</b><span>{engine === 'label' ? 'label_removal의 인페인팅 결과입니다.' : engine === 'deviation' ? '라벨 제거 이미지에 deviation_extraction의 지시선 끝점과 판독값을 겹쳐 표시합니다.' : 'zero_line_detection의 컬러바 기반 결과입니다.'}</span></div></div>{engineWarnings.length > 0 && <div className="summary-line warning"><MoveRight size={16} /><div><b>확인 필요</b><span>{engineWarnings[0]}</span></div></div>}</div><div className="card mini-table"><div className="card-title"><h3>검출 포인트</h3><span>라벨 {visibleLabelIds.size}/{result.points.length}</span></div>{result.points.map((point) => { const visible = visibleLabelIds.has(point.id); return <div className="point-list-row" key={point.id}><span>{point.id}</span><b className={point.value > 0 ? 'positive' : 'negative'}>{point.value > 0 ? '+' : ''}{point.value.toFixed(3)} mm</b><small>{point.xPx}, {point.yPx}</small><button type="button" className={visible ? 'label-visibility active' : 'label-visibility'} onClick={() => toggleLabel(point.id)} aria-label={`${point.id} 라벨 ${visible ? '숨기기' : '표시하기'}`} title={`라벨 ${visible ? 'OFF' : 'ON'}`}>{visible ? <Eye size={14} /> : <EyeOff size={14} />}</button></div>; })}{!result.points.length && <p className="empty-mini">검출된 포인트가 없습니다.</p>}</div></aside>
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
  const [busy, setBusy] = useState(false);
  const [operation, setOperation] = useState<'copy' | 'move'>('copy');
  const [conflict, setConflict] = useState<'rename' | 'skip' | 'overwrite'>('rename');
  const [notice, setNotice] = useState<{ tone: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [showDatabase, setShowDatabase] = useState(false);
  const [databaseUrl, setDatabaseUrl] = useState('');
  const [showFolderOrder, setShowFolderOrder] = useState(false);
  const [folderOrder, setFolderOrder] = useState<FolderAxis[]>(['family', 'category', 'product']);
  const [axisOptions, setAxisOptions] = useState<FolderAxisOption[]>([]);
  const [savingOrder, setSavingOrder] = useState(false);

  const axisLabel = (axis: FolderAxis) => axisOptions.find((option) => option.id === axis)?.label || axis;

  const loadFolderOrder = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/folder-order`);
      const data = await response.json() as FolderOrderResponse;
      if (!response.ok) throw new Error(data.error || '폴더 순서를 불러오지 못했습니다.');
      setFolderOrder(data.folderOrder);
      setAxisOptions(data.axes);
    } catch { /* 상태 카드/기본 순서로 충분히 안내되므로 조용히 넘어간다. */ }
  }, []);

  const [showPaths, setShowPaths] = useState(false);
  const [pathsInfo, setPathsInfo] = useState<OrganizerPathsResponse | null>(null);
  const [sourceRootInput, setSourceRootInput] = useState('');
  const [destinationRootInput, setDestinationRootInput] = useState('');
  const [savingPaths, setSavingPaths] = useState(false);

  const loadOrganizerPaths = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/paths`);
      const data = await response.json() as OrganizerPathsResponse & { error?: string };
      if (!response.ok) throw new Error(data.error || '경로 설정을 불러오지 못했습니다.');
      setPathsInfo(data);
      setSourceRootInput(data.sourceRoot);
      setDestinationRootInput(data.destinationRoot);
    } catch { /* 저장소 상태 카드에 이미 현재 경로가 표시되므로 조용히 넘어간다. */ }
  }, []);

  const saveOrganizerPaths = async () => {
    setSavingPaths(true);
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/paths`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sourceRoot: sourceRootInput, destinationRoot: destinationRootInput }),
      });
      const data = await response.json() as OrganizerPathsResponse & { error?: string };
      if (!response.ok) throw new Error(data.error || '경로를 저장하지 못했습니다.');
      setNotice({ tone: 'success', text: '경로를 저장했습니다. 다음 스캔부터 적용됩니다.' });
      setShowPaths(false);
      setPathsInfo(data);
      await loadStatus(true); await loadFolders();
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : '경로 저장 중 오류가 발생했습니다.' });
    } finally { setSavingPaths(false); }
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

  const scanSource = async () => {
    setBusy(true); setNotice(null);
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/scan`);
      const data = await response.json() as { items?: FileOrganizerItem[]; error?: string };
      if (!response.ok) throw new Error(data.error || '원본 폴더를 스캔하지 못했습니다.');
      mergeItems(data.items || []);
      setNotice({ tone: 'info', text: `원본 폴더에서 ${(data.items || []).length}개 파일을 분석했습니다.` });
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : '스캔 중 오류가 발생했습니다.' });
    } finally { setBusy(false); }
  };

  const uploadFiles = async (files: FileList | File[]) => {
    if (!files.length) return;
    setBusy(true); setNotice(null);
    try {
      const form = new FormData();
      Array.from(files).forEach((file) => form.append('files', file, file.name));
      const response = await fetch(`${API_BASE}/api/file-organizer/upload`, { method: 'POST', body: form });
      const data = await response.json() as { items?: FileOrganizerItem[]; error?: string };
      if (!response.ok) throw new Error(data.error || '파일을 등록하지 못했습니다.');
      mergeItems(data.items || []);
      setNotice({ tone: 'info', text: `${(data.items || []).length}개 파일의 자동 분류가 끝났습니다.` });
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : '파일 등록 중 오류가 발생했습니다.' });
    } finally { setBusy(false); }
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
    if (!window.confirm('실제 정리 대상 폴더 안의 폴더들을 새 순서로 지금 바로 옮깁니다. 계속할까요?')) return;
    setSavingOrder(true);
    try {
      const response = await fetch(`${API_BASE}/api/file-organizer/folder-order`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folderOrder: folderOrder }),
      });
      const data = await response.json() as FolderOrderResponse;
      if (!response.ok) throw new Error(data.error || '폴더 순서를 저장하지 못했습니다.');
      setFolderOrder(data.folderOrder);
      const migration = data.migration;
      const summary = migration
        ? migration.moved > 0
          ? `폴더 ${migration.moved}개를 새 구조로 옮겼습니다${migration.errors.length ? ` · 오류 ${migration.errors.length}건` : ''}.`
          : '이미 이 순서였습니다 — 옮길 폴더가 없습니다.'
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
    if (axis === 'family') return '64XX2';
    if (axis === 'category') return '01. 3D제품데이터';
    return 'JD PNL DASH 64XX2-DR000';
  }).join(' / ');
  return <section className="page page--file-organizer">
    <div className="page-heading"><div><h2>품번 태그로 파일을 자동 정리하세요</h2><p>파일명을 분석해 품번 계열, 자료유형, 상세 품번 폴더를 확인한 뒤 로컬 또는 NAS로 복사·이동합니다.</p></div><div className="file-heading-actions"><button type="button" className="file-order-pill" onClick={() => setShowFolderOrder((current) => !current)}><Layers3 size={14} /> {folderOrder.map(axisLabel).join(' → ')}</button><button type="button" className={`file-db-pill ${db?.connected ? 'connected' : db?.connected === false ? 'error' : ''}`} onClick={() => setShowDatabase((current) => !current)}><i /> {db?.label || 'MariaDB 확인 중'}</button></div></div>
    {notice && <div className={`file-organizer-notice ${notice.tone}`}>{notice.tone === 'success' ? <CheckCircle2 size={16} /> : notice.tone === 'error' ? <AlertTriangle size={16} /> : <CircleHelp size={16} />}<span>{notice.text}</span><button onClick={() => setNotice(null)} aria-label="알림 닫기"><X size={14} /></button></div>}
    {showFolderOrder && <div className="card file-order-settings">
      <div className="file-order-settings__intro"><ListFilter size={20} /><span><b>폴더 구조 순서</b><small>자료유형과 차종·상세품번, 두 축의 쌓는 순서를 자유롭게 바꿀 수 있습니다. 저장하면 기존 폴더도 세부 하위 구조를 보존한 채 새 순서로 옮겨집니다.</small></span></div>
      <ol className="file-order-axis-list">{folderOrder.map((axis, index) => <li key={axis}><span className="file-order-axis-index">{index + 1}</span><span className="file-order-axis-label">{axisLabel(axis)}</span><span className="file-order-axis-buttons"><button type="button" onClick={() => moveAxis(index, -1)} disabled={index === 0} aria-label="위로"><ChevronDown size={14} style={{ transform: 'rotate(180deg)' }} /></button><button type="button" onClick={() => moveAxis(index, 1)} disabled={index === folderOrder.length - 1} aria-label="아래로"><ChevronDown size={14} /></button></span></li>)}</ol>
      <div className="file-order-preview"><small>예시 경로</small><code>{examplePath}</code></div>
      <button type="button" className="primary-button" onClick={saveFolderOrder} disabled={savingOrder}>{savingOrder ? '저장 중…' : '이 순서로 저장'}</button>
    </div>}
    {showDatabase && <div className="card file-database-settings"><div><Database size={20} /><span><b>MariaDB 연결</b><small>파일은 로컬/NAS에, 태그와 작업 이력은 MariaDB에 저장됩니다.</small></span></div><input value={databaseUrl} onChange={(event) => setDatabaseUrl(event.target.value)} placeholder="mysql://사용자:비밀번호@서버:3306/file_organizer" /><button type="button" onClick={() => setDatabaseUrl('mysql://file_demo:file_demo_password@127.0.0.1:3307/file_organizer?charset=utf8mb4&connect_timeout=5')}>데모 설정</button><button type="button" className="primary-button" onClick={connectDatabase} disabled={busy}>연결 테스트·저장</button></div>}
    <div className="file-storage-strip">
      <div><Server size={18} /><span><small>원본 폴더</small><b title={status?.sourceRoot}>{status?.sourceRoot || '확인 중'}</b></span><em className={status?.sourceAvailable ? 'ok' : ''}>{status?.sourceAvailable ? '연결됨' : '경로 없음'}</em><button type="button" className="file-storage-action" onClick={() => void openInExplorer('source')} title="탐색기에서 열기" aria-label="원본 폴더 탐색기에서 열기"><FolderOpen size={14} /></button></div>
      <ChevronRight size={17} />
      <div><HardDrive size={18} /><span><small>정리 대상 · 추후 NAS</small><b title={status?.destinationRoot}>{status?.destinationRoot || '확인 중'}</b></span><em className={status?.destinationAvailable ? 'ok' : ''}>{status?.destinationAvailable ? '연결됨' : '경로 없음'}</em><button type="button" className="file-storage-action" onClick={() => void openInExplorer('destination')} title="탐색기에서 열기" aria-label="정리 대상 폴더 탐색기에서 열기"><FolderOpen size={14} /></button></div>
      <button type="button" className="file-storage-action file-storage-edit" onClick={() => setShowPaths((current) => !current)} title="경로 변경" aria-label="경로 변경"><Settings2 size={14} /></button>
      <div className="file-storage-metrics"><span>카탈로그 <b>{db?.catalogCount || 0}</b></span><span>작업 이력 <b>{db?.operationCount || 0}</b></span></div>
    </div>
    {showPaths && <div className="card file-path-settings">
      <div className="file-path-settings__intro"><Settings2 size={20} /><span><b>원본·정리 대상 경로</b><small>{(pathsInfo?.sourceLocked || pathsInfo?.destinationLocked) ? 'ui/.env에 경로가 고정되어 있어 여기서는 바꿀 수 없습니다.' : '두 경로를 바꾸면 다음 스캔부터 적용됩니다.'}</small></span></div>
      <label>원본 폴더<input value={sourceRootInput} onChange={(event) => setSourceRootInput(event.target.value)} disabled={pathsInfo?.sourceLocked} placeholder="C:\path\to\incoming-files" /></label>
      <label>정리 대상 폴더<input value={destinationRootInput} onChange={(event) => setDestinationRootInput(event.target.value)} disabled={pathsInfo?.destinationLocked} placeholder="C:\path\to\organized 또는 NAS 경로" /></label>
      <button type="button" className="primary-button" onClick={saveOrganizerPaths} disabled={savingPaths || pathsInfo?.sourceLocked || pathsInfo?.destinationLocked}>{savingPaths ? '저장 중…' : '저장'}</button>
    </div>}
    <div className="file-organizer-grid">
      <div className="card file-organizer-queue"><div className="card-title"><div><h3>정리 대기 파일</h3><p>외부 파일을 끌어 놓거나 지정된 원본 폴더를 스캔하세요.</p></div><div className="file-queue-actions"><button type="button" onClick={scanSource} disabled={busy}><RefreshCw size={14} /> 원본 스캔</button><label><UploadCloud size={14} /> 파일 선택<input type="file" multiple onChange={(event) => event.target.files && void uploadFiles(event.target.files)} /></label><span className="count-chip">{items.length}개</span></div></div>
        <label className={`file-organizer-drop ${dragging ? 'active' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event: DragEvent<HTMLLabelElement>) => { event.preventDefault(); setDragging(false); void uploadFiles(event.dataTransfer.files); }}><input type="file" multiple onChange={(event) => event.target.files && void uploadFiles(event.target.files)} /><UploadCloud size={25} /><b>{busy ? '처리 중입니다…' : '정리할 파일을 여기에 놓으세요'}</b><span>품번 · OP공정 · 자료유형 태그를 자동 감지합니다.</span></label>
        <div className="file-organizer-table"><div className="file-organizer-table__head"><input type="checkbox" checked={items.length > 0 && selected.size === items.length} onChange={(event) => setSelected(event.target.checked ? new Set(items.map((item) => item.id)) : new Set())} aria-label="전체 선택" /><span>파일명 / 감지 태그</span><span>자료유형</span><span>예정 위치</span><span>신뢰도</span><span /></div>{items.map((item) => <div className="file-organizer-row" key={item.id} draggable onDragStart={(event) => { const ids = selected.has(item.id) ? [...selected] : [item.id]; event.dataTransfer.setData('text/ajin-file-ids', JSON.stringify(ids)); event.dataTransfer.effectAllowed = 'move'; }} title={item.reasons.join('\n')}><input type="checkbox" checked={selected.has(item.id)} onChange={() => toggleSelected(item.id)} aria-label={`${item.name} 선택`} /><span className="file-organizer-name"><File size={17} /><span><b>{item.name}</b><small>{[item.customer, item.itemNo, item.productName, item.process].filter(Boolean).join(' · ') || '품번 태그 미검출'} <em>{item.sourceKind === 'upload' ? '업로드' : '원본'}</em></small></span></span><span className={`file-category category-${item.categoryKey || 'unknown'}`}>{item.categoryKey || '--'} {item.categoryLabel}</span><span className="file-target-path" title={item.targetPath}>{item.targetDir || '_미분류'}</span><span className={`file-confidence ${item.confidence >= 70 ? 'good' : ''}`}>{item.confidence}%</span><button type="button" className="file-row-delete" onClick={() => removeItem(item)} aria-label={`${item.name} 대기열에서 삭제`} title="대기열에서 삭제"><Trash2 size={14} /></button></div>)}{!items.length && <div className="file-organizer-empty">분류할 파일이 아직 없습니다.</div>}</div>
        <div className="file-execute-bar"><div className="file-operation-switch"><button type="button" className={operation === 'copy' ? 'active' : ''} onClick={() => setOperation('copy')}><Copy size={14} /> 복사</button><button type="button" className={operation === 'move' ? 'active' : ''} onClick={() => setOperation('move')}><Move size={14} /> 이동</button></div><label>동명 파일<select value={conflict} onChange={(event) => setConflict(event.target.value as typeof conflict)}><option value="rename">자동 이름 변경</option><option value="skip">건너뛰기</option><option value="overwrite">덮어쓰기</option></select></label><button type="button" className="primary-button file-execute" onClick={execute} disabled={busy || !activeItems.length}>{busy ? '처리 중…' : `선택 ${activeItems.length}개 정리 실행`} <ArrowRight size={16} /></button></div>
      </div>
      <aside className="card file-organizer-target"><div className="card-title"><div><h3>대상 폴더 탐색기</h3><p>파일을 끌어 놓아 위치를 바꾸세요. (실제 폴더 구조)</p></div></div><div className="file-path-preview"><FolderOpen size={18} /><span>{rootName}</span></div><div className="organizer-folder-tree"><button type="button" className="organizer-folder-root" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); try { assignTarget(JSON.parse(event.dataTransfer.getData('text/ajin-file-ids')) as string[], ''); } catch {} }}><ChevronDown size={14} /><FolderOpen size={17} /><span>{rootName}</span></button>{rootEntries.filter((entry) => entry.isDirectory).map((entry) => <OrganizerFolderNode key={entry.path} entry={entry} onAssign={assignTarget} />)}{rootEntries.filter((entry) => !entry.isDirectory).map((file) => <div className="organizer-folder-file" key={file.path} title={file.name}><File size={13} /><span>{file.name}</span></div>)}{!rootEntries.length && <div className="file-target-hint">대상 폴더 경로를 확인해 주세요.</div>}</div><div className="file-target-hint"><b>세부 자동 분류</b><span>금형도면은 LAYOUT·구조도·패턴도·완성도와 OP별로, 문서는 성형해석·보정이력으로, NC데이터는 OP10~OP50으로 나뉩니다.</span></div><div className="file-target-hint"><b>드래그 앤 드롭</b><span>파일을 폴더 위에 놓아 자동 분류 위치를 직접 수정할 수 있습니다.</span></div></aside>
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
    <div className="card-title"><div><h3>보정 이력</h3><p>{partNo} 품번 수동 수정 기록입니다.</p></div><button type="button" className="correction-history__reload" onClick={onReload} disabled={loading} aria-label="이력 새로고침" title="이력 새로고침">↻</button></div>
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

function ServicePreview({ scan, folderAvailable, hiddenPointIds, onPointToggle, onKeyPointsOnly, onAllPointsToggle, pointOverrides, onOverrideChange, onClearAllOverrides, annotations = [], setAnnotations, sheetTitle, onSheetTitleChange, sheetTitleFonts, onSheetTitleFontChange, sheetTitleFontSizes, onSheetTitleFontSizeChange, worker, onWorkerChange }: { scan: ScanItem; folderAvailable: boolean; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; onKeyPointsOnly: () => void; onAllPointsToggle: (visible: boolean) => void; pointOverrides: Record<string, number>; onOverrideChange: (id: string, value: number | null) => void; onClearAllOverrides: () => void; annotations: Annotation[]; setAnnotations: (updater: (current: Annotation[]) => Annotation[]) => void; sheetTitle: SheetTitleValues; onSheetTitleChange: (field: SheetTitleField, value: string) => void; sheetTitleFonts: SheetTitleFonts; onSheetTitleFontChange: (field: SheetTitleField, fontFamily: string) => void; sheetTitleFontSizes: SheetTitleFontSizes; onSheetTitleFontSizeChange: (field: SheetTitleField, size: number) => void; worker: string; onWorkerChange: (value: string) => void }) {
  const result = scan.result!; const points = result.points; const [coefficient, setCoefficient] = useState(1); const [showPoints, setShowPoints] = useState(true); const [showZero, setShowZero] = useState(true);
  /* 보정치 수치 라벨(+1.5 등) 글꼴 — "선택하면 자유롭게" 가 아니라 시트 전체 한 번에 바뀌는 값이라 여기 하나로 둔다. */
  const [pointLabelFont, setPointLabelFont] = useState(DEFAULT_POINT_LABEL_FONT);
  /* 보정시트에 들어가는 그림은 편차 히트맵이 아니라 깨끗한 제품데이터다.
     정렬된 제품데이터가 있으면 그쪽을 기본으로 쓰고, 없을 때만 스캔으로 물러선다. */
  const alignment = result.alignment;
  const productReady = Boolean(result.productImage && alignment);
  const [useProduct, setUseProduct] = useState(true);
  const onProduct = productReady && useProduct;
  /* 제로라인 오버레이는 스캔 좌표계에 그려진 이미지라 제품데이터 위에는 얹을 수 없다. */
  const zeroReady = Boolean(result.zeroOverlay) && !onProduct;
  const frameWidth = onProduct ? alignment!.productSize[0] : result.source.width;
  const frameHeight = onProduct ? alignment!.productSize[1] : result.source.height;
  const [tool, setTool] = useState<AnnotationTool>('select'); const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null); const [showAnnotations, setShowAnnotations] = useState(true); const [detailMode, setDetailMode] = useState(false); const [labelAreaMode, setLabelAreaMode] = useState<'hide' | 'show' | null>(null);
  /* 엑셀 내보내기가 실제 UI 배치(정면도·디테일 뷰의 캔버스 % 좌표)를
     알아야 주석·창 위치를 시트에 그대로 옮길 수 있다. SheetCanvas 안에
     있는 layouts 상태를 콜백으로 위로 끌어올린다. */
  const [sheetLayouts, setSheetLayouts] = useState<SheetLayout[]>([]);
  /* 엔진 결과는 그대로 두고 작업자가 찍은 포인트만 따로 얹는다. */
  const [addedPoints, setAddedPoints] = useState<PointResult[]>([]);
  const addedPointSequenceRef = useRef(0);
  const [addPointMode, setAddPointMode] = useState(false);
  /* 보정시트에는 검출된 라벨을 전부 올리지 않는다. 품번을 처음 열 때 주요 포인트만
     켜 두고, 그 뒤로 작업자가 손댄 것은 다시 건드리지 않는다. */
  const keySelection = result.keySelection;
  const presetAppliedRef = useRef('');
  useEffect(() => {
    if (presetAppliedRef.current === scan.id) return;
    presetAppliedRef.current = scan.id;
    if (keySelection?.ids.length && hiddenPointIds.size === 0) onKeyPointsOnly();
  }, [scan.id, keySelection, hiddenPointIds.size, onKeyPointsOnly]);
  const [sampling, setSampling] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  /* 엑셀 내보내기 */
  const [detailRegions, setDetailRegions] = useState<DetailRegion[]>([]);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const removeAddedPoint = (id: string) => setAddedPoints((current) => current.filter((item) => item.id !== id));
  const addPointAt = async (xNorm: number, yNorm: number) => {
    setSampling(true); setSampleError(null);
    try {
      /* 클릭 좌표는 화면에 보이는 이미지 기준이다. 색 역산은 편차 스캔에서만
         가능하므로, 제품데이터를 보고 있으면 변환을 되짚어 스캔 좌표로 보낸다. */
      let sampleX = xNorm; let sampleY = yNorm;
      if (onProduct && alignment) {
        const [a, , tx, , d, ty] = alignment.matrix;
        const [productW, productH] = alignment.productSize;
        const [scanW, scanH] = alignment.scanSize;
        if (!a || !d) { setSampleError('정렬 정보가 올바르지 않습니다.'); return; }
        sampleX = ((xNorm / 100 * productW - tx) / a) / scanW * 100;
        sampleY = ((yNorm / 100 * productH - ty) / d) / scanH * 100;
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
        const [a, , tx, , d, ty] = alignment.matrix;
        const [productW, productH] = alignment.productSize;
        productCoords = {
          xProduct: (a * data.xPx + tx) / productW * 100,
          yProduct: (d * data.yPx + ty) / productH * 100,
        };
      }
      setAddedPoints((current) => {
        addedPointSequenceRef.current += 1;
        return [...current, {
          id: `M-${String(addedPointSequenceRef.current).padStart(2, '0')}`,
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
  /* 제품데이터 위에 올릴 때는 같은 포인트의 좌표만 제품 기준으로 바꿔 넘긴다.
     전사되지 않은 포인트는 제품데이터 밖으로 나간 것이라 시트에서 뺀다. */
  const sheetPoints = [...points, ...addedPoints].flatMap((point) => {
    if (!onProduct) return [point];
    if (point.xProduct === undefined || point.yProduct === undefined) return [];
    return [{ ...point, x: point.xProduct, y: point.yProduct }];
  });
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
      const sheetImageUrl = onProduct ? result.productImage : result.cleanImage || scan.url;
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
        };
      });

      const payload = {
        partNumber: scan.partNo,
        title: {
          managementNo: sheetTitle.managementNo,
          partName: sheetTitle.partName,
          process: sheetTitle.process,
          partNo: sheetTitle.partNo,
          material: sheetTitle.material,
          appliedDate: sheetTitle.appliedDate,
        },
        titleFonts: {
          management_no: extractFontName(sheetTitleFonts.managementNo),
          part_name: extractFontName(sheetTitleFonts.partName),
          process: extractFontName(sheetTitleFonts.process),
          part_no: extractFontName(sheetTitleFonts.partNo),
          material: extractFontName(sheetTitleFonts.material),
          applied_date: extractFontName(sheetTitleFonts.appliedDate),
        },
        titleFontSizes: {
          management_no: sheetTitleFontSizes.managementNo,
          part_name: sheetTitleFontSizes.partName,
          process: sheetTitleFontSizes.process,
          part_no: sheetTitleFontSizes.partNo,
          material: sheetTitleFontSizes.material,
          applied_date: sheetTitleFontSizes.appliedDate,
        },
        pointFontFamily: extractFontName(pointLabelFont),
        points: payloadPoints,
        annotations: payloadAnnotations,
        details: payloadDetails,
        /* 정면도 picture 를 시트 캔버스 어디에 얼마 크기로 놓을지. UI 와 같은 % 좌표로 넘긴다. */
        frontPlacement: frontLayout ? { x: frontLayout.x, y: frontLayout.y, w: frontLayout.w, h: frontLayout.h } : null,
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
  /* 보정시트에 들어가는 그림 — 정렬된 제품데이터가 있으면 그쪽을 우선한다. */
  const baseImage = onProduct
    ? result.productImage!
    : showZero && result.zeroOverlay ? result.zeroOverlay : result.cleanImage || scan.url;
  return <section className="page page--service">
    <div className="page-heading page-heading--compact"><div><span className="breadcrumb">ADC · Ajin Die Compensation</span><h2>ADC 금형 보정 시트</h2><p>흰 시트 위에 정면도와 Detail View를 독립 레이아웃으로 구성합니다.</p></div></div>
    <div className="service-grid"><div className="correction-card card">
      <div className="viewer-toolbar"><div><span className="status status--done"><Check size={13} /> 레이아웃 편집</span><b>{scan.partNo} · 보정 작업 지시도</b></div>{keySelection && keySelection.total > 0 && <div className="point-preset" title={`피크 ${keySelection.peaks} · 부호변화 ${keySelection.signChanges} · 최대최소 ${keySelection.extremes}`}><span>포인트</span><button type="button" className={visiblePointIds.size === keySelection.selected ? 'active' : ''} onClick={onKeyPointsOnly}>주요 {keySelection.selected}</button><button type="button" className={visiblePointIds.size === sheetPoints.length ? 'active' : ''} onClick={() => onAllPointsToggle(true)}>전체 {keySelection.total}</button></div>}<div className="layer-toggles"><button className={onProduct ? 'active blue' : ''} onClick={() => setUseProduct(!useProduct)} disabled={!productReady} title={productReady ? '제품데이터 위에 보정치를 올립니다' : '이 품번의 제품데이터가 등록되어 있지 않습니다'}><i /> 제품데이터</button><button className={showPoints ? 'active orange' : ''} onClick={() => setShowPoints(!showPoints)}><i /> 보정치</button><button className={showZero && zeroReady ? 'active green' : ''} onClick={() => setShowZero(!showZero)} disabled={!zeroReady} title={onProduct ? '제로라인은 스캔 좌표계 이미지라 제품데이터 위에는 겹칠 수 없습니다' : ''}><i /> 제로라인</button><button className={showAnnotations ? 'active amber' : ''} onClick={() => { setShowAnnotations(!showAnnotations); setTool('select'); setSelectedAnnotationId(null); }}><i /> 주석</button></div></div>
      <AnnotationToolbar tool={tool} setTool={(next) => { setShowAnnotations(true); setTool(next); setDetailMode(false); setLabelAreaMode(null); if (next !== 'select') setSelectedAnnotationId(null); }} hasAnnotations={annotations.length > 0} onClearAll={clearAnnotations} selectedColor={selectedColor} onColorChange={changeColor} detailMode={detailMode} onDetailMode={() => { setDetailMode(!detailMode); setLabelAreaMode(null); setTool('select'); setSelectedAnnotationId(null); }} labelAreaMode={labelAreaMode} onLabelAreaMode={(mode) => { setLabelAreaMode((current) => current === mode ? null : mode); setDetailMode(false); setAddPointMode(false); setTool('select'); setSelectedAnnotationId(null); }} addPointMode={addPointMode} onAddPointMode={() => { setAddPointMode(!addPointMode); setDetailMode(false); setLabelAreaMode(null); setTool('select'); setSelectedAnnotationId(null); setSampleError(null); }} />
      <div className="sheet-page" ref={sheetRef}><SheetTitleBlock values={sheetTitle} onChange={onSheetTitleChange} fonts={sheetTitleFonts} onFontChange={onSheetTitleFontChange} fontSizes={sheetTitleFontSizes} onFontSizeChange={onSheetTitleFontSizeChange} /><div className="sheet-stage sheet-stage--light" ref={stageRef}><SheetCanvas key={`${scan.id}-${onProduct ? 'product' : 'scan'}`} scan={scan} imageUrl={baseImage} frameWidth={frameWidth} frameHeight={frameHeight} onRegionsChange={setDetailRegions} onLayoutsChange={setSheetLayouts} points={sheetPoints} coefficient={coefficient} showPoints={showPoints} visiblePointIds={visiblePointIds} onPointToggle={onPointToggle} pointOverrides={pointOverrides} onOverrideChange={handleOverrideChange} labelFontFamily={pointLabelFont} annotations={annotations} showAnnotations={showAnnotations} annotationTool={tool} setAnnotationTool={setTool} selectedAnnotationId={selectedAnnotationId} setSelectedAnnotationId={setSelectedAnnotationId} onAnnotationCommit={commitAnnotation} onAnnotationCreate={createAnnotation} onAnnotationDelete={deleteAnnotation} detailMode={detailMode} setDetailMode={setDetailMode} labelAreaMode={labelAreaMode} setLabelAreaMode={setLabelAreaMode} addPointMode={addPointMode} onAddPointAt={addPointAt} sampling={sampling} sampleError={sampleError} addedPoints={addedPoints} onRemoveAddedPoint={removeAddedPoint} /></div></div>
      <div className="sheet-note"><ShieldCheck size={17} /><span><b>상단 표의 모든 글자를 클릭해 수정할 수 있습니다. 레이아웃은 제목 막대와 선택 핸들로 이동·조절합니다.</b>{excelError && <><br /><b className="sheet-note__error">{excelError}</b></>}</span>
        <input ref={excelInputRef} type="file" accept=".xlsx" className="visually-hidden" onChange={(e) => { setExcelFile(e.target.files?.[0] || null); setExcelError(null); }} aria-label="이어붙일 기존 보정 시트 엑셀 파일" />
        <button type="button" className="sheet-print sheet-print--ghost" onClick={() => excelInputRef.current?.click()} title="기존 보정 시트 엑셀 파일을 골라두면 그 아래에 이어붙입니다"><UploadCloud size={14} /> {excelFile ? excelFile.name : '기존 엑셀 불러오기'}</button>
        {excelFile && <button type="button" className="sheet-print__clear" onClick={() => { setExcelFile(null); if (excelInputRef.current) excelInputRef.current.value = ''; }} aria-label="선택한 엑셀 파일 취소" title="선택 취소"><X size={12} /></button>}
        <button type="button" className="sheet-print" onClick={() => void saveSheetExcel()} disabled={excelSaving}><FileSpreadsheet size={14} /> {excelSaving ? '엑셀 저장 중…' : '보정 시트 엑셀 저장'}</button>
        <button type="button" className="sheet-print" onClick={savePdf}><Printer size={14} /> 보정 시트 PDF 저장</button>
      </div>
    </div><aside className="control-panel"><div className="card coefficient-card"><div className="card-title"><div><h3>보정 계수</h3><p>편차값에 곱할 비율을 조절합니다.</p></div><span>{coefficient.toFixed(2)}×</span></div><div className="coefficient-input"><input aria-label="보정 계수 직접 입력" type="number" min="0.5" max="1.5" step="0.01" value={coefficient} onChange={(e) => { const value = e.target.valueAsNumber; if (!Number.isNaN(value)) setCoefficient(Math.max(0.5, Math.min(1.5, value))); }} /><span>×</span></div><input aria-label="보정 계수" type="range" min="0.5" max="1.5" step="0.05" value={coefficient} onChange={(e) => setCoefficient(Number(e.target.value))} /><div className="range-labels"><span>보수적 0.50</span><span>기준 1.00</span><span>적극적 1.50</span></div><div className="formula"><span>보정치</span><b>= 편차 × {coefficient.toFixed(2)} × (−1)</b></div>{overrideCount > 0 && <p className="coefficient-note">수정된 {overrideCount}개 포인트는 계수 영향을 받지 않습니다.</p>}</div><div className="card correction-summary"><h3>실제 엔진 요약</h3><div><span>보정 포인트</span><b>{visiblePointIds.size}개</b></div>{overrideCount > 0 && <div><span>수정된 포인트</span><b className="blue">{overrideCount}개</b></div>}<div><span>최대 보정량</span><b className="orange">{maxCorrection.toFixed(3)} mm</b></div><div><span>제로라인</span><b className="green">{result.stats.zeroRegions}개 영역</b></div><div><span>처리 품번</span><b>{scan.partNo}</b></div><div><span>작업자</span><input type="text" className="worker-input" value={worker} onChange={(e) => onWorkerChange(e.target.value)} placeholder="이름 입력" aria-label="작업자 이름" /></div><div><span>보정치 글꼴</span><select className="worker-input" value={pointLabelFont} onChange={(e) => setPointLabelFont(e.target.value)} aria-label="보정치 수치 글꼴 선택">{FONT_FAMILY_OPTIONS.map((option) => <option key={option.label} value={option.value} style={{ fontFamily: option.value || undefined }}>{option.label}</option>)}</select></div>{overrideCount > 0 && <button type="button" className="reset-all-overrides" onClick={() => void handleClearAllOverrides()}>모든 수정 취소</button>}</div><CorrectionHistoryPanel partNo={scan.partNo} entries={history} loading={historyLoading} pendingPointIds={pendingPointIds} deletingEntryIds={deletingEntryIds} error={historyError} onReload={loadHistory} onRestore={restoreHistoryEntry} onDelete={(entry) => void deleteHistoryEntry(entry)} /></aside></div>{folderAvailable && <Explorer />}
  </section>;
}

function CadWorkspace() {
  const [mesh, setMesh] = useState<CadMesh | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const uploadCad = async (file: File) => {
    setLoading(true); setError(null);
    const form = new FormData(); form.append('file', file, file.name);
    try {
      const response = await fetch(`${API_BASE}/api/cad`, { method: 'POST', body: form });
      const data = await response.json() as CadMesh & { error?: string };
      if (!response.ok) throw new Error(data.error || 'CAD 파일을 읽지 못했습니다.');
      setMesh(data);
    } catch (err) {
      setError(String((err as Error).message || err));
    } finally { setLoading(false); }
  };

  return <section className="page page--workspace">
    <div className="page-heading"><div><span className="breadcrumb">ADC WORKSPACE</span><h2>3D CAD 뷰어</h2><p>STL, PLY, OBJ, GLB, 3MF 파일을 로컬에서 읽어 형상을 확인합니다.</p></div></div>
    <div className="card upload-panel">
      <label className="dropzone"><input type="file" accept=".stl,.ply,.obj,.off,.glb,.gltf,.3mf" onChange={(event: ChangeEvent<HTMLInputElement>) => event.target.files?.[0] && void uploadCad(event.target.files[0])} /><span className="dropzone__icon"><Layers3 size={29} /></span><b>{loading ? 'CAD 형상을 읽는 중…' : 'CAD 파일을 선택하세요'}</b><span>STEP 파일은 현재 서버에 OCCT를 설치한 뒤 활성화할 수 있습니다.</span></label>
      {error && <p className="sheet-note__error">{error}</p>}
    </div>
    {mesh && <div className="card" style={{ minHeight: 640 }}><CadViewer active mesh={mesh} showHoles={false} /></div>}
  </section>;
}

export default function Home() {
  const [view, setView] = useState<View>('workspace'); const [scans, setScans] = useState<ScanItem[]>([]); const [activeId, setActiveId] = useState<string>(); const [collapsed, setCollapsed] = useState(false); const [backendOnline, setBackendOnline] = useState<boolean | null>(null); const [folderAvailable, setFolderAvailable] = useState(false); const [hiddenPointIdsByScan, setHiddenPointIdsByScan] = useState<Record<string, Set<string>>>({}); const [pointOverridesByScan, setPointOverridesByScan] = useState<Record<string, Record<string, number>>>({}); const [annotationsByScan, setAnnotationsByScan] = useState<Record<string, Annotation[]>>({}); const [sheetTitlesByScan, setSheetTitlesByScan] = useState<Record<string, SheetTitleValues>>({});
  const [sheetTitleFontsByScan, setSheetTitleFontsByScan] = useState<Record<string, SheetTitleFonts>>({});
  const [sheetTitleFontSizesByScan, setSheetTitleFontSizesByScan] = useState<Record<string, SheetTitleFontSizes>>({});
  /* 작업자 이름은 보정 이력에 남기는 용도라 브라우저에 저장해 다음에도 다시 입력하지 않게 한다. */
  const [worker, setWorker] = useState(() => (typeof window === 'undefined' ? '' : window.localStorage.getItem('adc-worker-name') || ''));
  useEffect(() => { if (typeof window !== 'undefined') window.localStorage.setItem('adc-worker-name', worker); }, [worker]);
  useEffect(() => { fetch(`${API_BASE}/api/health`).then((response) => response.json() as Promise<HealthResponse>).then((data) => { setBackendOnline(Boolean(data.ok)); setFolderAvailable(Boolean(data.folderAvailable)); }).catch(() => setBackendOnline(false)); }, []);
  const resolvedActiveId = activeId || scans[0]?.id;
  const activeScan = scans.find((scan) => scan.id === resolvedActiveId); const completedScan = activeScan?.result ? activeScan : scans.find((scan) => scan.result); const hasResult = Boolean(completedScan?.result);
  const hiddenPointIds = completedScan ? hiddenPointIdsByScan[completedScan.id] || new Set<string>() : new Set<string>();
  const pointOverrides = completedScan ? pointOverridesByScan[completedScan.id] || {} : {};
  const sheetTitle = completedScan ? sheetTitlesByScan[completedScan.id] || createDefaultSheetTitleValues(completedScan) : undefined;
  const sheetTitleFonts = completedScan ? sheetTitleFontsByScan[completedScan.id] || DEFAULT_TITLE_FONTS : DEFAULT_TITLE_FONTS;
  const sheetTitleFontSizes = completedScan ? sheetTitleFontSizesByScan[completedScan.id] || {} : {};
  const togglePoint = (id: string) => completedScan && setHiddenPointIdsByScan((current) => { const next = new Set(current[completedScan.id] || []); if (next.has(id)) next.delete(id); else next.add(id); return { ...current, [completedScan.id]: next }; });
  const setAllPointsVisible = (visible: boolean) => completedScan && setHiddenPointIdsByScan((current) => ({ ...current, [completedScan.id]: visible ? new Set() : new Set(completedScan.result!.points.map((point) => point.id)) }));
  /* 주요 포인트만 남긴다. 나머지는 지우지 않고 숨기기만 해서 언제든 다시 켤 수 있다. */
  const showOnlyKeyPoints = () => completedScan && setHiddenPointIdsByScan((current) => {
    const keys = new Set(completedScan.result!.keySelection?.ids || []);
    if (!keys.size) return current;
    return { ...current, [completedScan.id]: new Set(completedScan.result!.points.filter((point) => !keys.has(point.id)).map((point) => point.id)) };
  });
  const setPointOverride = (id: string, value: number | null) => completedScan && setPointOverridesByScan((current) => { const next = { ...(current[completedScan.id] || {}) }; if (value === null) delete next[id]; else next[id] = value; return { ...current, [completedScan.id]: next }; });
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
  /* 방향만 다시 계산한다. Qwen 판독은 그대로 두고 좌표만 옮겨 받는다. */
  const realign = async (flipX?: boolean, flipY?: boolean) => {
    if (!completedScan?.result) return;
    const target = completedScan;
    const form = new FormData();
    form.append('file', target.file, target.name);
    if (target.productFile) form.append('product', target.productFile, target.productFile.name);
    /* 반전을 지정하지 않으면 서버가 자동 판정과 확정 저장분을 따른다. 단순 재계산이 그 경우다. */
    if (flipX !== undefined) form.append('flipX', String(flipX));
    if (flipY !== undefined) form.append('flipY', String(flipY));
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
  const openResults = (id: string) => { setActiveId(id); setView('results'); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const selectView = (next: View) => { if (next === 'workspace' || next === 'files' || next === 'cad' || hasResult) setView(next); };
  return <main className={`app-shell ${collapsed ? 'app-shell--collapsed' : ''}`}><Sidebar view={view} setView={selectView} collapsed={collapsed} setCollapsed={setCollapsed} hasResult={hasResult} /><div className="app-main"><Header scans={scans} activeId={resolvedActiveId} setActiveId={setActiveId} />{view === 'workspace' && <Workspace scans={scans} setScans={setScans} onOpenResults={openResults} backendOnline={backendOnline} />}{view === 'results' && completedScan?.result && <Results scan={completedScan} onService={() => setView('service')} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} onAllPointsToggle={setAllPointsVisible} onRealign={realign} onConfirmAlignment={confirmAlignment} />}{view === 'service' && completedScan?.result && sheetTitle && <ServicePreview scan={completedScan} folderAvailable={folderAvailable} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} onKeyPointsOnly={showOnlyKeyPoints} onAllPointsToggle={setAllPointsVisible} pointOverrides={pointOverrides} onOverrideChange={setPointOverride} onClearAllOverrides={clearAllOverrides} annotations={annotations} setAnnotations={setAnnotations} sheetTitle={sheetTitle} onSheetTitleChange={setSheetTitleField} sheetTitleFonts={sheetTitleFonts} onSheetTitleFontChange={setSheetTitleFontField} sheetTitleFontSizes={sheetTitleFontSizes} onSheetTitleFontSizeChange={setSheetTitleFontSizeField} worker={worker} onWorkerChange={setWorker} />}{view === 'files' && <FileOrganizerPage />}{view === 'cad' && <CadWorkspace />}</div></main>;
}
