/**
 * 작업 내용을 이 PC 에 남긴다.
 *
 * [왜 필요한가]
 * 작업자가 고친 보정값, 숨긴 포인트, 보정 계수, 공정 구역, 주석이 전부
 * React 상태에만 있었다. 새로고침 한 번이면 다 날아간다. 한 시간 걸려
 * 다듬은 시트를 잃는 도구는 현장에서 못 쓴다.
 *
 * [어디에 남기나]
 * localStorage 다. 브라우저 안이라 밖으로 나가지 않는다 — "모든 처리는
 * 이 PC 안에서" 라는 이 프로젝트의 전제와 맞는다. 그리고 파일(.json)로
 * 내보내고 불러올 수 있게 했다. 보관하거나 다른 사람에게 넘길 때 쓴다.
 *
 * [담지 않는 것]
 * 스캔 이미지와 CAD 자체는 담지 않는다. 수백 MB 라 브라우저 저장소에
 * 들어가지도 않고, 원본은 이미 파일로 있다. 여기 남기는 것은 **사람이
 * 판단한 내용**뿐이다.
 */

const KEY = 'adc.session.v1';

export type SessionSnapshot = {
  version: 1;
  savedAt: string;
  /* 스캔별 — 아이디가 아니라 품번으로 묶는다. 파일을 다시 올리면
     아이디는 새로 생기지만 품번은 그대로다. */
  byPart: Record<string, {
    coefficient?: number;
    overrides?: Record<string, number>;
    hidden?: string[];
  }>;
  /* CAD 별 — 파일 이름으로 묶는다. */
  byCad: Record<string, {
    notes?: unknown[];
    regions?: unknown[];
  }>;
};

export function emptySession(): SessionSnapshot {
  return { version: 1, savedAt: new Date().toISOString(), byPart: {}, byCad: {} };
}

/** 저장. 브라우저가 막아 두거나 용량이 차면 조용히 넘어간다 —
 *  저장이 안 된다고 작업이 멈추면 안 된다. */
export function saveSession(snapshot: SessionSnapshot): boolean {
  try {
    localStorage.setItem(KEY, JSON.stringify(
      { ...snapshot, savedAt: new Date().toISOString() }));
    return true;
  } catch {
    return false;
  }
}

export function loadSession(): SessionSnapshot | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as SessionSnapshot;
    if (parsed?.version !== 1) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function clearSession() {
  try { localStorage.removeItem(KEY); } catch { /* 지우기 실패는 무시 */ }
}

/** 파일로 내려받는다. */
export function downloadSession(snapshot: SessionSnapshot, name = 'ADC_작업내용') {
  const blob = new Blob([JSON.stringify(snapshot, null, 2)],
    { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${name}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

/** 파일에서 읽는다. 형식이 다르면 null 을 준다. */
export async function readSessionFile(file: File): Promise<SessionSnapshot | null> {
  try {
    const parsed = JSON.parse(await file.text()) as SessionSnapshot;
    if (parsed?.version !== 1 || !parsed.byPart) return null;
    return parsed;
  } catch {
    return null;
  }
}
