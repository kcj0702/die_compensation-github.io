"""
금형 보정 데모 — 3D 스캔 편차 히트맵에서 형상 특징과 편차를 추출하는 파이프라인.

로컬 전용. 네트워크 호출 · 모델 다운로드 없음. OpenCV 고전 기법 기반.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image as PILImage
from streamlit_image_coordinates import streamlit_image_coordinates

from pipeline import (
    apply_flip,
    build_color_mm_lut,
    clean_and_extract,
    detect_colorbar_bbox,
    detect_features,
    detect_label_boxes,
    extract_relief,
    get_body_mask,
    image_to_deviation_mm,
    render_feature_overlay,
    summarize_features,
    to_grayscale_frontview,
)

# ---------------------------------------------------------------- 페이지 설정

st.set_page_config(
    page_title="금형 보정 데모 · 형상 특징 추출",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink:#131820; --muted:#5b6675; --line:#dde3ea;
        --accent:#1d5fa8; --accent-soft:#eaf1f9;
        --ok:#1e844a; --ng:#c23b32;
      }
      .block-container { padding-top: 2.2rem; max-width: 1500px; }
      h1,h2,h3 { color:#12314f; letter-spacing:-.01em; }
      h1 { font-size:1.9rem !important; font-weight:700; }

      /* 탭 */
      .stTabs [data-baseweb="tab-list"] { gap:3px; border-bottom:1px solid var(--line); }
      .stTabs [data-baseweb="tab"] {
        background:#f4f7fa; border-radius:6px 6px 0 0; padding:9px 18px;
        font-size:.92rem; font-weight:500; color:#48566a;
      }
      .stTabs [aria-selected="true"] { background:#1d5fa8 !important; color:#fff !important; }

      /* KPI 카드 */
      .kpi { background:#fff; border:1px solid var(--line); border-left:3px solid var(--accent);
             border-radius:6px; padding:.85rem 1rem; height:100%; }
      .kpi .lab { font-size:.72rem; letter-spacing:.07em; text-transform:uppercase;
                  color:var(--muted); font-weight:600; }
      .kpi .val { font-size:1.65rem; font-weight:700; color:#12314f;
                  font-variant-numeric:tabular-nums; line-height:1.25; }
      .kpi .sub { font-size:.76rem; color:var(--muted); }
      .kpi.ok  { border-left-color:var(--ok); }
      .kpi.ng  { border-left-color:var(--ng); }
      .kpi.ng .val { color:var(--ng); }

      /* 범례 칩 */
      .chips { display:flex; flex-wrap:wrap; gap:.45rem; margin:.5rem 0 .2rem; }
      .chip { display:inline-flex; align-items:center; gap:.4rem; font-size:.78rem;
              font-weight:600; padding:.24rem .6rem; border-radius:20px;
              background:#f4f7fa; border:1px solid var(--line); color:#3c4a5c; }
      .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }

      .subtle { color:var(--muted); font-size:.88rem; }
      .step   { color:var(--muted); font-size:.86rem; border-left:3px solid var(--accent-soft);
                padding:.35rem .8rem; margin:.2rem 0 .9rem; background:#fbfcfe; }
      [data-testid="stMetricValue"] { color:#12314f; font-variant-numeric:tabular-nums; }
      footer, #MainMenu { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent.parent   # mold-correction-demo/
DATA_DIR = ROOT / "data" / "sample"             # 공용 경로 (이미지는 git 제외 대상)
TOL_DEFAULT = 0.5

FEATURE_COLORS = {           # (표시명, HTML색)
    "flat": ("평면", "#9aa7b6"),
    "bead": ("비드·엠보싱", "#d98324"),
    "near_edge": ("형상선 근처", "#2f9e44"),
    "surface": ("일반 곡면", "#4c8fd1"),
}


# ------------------------------------------------------------- 캐시 파이프라인

def _hash(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


@st.cache_data(show_spinner=False)
def _decode(_k: str, b: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def _oriented(_k: str, b: bytes, fh: bool, fv: bool) -> np.ndarray:
    return apply_flip(_decode(_k, b), fh, fv)


@st.cache_data(show_spinner=False)
def _body(_k: str, b: bytes, fh: bool, fv: bool, wt: int) -> np.ndarray:
    return get_body_mask(_oriented(_k, b, fh, fv), white_thresh=wt)


@st.cache_data(show_spinner=False)
def _extract(_k: str, b: bytes, fh: bool, fv: bool, wt: int) -> dict[str, Any]:
    img = _oriented(_k, b, fh, fv)
    body = _body(_k, b, fh, fv, wt)
    boxes_mask, boxes = detect_label_boxes(img)
    ext = clean_and_extract(img, body, boxes_mask)
    return {"clean": ext["clean"], "part": ext["part"], "boxes": boxes, "body": body}


@st.cache_data(show_spinner=False)
def _frontview(_k: str, b: bytes, fh: bool, fv: bool, wt: int) -> np.ndarray:
    return to_grayscale_frontview(_extract(_k, b, fh, fv, wt)["clean"])


@st.cache_data(show_spinner=False)
def _relief(_k: str, b: bytes, fh: bool, fv: bool, wt: int, sg: float) -> np.ndarray:
    return extract_relief(_extract(_k, b, fh, fv, wt)["clean"], sigma=sg)


@st.cache_data(show_spinner=False)
def _features(_k: str, b: bytes, fh: bool, fv: bool,
              wt: int, sg: float, mhr: int) -> dict[str, Any]:
    img = _oriented(_k, b, fh, fv)
    e = _extract(_k, b, fh, fv, wt)
    rel = _relief(_k, b, fh, fv, wt, sg)
    return detect_features(img, e["clean"], rel, e["body"],
                           min_hole_r=mhr, white_thresh=wt)


@st.cache_data(show_spinner=False)
def _deviation(_k: str, b: bytes, fh: bool, fv: bool,
               wt: int, lo: float, hi: float) -> dict[str, Any]:
    img = _oriented(_k, b, fh, fv)
    body = _body(_k, b, fh, fv, wt)
    bbox = detect_colorbar_bbox(img, body)
    if bbox is None:
        return {"mm": None, "bbox": None}
    cols, vals = build_color_mm_lut(img, bbox, lo, hi)
    return {"mm": image_to_deviation_mm(img, body, cols, vals), "bbox": bbox}


# --------------------------------------------------------------------- 유틸

def png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    return fn(*a, **kw), (time.perf_counter() - t0) * 1000


def kpi(col, label: str, value: str, sub: str = "", tone: str = ""):
    col.markdown(
        f"<div class='kpi {tone}'><div class='lab'>{label}</div>"
        f"<div class='val'>{value}</div><div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def part_on_white(img_bgr: np.ndarray, body: np.ndarray) -> np.ndarray:
    return np.where(cv2.merge([body] * 3) > 0, img_bgr, np.full_like(img_bgr, 255))


# ------------------------------------------------------------------- 사이드바

with st.sidebar:
    st.markdown("### ◈ 입력 방향")
    flip_h = st.toggle("좌우 반전", value=True,
                       help="스캔 뷰어 캡처가 좌우 반전된 경우 (라벨 숫자가 거울상)")
    flip_v = st.toggle("상하 반전", value=False)

    st.markdown("---")
    st.markdown("### ◈ 형상 추출")
    white_thresh = st.slider("배경 흰색 임계값", 200, 250, 235, 1,
                             help="이 값보다 밝은 픽셀을 배경/홀로 판정")
    sigma = st.slider("릴리프 블러 반경 σ", 5, 60, 21, 1,
                      help="작을수록 미세한 굴곡만, 클수록 넓은 형상까지")
    min_hole_r = st.slider("홀 최소 반경 (px)", 3, 40, 8, 1)

    st.markdown("---")
    st.markdown("### ◈ 편차 스케일")
    st.caption("스캔 뷰어 컬러바 눈금의 min/max 를 입력합니다.")
    c1, c2 = st.columns(2)
    mm_min = c1.number_input("min (mm)", -10.0, 0.0, -1.5, 0.1)
    mm_max = c2.number_input("max (mm)", 0.0, 10.0, 2.0, 0.1)
    tol = st.slider("공차 (± mm)", 0.1, 2.0, TOL_DEFAULT, 0.1,
                    help="이 범위를 벗어나면 NG 로 판정")

    st.markdown("---")
    st.markdown("### ◈ 표시")
    show_circles = st.checkbox("원형 홀", True)
    show_rects = st.checkbox("사각 홀", True)
    show_others = st.checkbox("기타 홀", True)
    show_lines = st.checkbox("형상선 / 비드", True)
    show_flat = st.checkbox("평면 영역", True)
    overlay_bg = st.radio("오버레이 배경",
                          ["정면도 (회색)", "부품 (컬러)", "원본 (컬러바 포함)"], index=0)


# ---------------------------------------------------------------------- 헤더

st.markdown("# 금형 보정 데모 · 형상 특징 추출")
st.markdown(
    "<span class='subtle'>3D 스캔 편차 히트맵 한 장에서 <b>부품 형상 · 형상 특징 · 편차값</b>을 "
    "추출하고, 보정치 산출 직전 단계의 통합 데이터를 생성합니다. "
    "전 과정 로컬 실행 — 네트워크 · 클라우드 · 모델 다운로드 없음.</span>",
    unsafe_allow_html=True,
)
st.markdown("")

# ---------------------------------------------------------------------- 입력

cu, cs = st.columns([3, 1])
with cu:
    up = st.file_uploader("스캔 이미지 업로드", type=["png", "jpg", "jpeg"],
                          label_visibility="collapsed")
samples = sorted(DATA_DIR.glob("*.png")) + sorted(DATA_DIR.glob("*.jpg"))
with cs:
    pick = st.selectbox("샘플", ["— 샘플 선택 —"] + [p.name for p in samples],
                        label_visibility="collapsed", disabled=not samples)

img_bytes, input_name = None, ""
if up is not None:
    img_bytes, input_name = up.getvalue(), up.name
elif pick and not pick.startswith("—"):
    p = DATA_DIR / pick
    img_bytes, input_name = p.read_bytes(), p.name

if img_bytes is None:
    st.info("스캔 이미지를 업로드하거나 우측에서 샘플을 선택하세요.")
    st.stop()

key = _hash(img_bytes)
raw = _decode(key, img_bytes)
if raw is None:
    st.error("이미지를 읽을 수 없습니다. PNG / JPG 만 지원합니다.")
    st.stop()

# ------------------------------------------------------------ 파이프라인 실행

P = (key, img_bytes, flip_h, flip_v)
with st.spinner("파이프라인 실행 중…"):
    ex, ms_ext = timed(_extract, *P, white_thresh)
    fv, ms_fv = timed(_frontview, *P, white_thresh)
    rel, ms_rel = timed(_relief, *P, white_thresh, sigma)
    feats, ms_feat = timed(_features, *P, white_thresh, sigma, min_hole_r)
    dev, ms_dev = timed(_deviation, *P, white_thresh, mm_min, mm_max)

body = ex["body"]
summ = summarize_features(feats, body)
H, W = raw.shape[:2]

# 편차 통계는 편차맵에서 직접 산출 (부품 영역 픽셀 기준)
if dev["mm"] is not None:
    _v = dev["mm"].copy()
    _v[body == 0] = np.nan
    valid = _v[~np.isnan(_v)]
    dev_lo, dev_hi = float(valid.min()), float(valid.max())
    ng_pct = float((np.abs(valid) > tol).mean() * 100)
else:
    valid = np.array([])
    dev_lo = dev_hi = None
    ng_pct = 0.0

# --------------------------------------------------------------- KPI 요약

k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "입력", f"{W}×{H}", input_name[:26])
kpi(k2, "검출 홀", f"{summ['n_circles'] + summ['n_rects'] + summ['n_others']}",
    f"원 {summ['n_circles']} · 사각 {summ['n_rects']} · 기타 {summ['n_others']}")
kpi(k3, "형상선", f"{summ['n_lines']}", f"평면 비율 {summ['flat_ratio']}%")
kpi(k4, "편차 범위",
    f"{dev_lo:+.2f} ~ {dev_hi:+.2f}" if dev_lo is not None else "—", "mm")
kpi(k5, "공차 초과 면적", f"{ng_pct:.0f}%", f"공차 ±{tol} mm 기준",
    tone="ng" if ng_pct > 30 else "ok")

st.markdown("")

# ------------------------------------------------------------------- 탭

tabs = st.tabs([
    "① 형상 추출", "② 정면도", "③ 릴리프", "④ 형상 특징 검출", "⑤ 편차 조회",
])

# ── ① 형상 추출 ────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(
        "<div class='step'>흰 배경 · 컬러바 · 지시선 · 라벨박스를 제거하고 부품만 남깁니다. "
        "라벨이 있던 자리는 inpaint 로 주변 형상에서 복원합니다.</div>",
        unsafe_allow_html=True)
    st.image(ex["part"], channels="BGR", width='stretch')
    c1, c2, c3 = st.columns([1, 1, 3])
    c1.download_button("PNG 저장", png_bytes(ex["part"]),
                       f"01_part_{input_name}.png", "image/png", width='stretch')
    c2.metric("제거한 라벨박스", len(ex["boxes"]))
    nb = ex["boxes"]
    c3.caption(f"처리 {ms_ext:.0f} ms · "
               f"red {sum(1 for b in nb if b['color']=='red')} · "
               f"mint {sum(1 for b in nb if b['color']=='mint')} · "
               f"gray {sum(1 for b in nb if b['color']=='gray')}")

# ── ② 정면도 ──────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "<div class='step'>편차를 나타내던 색을 벗겨내면 렌더에 깔려 있던 <b>실제 음영</b>이 드러납니다. "
        "비드 · 엠보싱 · 굴곡이 실사진처럼 보이는 정면도가 됩니다.</div>",
        unsafe_allow_html=True)
    st.image(part_on_white(fv, body), channels="BGR", width='stretch')
    c1, c2 = st.columns([1, 4])
    c1.download_button("PNG 저장", png_bytes(part_on_white(fv, body)),
                       f"02_frontview_{input_name}.png", "image/png",
                       width='stretch')
    c2.caption(f"처리 {ms_fv:.0f} ms · 히스토그램 스트레칭 적용")

# ── ③ 릴리프 ──────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown(
        "<div class='step'>하이패스 필터로 전체 곡률(저주파)을 제거하고 미세 굴곡만 남깁니다. "
        "보정량에 영향을 주는 <b>비드 · 필릿 · 형상선</b>이 드러납니다.</div>",
        unsafe_allow_html=True)
    st.image(part_on_white(rel, body), channels="BGR", width='stretch')
    c1, c2 = st.columns([1, 4])
    c1.download_button("PNG 저장", png_bytes(part_on_white(rel, body)),
                       f"03_relief_{input_name}.png", "image/png",
                       width='stretch')
    c2.caption(f"처리 {ms_rel:.0f} ms · σ={sigma}")

# ── ④ 형상 특징 검출 ──────────────────────────────────────────────────
with tabs[3]:
    st.markdown(
        "<div class='step'>홀은 <b>부품 안쪽 흰 영역</b>의 컨투어를 원형도와 bbox 채움도로 분류하고, "
        "형상선은 릴리프 엣지에서 추출합니다.</div>", unsafe_allow_html=True)

    show = dict(circles=show_circles, rects=show_rects, others=show_others,
                lines=show_lines, flat=show_flat)
    if overlay_bg.startswith("정면도"):
        base = part_on_white(fv, body)
    elif overlay_bg.startswith("부품"):
        base = ex["part"]
    else:
        base = ex["clean"]
    overlay, ms_ov = timed(render_feature_overlay, base, feats, show)

    cp = st.session_state.get("pick_xy")
    if cp and 0 <= cp[0] < W and 0 <= cp[1] < H:
        for col, sz, th in [((0, 0, 0), 26, 4), ((0, 255, 255), 22, 2)]:
            cv2.drawMarker(overlay, cp, col, cv2.MARKER_CROSS, sz, th, cv2.LINE_AA)

    st.image(overlay, channels="BGR", width='stretch')
    st.markdown(
        "<div class='chips'>"
        "<span class='chip'><span class='dot' style='background:#2878e6'></span>원형 홀</span>"
        "<span class='chip'><span class='dot' style='background:#ff8c00'></span>사각 홀</span>"
        "<span class='chip'><span class='dot' style='background:#c83cc8'></span>기타 홀</span>"
        "<span class='chip'><span class='dot' style='background:#3cc83c'></span>형상선 · 비드</span>"
        "<span class='chip'><span class='dot' style='background:#aaaaaa'></span>평면 영역</span>"
        "</div>", unsafe_allow_html=True)

    m = st.columns(5)
    m[0].metric("원형 홀", summ["n_circles"])
    m[1].metric("사각 홀", summ["n_rects"])
    m[2].metric("기타 홀", summ["n_others"])
    m[3].metric("형상선", summ["n_lines"])
    m[4].metric("평면 비율", f"{summ['flat_ratio']} %")

    c1, c2 = st.columns([1, 4])
    c1.download_button("PNG 저장", png_bytes(overlay),
                       f"04_features_{input_name}.png", "image/png",
                       width='stretch')
    c2.caption(f"검출 {ms_feat:.0f} ms · 렌더 {ms_ov:.0f} ms")

    det: list[dict[str, Any]] = []
    for i, c in enumerate(feats["circles"], 1):
        det.append(dict(no=i, kind="circle", x=c["cx"], y=c["cy"],
                        size=f"r={c['r']}", area=int(np.pi * c["r"] ** 2)))
    for i, r in enumerate(feats["rects"], 1):
        det.append(dict(no=i, kind="rect", x=r["x"] + r["w"] // 2, y=r["y"] + r["h"] // 2,
                        size=f"{r['w']}×{r['h']}", area=r["w"] * r["h"]))
    for i, o in enumerate(feats.get("others", []), 1):
        det.append(dict(no=i, kind="other", x=o["x"] + o["w"] // 2, y=o["y"] + o["h"] // 2,
                        size=f"{o['w']}×{o['h']}", area=o["w"] * o["h"]))
    if det:
        st.dataframe(det, width='stretch', hide_index=True)

# ── ⑤ 편차 조회 ──────────────────────────────────────────────────────
with tabs[4]:
    if dev["mm"] is None:
        st.warning("컬러바를 자동 검출하지 못했습니다. 배경 임계값을 조정해 보세요.")
    else:
        st.markdown(
            "<div class='step'>컬러바를 자동 검출해 <b>색 → 편차(mm)</b> 대응표를 만들고, "
            "부품의 모든 픽셀을 Lab 색공간에서 최근접 매칭합니다. "
            "이미지를 클릭하면 그 지점의 편차가 표시되고 좌표는 ④ 탭에 공유됩니다.</div>",
            unsafe_allow_html=True)

        mm_arr = dev["mm"].copy()
        mm_arr[body == 0] = np.nan
        part_bgr = part_on_white(ex["clean"], body)

        DW = 1150
        sc = DW / W
        disp = cv2.resize(part_bgr, (DW, int(H * sc)), interpolation=cv2.INTER_AREA)
        cp = st.session_state.get("pick_xy")
        if cp:
            d = (int(cp[0] * sc), int(cp[1] * sc))
            cv2.circle(disp, d, 13, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(disp, d, 13, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(disp, d, (0, 0, 0), cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)
            cv2.drawMarker(disp, d, (0, 255, 255), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)

        click = streamlit_image_coordinates(
            PILImage.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)),
            key="devclick", width=DW)

        if click is not None:
            cx = max(0, min(W - 1, int(round(click["x"] / sc))))
            cy = max(0, min(H - 1, int(round(click["y"] / sc))))
            if st.session_state.get("pick_xy") != (cx, cy):
                st.session_state["pick_xy"] = (cx, cy)
                v = mm_arr[cy, cx]
                st.session_state["pick_mm"] = None if np.isnan(v) else float(v)
                st.rerun()

        cp = st.session_state.get("pick_xy")
        cm = st.session_state.get("pick_mm")
        q = st.columns([1, 1, 1, 1, 2])
        kpi(q[0], "좌표 (x, y)", f"({cp[0]}, {cp[1]})" if cp else "—", "px")
        kpi(q[1], "편차", f"{cm:+.2f}" if cm is not None else "—", "mm")
        if cm is None:
            kpi(q[2], "판정", "—", f"공차 ±{tol}")
        else:
            ng = abs(cm) > tol
            kpi(q[2], "판정", "NG" if ng else "OK", f"공차 ±{tol} mm",
                tone="ng" if ng else "ok")
        if cp:
            from pipeline import classify_point_feature
            fk = classify_point_feature(
                cp[0], cp[1], feats, cv2.cvtColor(rel, cv2.COLOR_BGR2GRAY), body)
            kpi(q[3], "형상 특징", FEATURE_COLORS.get(fk, (fk, ""))[0], fk)
        else:
            kpi(q[3], "형상 특징", "—", "")
        with q[4]:
            st.caption(f"컬러바 {dev['bbox']} · 범위 {mm_min:+.1f}~{mm_max:+.1f} mm · "
                       f"편차맵 {ms_dev:.0f} ms")
            if st.button("선택 해제", width='stretch'):
                st.session_state.pop("pick_xy", None)
                st.session_state.pop("pick_mm", None)
                st.rerun()

        with st.expander("마우스 hover 로 실시간 편차 확인"):
            fig = go.Figure()
            fig.add_layout_image(dict(
                source=PILImage.fromarray(cv2.cvtColor(part_bgr, cv2.COLOR_BGR2RGB)),
                xref="x", yref="y", x=0, y=0, sizex=W, sizey=H,
                xanchor="left", yanchor="top", sizing="stretch", layer="below"))
            fig.add_trace(go.Heatmap(
                z=mm_arr, opacity=0.0, hoverongaps=False, showscale=False,
                colorscale="Gray",
                hovertemplate="x=%{x} · y=%{y}<br>편차 %{z:+.2f} mm<extra></extra>"))
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=540,
                              dragmode="pan", plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(visible=False, range=[0, W], constrain="domain"),
                              yaxis=dict(visible=False, range=[H, 0],
                                         scaleanchor="x", scaleratio=1))
            st.plotly_chart(fig, width='stretch', key="hoverfig")

# --------------------------------------------------------------------- 푸터

st.markdown("---")
st.caption(
    "로컬 전용 실행 · 네트워크/클라우드 호출 없음 · OpenCV 고전 기법 기반  |  "
    "데모 범위: 형상 특징 · 편차 추출까지  |  "
    "다음 단계: 편차 × 형상 특징 결합 · 0-라인 자동 생성 · CAD 정합 · 보정치 산출"
)
