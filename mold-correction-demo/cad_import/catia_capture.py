"""CATIA 뷰포트를 캡처해 "제품데이터 이미지" 한 장으로 정리한다.

[왜 이 모듈이 있는가]
사용자는 "카티아가 화면에 그려 주는 그대로의 셰이딩" 을 원한다 — VTK 로
우리가 직접 그린 회색 재질이 아니라, CATIA 가 부품마다 지정한 실제 색상·
재질이 나온 진짜 렌더. 그 그림은 CATIA COM 의 `Viewer3D.CaptureToFile`
로만 얻을 수 있으므로 여기서 CATIA 를 조종한다.

[정합은 이 모듈이 하지 않는다 — 왜]
처음엔 CATIA 가 잡은 화면을 우리 mesh 좌표로 역산해 스캔 프레임에 직접
아핀 워프하려 했다. 수학은 무작위 좌표로 검증됐지만(오차 <0.5px), 그
검증 자체가 "CATIA 화면의 왼쪽이 mesh u 최솟값" 이라는 가정을 스스로
세우고 스스로 확인하는 순환 논리였다 — CATIA 카메라가 실제로 어느
손대칭(handedness) 규칙을 쓰는지는 API 문서만으로 확신할 수 없고, 실측
결과 좌우가 실제로 뒤집혀 나온 사례가 있었다.

그래서 정합은 이 모듈이 떠맡지 않는다. 대신 이 모듈은 "부품만 깨끗하게
잘라낸 이미지" 한 장만 만들고, 그 이미지를 실제 촬영된 제품데이터 PNG와
완전히 동일하게 취급해 `product_alignment.estimate_alignment` 에 넘긴다.
그 파이프라인은 좌우반전 4가지를 모두 시도해 스캔 실루엣·구멍 패턴과
실제로 가장 잘 겹치는 것을 픽셀 단위로 골라주는, 이미 검증되어 있는
코드다 — 우리가 3D 카메라 수식만으로 장담하려다 여러 번 틀렸던 지점을,
실측 비교로 대체한다.

[흐름]
    1. CATPart(또는 STEP) 를 CATIA 로 연다.
    2. 배경을 흰색으로, 카메라를 fit.axis/sign/swap 이 정한 6방향 표준
       뷰 중 하나로 돌린다(정밀한 각도·스케일은 안 맞춘다 — 2번에서
       어차피 다시 맞춘다).
    3. Reframe 뒤 CaptureToFile 로 JPEG 저장.
    4. 배경 아닌 픽셀 중 "몸통이 두꺼운 덩어리" 만 남기고 나머지(축
       표시·지시선·주석)는 지운다 — 모폴로지 오프닝으로 얇은 돌출부를
       걸러내 판단하되, 실제로 남기는 픽셀은 원본에서 그대로 가져온다.

[캐시]
원본 캡처는 (파일, axis, sign, swap) 별로 캐시한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .overlay import ViewFit, _fit_axes


_AXIS_UNIT = {0: np.array([1.0, 0.0, 0.0]),
              1: np.array([0.0, 1.0, 0.0]),
              2: np.array([0.0, 0.0, 1.0])}


def _view_directions(axis: int, sign: int, swap: bool) -> tuple[list[float], list[float]]:
    """CATIA Viewpoint3D 에 넣을 (SightDirection, UpDirection).

    shaded_render._render_canonical 의 VTK 카메라 설정과 정확히 같은 규칙:
    시선은 axis/sign 축을 따라가고(카메라 -> 원점 방향), up 은 `_fit_axes`
    가 고른 v_axis 의 반대 방향(= v 증가가 화면 아래로 가도록, pixel_y
    증가 관례와 맞춘다).
    """
    class _Probe:
        pass
    probe = _Probe()
    probe.axis, probe.swap = axis, swap
    _u_axis, v_axis = _fit_axes(probe)
    sight = (_AXIS_UNIT[axis] * float(sign)).tolist()
    up = (-_AXIS_UNIT[v_axis]).tolist()
    return sight, up


def _cache_path(source: Path, axis: int, sign: int, swap: bool, cache_dir: Path) -> Path:
    swap_tag = 's1' if swap else 's0'
    return cache_dir / f"{source.stem}__catia_axis{axis}_sign{'p' if sign > 0 else 'm'}_{swap_tag}.jpg"


# HybridBody/AnnotationSet 이름에 이 중 하나라도 들어 있으면 화면에서
# 감춘다. 대소문자 구분 없이 부분 일치로 본다 — 실측 67XX6 이 이 관례를
# 썼다("#Annotation for Informations", "#External Geometry", "#Standards
# and Informations", "External References"). 실제 형상(Body)에는 보통
# 이런 단어가 안 들어가므로 오탐 위험이 낮다.
_HIDE_NAME_KEYWORDS = ("annotation", "external", "standard", "reference", "information")


def _hide_non_geometry_elements(doc: Any) -> None:
    """지시선·주석·기준축·외부 참조 형상을 캡처 전에 감춘다.

    실패해도(문서가 CATPart 가 아니거나, 예상한 컬렉션이 없거나) 캡처
    자체는 계속 진행해야 하므로 전체를 try/except 로 감싼다 — 여기서
    무엇을 못 감췄는지는 화면 후처리(_isolate_part 의 색조 필터)가
    2차 방어선으로 남아 있다.
    """
    try:
        part = doc.Part
    except Exception:
        return  # CATPart 가 아니거나(예: 다른 문서 타입) 접근 불가.

    selection = doc.Selection
    hidden: list[str] = []

    def _hide_collection(collection: Any, *, filter_by_name: bool) -> None:
        try:
            count = collection.Count
        except Exception:
            return
        for i in range(1, count + 1):
            try:
                item = collection.Item(i)
            except Exception:
                continue
            if filter_by_name:
                name = str(getattr(item, "Name", "") or "")
                if not any(keyword in name.lower() for keyword in _HIDE_NAME_KEYWORDS):
                    continue
            try:
                selection.Clear()
                selection.Add(item)
                selection.VisProperties.SetShow(1)  # 1 = catVisPropertiesNoShow
                hidden.append(str(getattr(item, "Name", i)))
            except Exception:
                pass

    try:
        _hide_collection(part.HybridBodies, filter_by_name=True)
        _hide_collection(part.AxisSystems, filter_by_name=False)
        _hide_collection(part.AnnotationSets, filter_by_name=False)
    except Exception as exc:
        print(f"[catia_capture] hide non-geometry elements failed (continuing): {exc}")
    finally:
        try:
            selection.Clear()
        except Exception:
            pass
    if hidden:
        print(f"[catia_capture] hidden: {hidden}")


def capture_view(source: str | Path, axis: int, sign: int, cache_dir: str | Path,
                  swap: bool = False, prefer_native: bool = True) -> Path:
    """CATIA 로 부품을 열어 지정된 축 방향에서 셰이딩 스크린샷을 찍는다.

    반환된 이미지는 아직 스캔 프레임과 정합되지 않은 "원본 캡처" 다 —
    정합까지 하려면 `capture_registered_view` 를 쓴다.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    open_path = source_path
    if prefer_native and source_path.suffix.lower() != ".catpart":
        sibling = source_path.with_suffix(".CATPart")
        if not sibling.is_file():
            sibling = source_path.with_suffix(".catpart")
        if sibling.is_file():
            open_path = sibling

    cache_target = _cache_path(source_path, axis, sign, swap, cache_root)
    if cache_target.is_file() and cache_target.stat().st_mtime >= open_path.stat().st_mtime:
        return cache_target

    sight_dir, up_dir = _view_directions(axis, sign, swap)

    try:
        import pythoncom  # noqa: WPS433
        import win32com.client
    except ImportError as exc:
        raise ValueError("CATIA 캡처에 pywin32 가 필요합니다: " + str(exc)) from exc

    pythoncom.CoInitialize()
    try:
        try:
            catia = win32com.client.Dispatch("CATIA.Application")
        except pythoncom.com_error as exc:
            raise ValueError(
                f"CATIA 를 실행할 수 없습니다. CATIA 설치·라이선스를 확인하세요. 원인: {exc}"
            ) from exc

        previous_visible = None
        try:
            previous_visible = catia.Visible
        except Exception:
            pass
        try:
            catia.Visible = True
        except Exception:
            pass

        doc = None
        active_viewer = None
        try:
            try:
                doc = catia.Documents.Open(str(open_path.resolve()))
            except pythoncom.com_error as exc:
                raise ValueError(f"CATIA 가 {open_path.name} 을 열지 못했습니다: {exc}") from exc

            try:
                active_viewer = catia.ActiveWindow.ActiveViewer
            except Exception as exc:
                raise ValueError(f"활성 뷰어에 접근하지 못했습니다: {exc}") from exc

            # 지시선/치수/주석·기준축·외부 참조 형상은 화면에 같이 찍혀 나오면
            # 안 된다(실측 67XX6: 분홍 화살표가 부품에 겹쳐 나왔다). CATIA
            # API 로 이런 요소를 감춘다 — 이미지 후처리로 지우는 것보다
            # 확실하다(색이 우연히 재질색과 비슷하면 후처리는 못 잡는다).
            # 회사 표준 CATPart 는 HybridBody 이름에 "Annotation"/"External"/
            # "Standard"/"Reference" 가 들어가는 관례가 있어(#Annotation for
            # Informations, #External Geometry 등) 이름 키워드로 찾는다 —
            # 정확한 이름을 하드코딩하면 다른 부품에서 이름이 다를 때 못 찾는다.
            _hide_non_geometry_elements(doc)

            # [버그였던 부분] GetBackgroundColor() 는 이 CATIA 버전에서 항상
            # 예외를 던진다. 예전 코드는 이 호출과 PutBackgroundColor 를 같은
            # try 블록에 넣어서, 예외가 나면 PutBackgroundColor 까지 통째로
            # 건너뛰어 배경이 계속 CATIA 기본 남색 그라디언트로 남았다.
            try:
                active_viewer.PutBackgroundColor([1.0, 1.0, 1.0])
            except Exception as exc:
                print(f"[catia_capture] background set failed (continuing): {exc}")

            try:
                vp = active_viewer.Viewpoint3D
                vp.PutSightDirection(sight_dir)
                vp.PutUpDirection(up_dir)
                active_viewer.Viewpoint3D = vp
            except Exception as exc:
                print(f"[catia_capture] viewpoint set failed (continuing): {exc}")

            try:
                active_viewer.Reframe()
            except Exception as exc:
                print(f"[catia_capture] reframe failed (continuing): {exc}")

            active_viewer.CaptureToFile(2, str(cache_target.resolve()))  # 2 = JPEG
        finally:
            if doc is not None:
                try:
                    doc.Close()
                except Exception:
                    pass
            if previous_visible is not None:
                try:
                    catia.Visible = previous_visible
                except Exception:
                    pass
    finally:
        pythoncom.CoUninitialize()

    if not cache_target.is_file() or cache_target.stat().st_size == 0:
        raise ValueError(f"CATIA 캡처 파일이 생성되지 않았습니다: {cache_target.name}")
    return cache_target


def _isolate_part(captured: np.ndarray) -> tuple[np.ndarray, int, int]:
    """배경 흰색이 아닌 픽셀 중 부품 본체만 남기고 나머지(축 표시·나침반
    등 화면 UI 요소)는 흰색으로 지운 뒤, 그 영역의 bbox 로 딱 맞게 잘라낸다.

    모폴로지 오프닝으로 "몸통이 두꺼운 덩어리"의 bbox 를 찾는다 — 코너의
    작은 축 표시 아이콘처럼 가늘거나 작은 UI 요소는 오프닝에 지워져 bbox
    후보에서 빠진다. 지시선·주석 자체는 `_hide_non_geometry_elements` 가
    캡처 전에 CATIA 쪽에서 이미 꺼 두므로 여기서는 다시 다루지 않는다.

    [버그였던 부분] 한때 bbox 안에서 CATIA 재질색과 색조(hue)가 다른
    픽셀을 추가로 지우는 안전망을 뒀었다. JPEG 압축이 평평한 색 위에도
    자잘한 색조 잡음을 남기는데, 그 잡음이 안전망에 걸려 표면 전체가
    "홀로그램처럼" 얼룩덜룩하게 지워졌다. 근본 원인(주석)은 CATIA 쪽에서
    막았으니 이 안전망은 득보다 실이 커 제거했다.

    Returns:
        (cropped_bgr, left, top) — left/top 은 원본 캡처 안에서의 crop 위치
        (지금은 쓰지 않지만 디버깅에 유용해 남겨 둔다).
    """
    gray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
    foreground = (gray < 245).astype(np.uint8)

    long_side = max(captured.shape[:2])
    open_kernel_size = max(5, long_side // 150)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    opened = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, open_kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_labels <= 1:
        # 오프닝이 전부 지워버렸으면(너무 작은 부품 등) 오프닝 없이 원래대로.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        if n_labels <= 1:
            return captured, 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    left, top, w, h = (stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP],
                       stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT])

    part_mask = np.zeros(foreground.shape, dtype=bool)
    part_mask[top:top + h, left:left + w] = foreground[top:top + h, left:left + w] > 0
    cleaned = captured.copy()
    cleaned[~part_mask] = 255

    # 부품에 딱 맞춰 자르면 흰 여백이 하나도 없어 "화면에 꽉 찬" 답답한
    # 느낌을 준다(실측 사용자 피드백) — 실제 촬영된 제품데이터 사진은
    # 보통 부품 둘레에 여백이 있다. bbox 바깥의 이미 흰 배경 영역을 조금
    # 더 끌어와 여백처럼 보이게 한다(원본 캡처 범위 안에서만, 새로 칠하지
    # 않고 그대로 있던 배경을 쓴다).
    margin_h, margin_w = captured.shape[0], captured.shape[1]
    margin = int(round(max(w, h) * 0.06))
    exp_left = max(0, left - margin)
    exp_top = max(0, top - margin)
    exp_right = min(margin_w, left + w + margin)
    exp_bottom = min(margin_h, top + h + margin)
    return cleaned[exp_top:exp_bottom, exp_left:exp_right], exp_left, exp_top


def capture_product_image(source: str | Path, fit: ViewFit, cache_dir: str | Path) -> np.ndarray:
    """CATIA 셰이딩 캡처를 정리해 "제품데이터 이미지" 한 장으로 돌려준다.

    스캔 프레임에 맞춰 배치/정합하지는 않는다 — 그 일은 이 결과를
    `product_alignment.estimate_alignment` 에 실제 촬영된 제품데이터
    PNG 와 똑같이 넘겨서 시키는 게 낫다. 그 파이프라인은 좌우반전 4가지를
    다 시도해 스캔 실루엣과 가장 잘 겹치는 것을 실측으로 고르는, 이미
    검증된 코드다 — 우리가 3D 카메라 수식만으로 좌우 방향(handedness)을
    장담하려다 여러 번 틀렸던 자리를, 대신 픽셀 겹침을 실제로 재는 방식
    으로 대체한다.

    Args:
        source:    CATPart/STEP 원본.
        fit:       overlay.fit_view 결과 — axis/sign/swap 만 쓴다(뷰 방향
                   선택용). flip_u/flip_v/angle/mm_per_px/origin 은 여기서
                   안 쓴다 — 그건 2D 정합 단계가 다시 계산한다.
        cache_dir: 원본 캡처 JPEG 캐시 폴더.

    Returns:
        (h, w, 3) uint8 BGR. 배경 흰색, 부품은 CATIA 실제 셰이딩, 자기
        나름의 tight crop 크기(스캔 크기와 무관).
    """
    axis = int(fit.axis)
    sign = 1 if fit.sign > 0 else -1
    swap = bool(getattr(fit, "swap", False))

    capture_path = capture_view(source, axis, sign, cache_dir, swap=swap)
    captured = cv2.imdecode(np.fromfile(str(capture_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if captured is None:
        raise ValueError(f"CATIA 캡처 이미지를 읽지 못했습니다: {capture_path.name}")

    cleaned, _left, _top = _isolate_part(captured)
    if cleaned.shape[0] < 2 or cleaned.shape[1] < 2:
        raise ValueError("CATIA 캡처에서 부품 영역을 찾지 못했습니다(배경만 남음).")
    return cleaned


__all__ = ["capture_view", "capture_product_image"]
