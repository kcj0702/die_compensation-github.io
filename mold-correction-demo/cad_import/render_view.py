"""메시를 스캔과 같은 방향·프레임으로 라인 아트 PNG 로 렌더한다.

[왜 이 모듈이 있는가]
`overlay.fit_view` 가 이미 스캔 방향과 맞는 뷰(ViewFit)를 찾아 준다.
그 뷰로 정점을 스캔 픽셀 좌표에 그대로 투영하면(=`overlay.to_pixels`),
정렬 행렬 없이도 **"제품 이미지가 곧 스캔 프레임"** 이 된다 —
alignment 는 자연히 identity 다.

그리는 방식은 도면 스타일이다: 실루엣을 연한 회색으로 채우고, 외곽선을
굵게, 내부 구멍 윤곽을 얇은 검정선으로 얹는다. 이 정도면 보정 시트의
"깨끗한 파트 이미지" 자리에 그대로 쓸 수 있다.

[뒷면 컬링]
판재 부품은 앞뒤 두 껍질과 그 사이 리브·엠보스·스팟용접 같은 내부
구조가 함께 들어 있다. 삼각형을 전부 채우면 내부 리브의 실루엣이
표면 위에 겹쳐 나와 "단면도" 처럼 보인다. 카메라 방향과 반대인 면
(back-facing) 을 지워 앞면만 남기면, 실제로 눈에 보이는 그림이 된다.
법선은 3D 좌표에서 계산하되 판별에 쓰는 축은 fit 이 고른 축이다.
"""
from __future__ import annotations

import cv2
import numpy as np

from .overlay import ViewFit, to_pixels, _fit_axes


# 배경(순수 흰색)과 부품 영역이 그림에서 구분되지 않으면 라인만 떠 보여서
# 도면이 아니라 낙서처럼 읽힌다. 아주 살짝 회색을 채워 두면 채워짐이
# 인지된다.
BODY_FILL_BGR = (245, 245, 245)
BACKGROUND_BGR = (255, 255, 255)
LINE_BGR = (30, 30, 30)


def _front_facing_mask(vertices: np.ndarray, faces: np.ndarray,
                        fit: ViewFit) -> np.ndarray:
    """카메라 방향으로 향한 삼각형만 True 인 bool 배열.

    fit.axis 가 보는 축이고, fit.sign 이 그 축의 양/음 방향을 정한다.
    삼각형 법선의 그 축 성분이 카메라 반대 방향이면 앞면 — 뒷껍질과
    옆면·리브 뒷면을 걸러 낸다.
    """
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    edge1 = v[f[:, 1]] - v[f[:, 0]]
    edge2 = v[f[:, 2]] - v[f[:, 0]]
    # cross 로 삼각형 법선(정규화 불필요 — 부호만 본다).
    normals = np.cross(edge1, edge2)
    # 카메라는 fit.axis 축의 +/- 방향에서 부품을 바라본다. sign>0 이면
    # +축에서 본 것 — 카메라 방향은 -axis, 그러니 법선 axis 성분이 양이면
    # 카메라 반대(=앞면). sign<0 이면 그 반대.
    component = normals[:, fit.axis] * (1.0 if fit.sign > 0 else -1.0)
    # 0 근처(옆면) 는 앞뒤 어디에도 아닌 얇은 스트립인데, 이걸 그리면
    # 옆면 자국이 남는다. 살짝 여유(1e-9) 만 두고 잘라 낸다.
    return component > 0.0


def render_line_drawing(
    vertices: np.ndarray,
    faces: np.ndarray,
    fit: ViewFit,
    scan_shape: tuple[int, int],
) -> np.ndarray:
    """스캔 프레임 안에 부품을 도면 스타일로 그린다.

    Args:
        vertices: (N, 3) 부품 좌표(mm).
        faces:    (M, 3) 삼각형 정점 인덱스.
        fit:      스캔 마스크에 맞춘 뷰. to_pixels 가 정점을 스캔
                  픽셀 좌표로 옮기는 데 쓴다.
        scan_shape: (H, W). 결과 이미지 크기 — 스캔과 같아야
                    alignment 를 identity 로 둘 수 있다.

    Returns:
        (H, W, 3) uint8 BGR. 부품 영역은 회색 채움, 외곽·구멍 윤곽선은
        검정. 부품이 프레임 밖으로 완전히 벗어난 경우 흰 이미지를 준다.
    """
    height, width = scan_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"scan_shape 가 잘못됐습니다: {scan_shape}")

    verts = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)

    xs, ys = to_pixels(verts, fit)
    projected = np.stack([np.asarray(xs, dtype=np.int32),
                          np.asarray(ys, dtype=np.int32)], axis=1)

    # 앞면만 남긴다 — 뒷껍질·리브 뒤가 겹쳐 "단면도" 처럼 보이는 문제를
    # 여기서 잘라낸다. 결과 실루엣이 극단적으로 작아지면(뷰가 뒤집혔거나
    # 얇은 판재라 앞·뒤 판정 여유가 없으면) 원래대로 전체 삼각형을 쓴다.
    silhouette = np.zeros((height, width), dtype=np.uint8)
    if faces_arr.size:
        front_mask = _front_facing_mask(verts, faces_arr, fit)
        selected = faces_arr[front_mask] if front_mask.any() else faces_arr
        triangles = projected[selected].astype(np.int32)
        cv2.fillPoly(silhouette, triangles, 255)

    output = np.full((height, width, 3), BACKGROUND_BGR, dtype=np.uint8)
    if not silhouette.any():
        return output

    output[silhouette > 0] = BODY_FILL_BGR

    # 라인 굵기는 프레임 크기에 비례해 잡는다 — 4K 스캔에서도 선이
    # 얇지 않게, 작은 스캔에서 굵어 뭉개지지도 않게.
    long_side = max(height, width)
    outer_thickness = max(2, long_side // 400)
    inner_thickness = max(1, long_side // 700)

    # RETR_CCOMP 로 외부(부모 없음)·내부(부모 있음) 계층만 두 층으로 받는다.
    contours, hierarchy = cv2.findContours(
        silhouette, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours and hierarchy is not None:
        for idx, contour in enumerate(contours):
            parent = hierarchy[0, idx, 3]
            thickness = outer_thickness if parent < 0 else inner_thickness
            cv2.drawContours(output, [contour], -1, LINE_BGR, thickness, cv2.LINE_AA)

    return output


__all__ = ["render_line_drawing"]
