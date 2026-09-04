"""mesh 를 스캔과 픽셀 단위로 정확히 정합된 셰이딩 이미지로 렌더한다.

[왜 CATIA 캡처를 버렸나]
`catia_capture.py` 는 CATIA 화면을 그대로 긁어 왔지만 두 가지 근본 문제가
있었다.

  1. CATIA 의 `Reframe()` 이 알아서 카메라를 맞추기 때문에, 우리가
     `overlay.fit_view` 로 계산한 "이 CAD 점이 스캔의 이 픽셀에 해당한다"
     는 매핑과 CATIA 카메라가 실제로 잡은 프레임이 **일치한다는 보장이
     없다**. 사후에 부품 bounding box 를 crop 해서 근사로 끼워 맞춰도
     정합 오차가 남는다.
  2. CATIA 문서에 저장된 주석·구속조건·스케치·축 시스템이 화면에 같이
     찍혀 나온다. 지우려면 CATIA 쪽 표시 설정을 일일이 꺼야 하는데
     문서마다 무엇이 켜져 있는지 다르다.

[이 모듈이 하는 일 — 우리가 카메라를 완전히 통제한다]
VTK 로 직접 오프스크린 렌더링하면 이 둘이 한 번에 풀린다. 화면에는
우리가 그린 메시 외에 아무것도 없고(주석 원천 차단), 카메라를 우리가
원하는 수식 그대로 세팅하므로 `overlay.to_pixels` 와 **픽셀 단위로
정확히** 일치시킬 수 있다.

[정합 수학 — 캔버스를 두 단계로 나눈다]
`to_pixels` 는 3D 정점을 이렇게 스캔 픽셀로 옮긴다(overlay.py 참고):

    flat_u = v[u_axis] * (sign, 뒤집혔으면 -sign)
    flat_v = v[v_axis] * (안 뒤집었으면 1, 뒤집었으면 -1)
    turned = R(angle) @ [flat_u, flat_v]
    pixel  = (turned - [origin_u, origin_v]) / mm_per_px

뒤집기(flip)와 회전(angle)은 정투영(orthographic) 위에서는 **투영 전
3D 좌표에 적용하든, 투영 후 2D 이미지에 적용하든 결과가 같다** —
원근이 없기 때문이다. 그래서:

  1. VTK 로는 flip·angle 없이 "캔버스" 뷰만 그린다 — 카메라 방향은
     axis/sign 이 정하는 6방향 표준 뷰 그대로, 화면 안에서 더 돌거나
     뒤집지 않는다. 이 캔버스 전용 원점(canon_origin_u/v)과 스케일
     (mm_per_px, fit 과 동일값)을 우리가 직접 정한다.
  2. 캔버스 픽셀 -> 최종 스캔 픽셀로 옮기는 아핀 변환 A, b 를
     flip/angle/origin 차이로부터 유도해 `cv2.warpAffine` 한 번으로
     적용한다. 아래 `_registration_affine` 이 그 유도식이고, 무작위
     좌표로 `overlay.to_pixels` 와 오차 <0.5px(반올림 오차) 로 검증했다.

[뒷면 가림]
`render_view.render_line_drawing` 은 법선 부호로 뒷면을 걸러내는
근사였다. VTK 는 실제 3D 레스터라이저라 z-buffer 로 가장 가까운 면만
자연히 남는다 — 리브·엠보스가 서로 가리는 관계까지 실제 형상처럼
나온다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .overlay import ViewFit, _fit_axes


# 6방향 표준 뷰 — catia_capture.py 와 같은 표(문서 정합용으로 재사용).
# (axis, sign) -> (카메라가 보는 방향 단위벡터, view-up 단위벡터).
# view-up 은 "이 방향이 화면 위쪽" 이라는 뜻이며, 아래 _render_canonical
# 에서 부호를 뒤집어 v_axis 증가 방향이 화면 아래로 가도록 맞춘다
# (to_pixels 의 pixel_y 증가 = 아래쪽 관례와 맞추기 위해).
_AXIS_UNIT = {0: np.array([1.0, 0.0, 0.0]),
              1: np.array([0.0, 1.0, 0.0]),
              2: np.array([0.0, 0.0, 1.0])}

# 배경·재질 — 시트에 얹을 "깨끗한 부품 사진" 톤. 순백 배경에 은은한 금속
# 그레이 재질, 위에서 살짝 비스듬한 조명으로 곡면 굴곡이 읽히게 한다.
BACKGROUND_RGB = (1.0, 1.0, 1.0)
MATERIAL_RGB = (0.62, 0.65, 0.70)
EDGE_RGB = (0.12, 0.12, 0.14)
# 이 각도보다 급하게 꺾이는 곳만 "모서리 선" 으로 그린다. STEP 테셀레이션은
# 곡면을 작은 평면 조각으로 잘게 쪼개므로, 값을 낮게 두면 곡면 자체의
# 테셀레이션 결(진짜 형상 특징이 아닌 삼각형 격자)까지 선으로 잡혀 나온다.
FEATURE_ANGLE_DEG = 55.0


def _canonical_uv(vertices: np.ndarray, u_axis: int, v_axis: int, sign: int) -> tuple[np.ndarray, np.ndarray]:
    """flip·회전이 없는 '캔버스' 좌표. to_pixels 의 flat_u/flat_v 에서
    flip 배수만 뺀 것과 같다 — sign 은 축의 원래 정의라 캔버스에도 넣는다.
    """
    canonical_u = vertices[:, u_axis] * sign
    canonical_v = vertices[:, v_axis] * 1.0
    return canonical_u, canonical_v


def registration_affine(fit: ViewFit, canon_origin_u: float, canon_origin_v: float,
                         canon_mm_per_px: float | None = None) -> np.ndarray:
    """캔버스 픽셀 -> 최종(스캔) 픽셀 2x3 아핀 행렬. cv2.warpAffine 용.

    유도는 모듈 docstring 참고. `canon_mm_per_px` 를 안 주면(=None) 캔버스와
    최종이 같은 스케일(fit.mm_per_px)을 쓴다고 보고 스케일 항이 사라진다
    (VTK 렌더 — 우리가 캔버스를 fit.mm_per_px 그대로 그렸을 때).
    CATIA 캡처처럼 캔버스 자체의 실제 스케일을 모르고 사후에 "이 픽셀
    범위가 이 mm 범위였다" 로 역산해야 하는 경우엔 그 값을 넘긴다 —
    이땐 스케일 비율(canon_mm_per_px / fit.mm_per_px)이 행렬에 곱해진다.
    """
    fu = -1.0 if fit.flip_u else 1.0
    fv = -1.0 if fit.flip_v else 1.0
    angle = float(getattr(fit, "angle", 0.0))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    scale = 1.0 if canon_mm_per_px is None else (canon_mm_per_px / fit.mm_per_px)
    # A = scale * R(angle) @ diag(fu, fv)
    a11, a12 = scale * cos_a * fu, scale * -sin_a * fv
    a21, a22 = scale * sin_a * fu, scale * cos_a * fv
    canon_origin = np.array([canon_origin_u, canon_origin_v])  # mm 단위 고정점 — scale 과 무관
    origin = np.array([fit.origin_u, fit.origin_v])
    # b = (R @ diag(fu,fv) @ canon_origin - origin) / mm_per_px.
    # canon_origin 은 이미 mm 값이라 scale(=canon_mm_per_px/mm_per_px) 을
    # 또 곱하면 안 된다 — scale 은 canon_PIXEL 을 mm 로 바꿀 때만 필요한
    # 항이라 A(선형 부분)에만 들어간다. (여기서 canon_origin 에 scale 을
    # 곱하는 버그가 있었다 — 위치가 원점에서 scale 배만큼 밀려나 있었다.)
    unscaled = np.array([[cos_a * fu, -sin_a * fv], [sin_a * fu, cos_a * fv]])
    b = (unscaled @ canon_origin - origin) / fit.mm_per_px
    return np.array([[a11, a12, b[0]], [a21, a22, b[1]]], dtype=np.float64)


# 하위 호환: shaded_render 내부에서 쓰던 이름.
_registration_affine = registration_affine


def _render_canonical(vertices: np.ndarray, faces: np.ndarray, fit: ViewFit,
                       margin_px: int = 24) -> tuple[np.ndarray, float, float]:
    """flip·angle 을 넣지 않은 캔버스 뷰를 VTK 로 오프스크린 렌더한다.

    Returns:
        (image, canon_origin_u, canon_origin_v). image 는 (H, W, 3) uint8 BGR,
        사람이 보는 방향으로 이미 위아래가 맞다(top-down).
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    u_axis, v_axis = _fit_axes(fit)
    axis = int(fit.axis)
    sign = 1 if fit.sign > 0 else -1

    canonical_u, canonical_v = _canonical_uv(vertices, u_axis, v_axis, sign)
    canon_min_u, canon_max_u = float(canonical_u.min()), float(canonical_u.max())
    canon_min_v, canon_max_v = float(canonical_v.min()), float(canonical_v.max())
    mm_per_px = float(fit.mm_per_px)

    canon_origin_u = canon_min_u - margin_px * mm_per_px
    canon_origin_v = canon_min_v - margin_px * mm_per_px
    width = max(2, int(math.ceil((canon_max_u - canon_min_u) / mm_per_px)) + 2 * margin_px)
    height = max(2, int(math.ceil((canon_max_v - canon_min_v) / mm_per_px)) + 2 * margin_px)

    # --- VTK 폴리데이터 ---
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    faces_arr = np.ascontiguousarray(faces, dtype=np.int64)
    # vtkCellArray 는 [n, i0, i1, ..., in, n, ...] 형태의 연결 배열을 받는다.
    # numpy_to_vtkIdTypeArray 전용 헬퍼를 써야 한다 — vtkIdTypeArray.SetArray 를
    # 직접 low-level 로 부르면(이전 시도) 버퍼 소유권이 꼬여 세그폴트가 났다.
    n_faces = len(faces_arr)
    connectivity = np.empty((n_faces, 4), dtype=np.int64)
    connectivity[:, 0] = 3
    connectivity[:, 1:] = faces_arr
    id_type_array = numpy_to_vtkIdTypeArray(connectivity.ravel(), deep=True)
    cells.SetCells(n_faces, id_type_array)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    # STEP 을 면별로 각각 테셀레이션하다 보면 삼각형 감김 방향(winding)이
    # 국소적으로 뒤집혀 있는 곳이 섞여 나온다 — 그 삼각형만 조명을 등지고
    # 검게 나와 "얼룩"처럼 보인다. Consistency/AutoOrientNormals 로 인접
    # 삼각형끼리 방향을 맞춘다(완전한 워터타이트 솔리드가 아니라도 대부분
    # 교정된다).
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.NonManifoldTraversalOff()
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*MATERIAL_RGB)
    actor.GetProperty().SetAmbient(0.35)
    actor.GetProperty().SetDiffuse(0.75)
    actor.GetProperty().SetSpecular(0.15)
    actor.GetProperty().SetSpecularPower(12)
    # 삼각형 엣지를 전부 그리면(EdgeVisibilityOn) 테셀레이션 격자가 그대로
    # 드러나 지저분하다 — 대신 vtkFeatureEdges 로 실제 형상 경계(구멍·바깥
    # 윤곽·급격히 꺾이는 모서리)만 뽑아 별도 선으로 얹는다.
    actor.GetProperty().EdgeVisibilityOff()

    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputConnection(normals.GetOutputPort())
    feature_edges.BoundaryEdgesOn()       # 구멍 테두리 등 한쪽 면만 있는 경계
    # [버그였던 부분] NonManifoldEdgesOn 은 "세 면 이상이 만나는 모서리" 를
    # 잡으려던 의도였지만, STEP 은 B-Rep 의 면(face)마다 따로 테셀레이션해서
    # 이어붙이기 때문에 이웃 면끼리 정점이 딱 맞아떨어지지 않는 경우가 흔하다
    # — 그 모든 면 경계 이음매가 전부 "non-manifold" 로 잡혀, 실측 64XX2 에서
    # 전체 엣지의 33%(18만 개)가 여기서 나왔다(진짜 형상 특징은 각도 필터
    # 기준 7천 개 수준). 그래서 CAD 도면이 아니라 테셀레이션 격자처럼
    # 보였다. 꺼서 진짜 모서리(각도 필터)와 진짜 구멍 테두리(경계)만 남긴다.
    feature_edges.NonManifoldEdgesOff()
    feature_edges.ManifoldEdgesOff()      # 평평하게 이어지는 삼각형 사이 선은 제외
    feature_edges.FeatureEdgesOn()
    feature_edges.SetFeatureAngle(FEATURE_ANGLE_DEG)
    feature_edges.ColoringOff()

    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputConnection(feature_edges.GetOutputPort())
    edge_mapper.ScalarVisibilityOff()
    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetColor(*EDGE_RGB)
    edge_actor.GetProperty().SetLineWidth(1.4)
    # z-fighting 방지: 엣지 선을 면보다 살짝 카메라 쪽으로 당겨 그린다.
    edge_mapper.SetResolveCoincidentTopologyToPolygonOffset()

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(edge_actor)
    renderer.SetBackground(*BACKGROUND_RGB)
    # [버그였던 부분] SceneLight 를 world 좌표 (1,1,1) 에 뒀는데, 부품은
    # 보통 그로부터 수백~수천 mm 떨어진 차체 좌표계에 있다 — 조명이 사실상
    # 부품 한가운데 박혀 있는 셈이라 대부분의 면이 광원을 등지고 컴컴하게
    # 나왔다. CameraLight 는 카메라를 기준으로 항상 따라다니는 "헤드램프"
    # 라 부품 크기·위치와 무관하게 항상 정면에서 비춘다.
    key_light = vtk.vtkLight()
    key_light.SetLightTypeToCameraLight()
    key_light.SetPosition(0.3, 0.4, 1.0)
    key_light.SetFocalPoint(0.0, 0.0, 0.0)
    key_light.SetIntensity(0.85)
    renderer.AddLight(key_light)
    fill_light = vtk.vtkLight()
    fill_light.SetLightTypeToCameraLight()
    fill_light.SetPosition(-0.4, -0.2, 1.0)
    fill_light.SetFocalPoint(0.0, 0.0, 0.0)
    fill_light.SetIntensity(0.45)
    renderer.AddLight(fill_light)

    # --- 카메라: 평행투영, axis/sign 방향으로 바라보되 캔버스 좌표계로
    # 정확한 스케일·중심을 맞춘다. flip/angle 은 넣지 않는다(모듈 docstring). ---
    view_dir = _AXIS_UNIT[axis] * float(sign)  # 카메라가 바라보는 방향(0 -> 이 방향으로 전진)
    # up 벡터: v_axis 증가가 "화면 아래" 로 가야 pixel_y 관례와 맞으므로
    # 카메라 up(=화면 위쪽)은 v_axis 의 반대 방향.
    up_vec = -_AXIS_UNIT[v_axis]
    # 카메라 위치는 focal point 에서 view_dir 반대로 충분히 떨어진 곳.
    focal_point = vertices.mean(axis=0)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    distance = max(diag * 3.0, 100.0)
    camera_pos = focal_point - view_dir * distance

    camera = vtk.vtkCamera()
    camera.SetParallelProjection(True)
    camera.SetPosition(*camera_pos.tolist())
    camera.SetFocalPoint(*focal_point.tolist())
    camera.SetViewUp(*up_vec.tolist())
    # ParallelScale = 뷰의 세로 절반 높이(world 단위). 캔버스 높이(px)*mm_per_px/2.
    camera.SetParallelScale(height * mm_per_px / 2.0)
    # [버그였던 부분] 카메라 근/원 클리핑 평면을 안 정하면 VTK 기본값(수십~수백
    # 단위)이 그대로 남는다. 카메라를 부품 대각선의 3배 거리에 놓았으므로 그
    # 기본값보다 훨씬 멀어 부품 전체가 잘려 나가 배경만 렌더됐다 — "성공"
    # 로그만 보고 실제 화면이 비어 있는 줄 몰랐다. ResetCameraClippingRange 로
    # 액터 bounds 기준 near/far 를 다시 계산해야 한다.
    renderer.SetActiveCamera(camera)
    renderer.ResetCameraClippingRange()

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(width, height)
    render_window.Render()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(render_window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.Update()

    vtk_image = capture.GetOutput()
    from vtk.util.numpy_support import vtk_to_numpy
    raw = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    dims = vtk_image.GetDimensions()
    img = raw.reshape(dims[1], dims[0], -1)[:, :, :3]
    # VTK 프레임버퍼는 원점이 좌하단이라 위아래가 뒤집혀 있다 — 사람이 보는
    # 방향(원점 좌상단)으로 되돌린다. 이래야 위에서 고른 up_vec 규칙이
    # 실제 numpy 배열의 row 방향과 맞아떨어진다.
    img = img[::-1, :, :]
    # RGB(VTK) -> BGR(OpenCV 관례, 이 저장소 전체가 BGR 을 쓴다).
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    render_window.Finalize()
    return img_bgr, canon_origin_u, canon_origin_v


def render_registered_view(vertices: np.ndarray, faces: np.ndarray, fit: ViewFit,
                            scan_shape: tuple[int, int]) -> np.ndarray:
    """mesh 를 스캔 픽셀 좌표계에 정확히 정합된 셰이딩 이미지로 렌더한다.

    Args:
        vertices, faces: mesh 데이터.
        fit:              overlay.fit_view 가 계산한, 스캔 마스크에 맞춘 뷰.
        scan_shape:       (H, W). 결과 이미지 크기 — 반드시 스캔과 같아야
                          포인트·제로라인과 같은 좌표계가 된다.

    Returns:
        (H, W, 3) uint8 BGR. 배경은 순백, 부품은 셰이딩된 실제 3D 렌더.
        CATIA 캡처와 달리 주석·축·글자가 전혀 없다 — 우리가 그린 메시
        외에는 아무것도 화면에 없기 때문이다.
    """
    height, width = scan_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"scan_shape 가 잘못됐습니다: {scan_shape}")

    verts = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0 or len(faces_arr) == 0:
        return np.full((height, width, 3), 255, dtype=np.uint8)

    canonical_image, canon_origin_u, canon_origin_v = _render_canonical(verts, faces_arr, fit)
    affine = _registration_affine(fit, canon_origin_u, canon_origin_v)
    registered = cv2.warpAffine(
        canonical_image, affine, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return registered


__all__ = ["render_registered_view"]
