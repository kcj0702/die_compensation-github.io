# CATIA 관리면 엔진

`management_surface.py`는 CATIA 관리면 트리와 스캔 PNG의 편차장을 연결한다.

## CATIA 트리 규칙

```text
+0.5관리 기준면
├─ SPEC SURFACE-01          원래 면
├─ 01) -0.5 방향 적용면    Offset 결과면
└─ BOUNDARY                 관리 영역 외곽선
```

목표값은 상위 Geometrical Set 이름에서만 읽는다. 위 예제의 목표는 `+0.5mm`다.
적용면 이름의 `-0.5`는 실제 형상 방향과 다를 수 있으므로 값이나 부호 계산에
사용하지 않는다. `01`, `A`, `B` 같은 접두 ID만 SPEC 면과 적용면을 연결하는
데 사용한다.

```python
from cad_import.management_surface import discover_management_groups

groups = discover_management_groups(catia_part)
manifest = [group.to_dict() for group in groups]
```

## 정합 재사용

엔진은 별도의 정합식을 갖지 않는다. CAD 뷰어의 `cad_import.overlay.ViewFit`을
그대로 받고, CAD 원래 좌표에서 뷰어가 사용한 `cad_offset`을 뺀 후
`overlay.to_pixels()`로 투영한다.

```python
from cad_import.management_surface import project_boundary

mask, polygon_px, visible_ratio = project_boundary(
    boundary_points_cad,
    viewer_fit,
    deviation_values.shape,
    cad_offset=viewer_cad_offset,
)
```

CAD 뷰어에서 아직 정합하지 않은 배치 작업에서는 `fit_with_cad_viewer()`를
호출한다. 이 함수도 내부에서 기존 `fit_view()`와 `polish_by_hit_rate()`만
사용한다.

## 보정량 계산

```python
from cad_import.management_surface import (
    apply_management_value,
    calculate_management_result,
)

corrected_values = apply_management_value(
    deviation_values,
    group,
    mask,
    part_mask=scan_part_mask,
)

result = calculate_management_result(
    group,
    deviation_values,
    mask,
    polygon_px=polygon_px,
    visible_ratio=visible_ratio,
    part_mask=scan_part_mask,
    machining_sign=1.0,
)
```

계산식은 아래와 같다.

```text
대표 측정값 = BOUNDARY 내부 유효 편차의 중앙값
수정 편차   = 대표 측정값 - 상위 그룹 관리치
가공 보정치 = machining_sign × 수정 편차
```

`machining_sign`은 절삭/용접에 사용하는 회사 부호 규칙이다. 실제 품번으로
부호가 검증되기 전에는 기본값 `1.0`을 사용해 수정 편차와 가공 지시를 구분한다.

## 옆면 처리

현재 `project_boundary()`의 `visible_ratio`는 경계가 PNG 화면 안에 들어왔는지만
판정한다. 다른 CAD 면에 가려진 옆면/후면까지 판정하는 깊이 검사는 후속 연결
단계에서 전체 CAD 메시의 광선 교차로 추가해야 한다. 가시 비율이 기준보다
낮거나 유효 픽셀이 부족하면 결과는 `PARTIAL` 또는 `HIDDEN`이며 보이지 않는
영역을 임의의 스캔값으로 채우지 않는다.
