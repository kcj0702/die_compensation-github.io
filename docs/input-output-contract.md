# 입출력 규격 (I/O Contract)

각 파트는 서로의 **코드를 import 하지 않는다.** `data/intermediate/` 의 파일로만 연결한다.
그래야 한 파트의 내부 구현이 바뀌어도 다른 사람 작업이 멈추지 않는다.

---

## 1. 파일 흐름

```
              deviation_map.png  (원본 편차 이미지)
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
    [4] 라벨제거    [2] 0-Line     [3] 포인트추출   [5] 깊이측정
        │              │              │              │
        ▼              ▼              ▼              ▼
 clean_deviation  zero_line_mask  deviation_    depth_
   _map.png          .png          points.csv   measurements.csv
        │              │              │              │
        └──────────────┴──────┬───────┴──────────────┘
                              ▼
                        result.json
                              │
                              ▼
                          [1] UI
```

`[4]` 의 `clean_deviation_map.png` 가 있으면 `[2]`, `[3]` 이 자동으로 그것을 우선 사용한다.
없으면 원본으로 동작한다. **파트 4를 기다릴 필요가 없다.**

---

## 2. 공통 파일 규격

### `deviation_map.png` — 입력
3D 스캔 편차 히트맵. 아래를 **반드시 포함**해야 한다.

| 요소 | 필요한 이유 |
|---|---|
| 컬러바(범례) | 파트 2가 색→편차값 보정에 사용. 지우면 안 된다 |
| 히트맵 본체 | — |

크기는 제각각이어도 된다. 좌우 어느 쪽에 컬러바가 있어도, 이미지가 180° 회전돼 있어도
파트 2가 자동으로 판별한다.

### `clean_deviation_map.png` — [4] 산출
라벨·지시선·제목을 지운 이미지.
- **원본과 같은 해상도**를 유지할 것 (파트 2·3이 좌표를 그대로 쓴다)
- **컬러바는 남길 것**

### `zero_line_mask.png` — [2] 산출
0 **영역** 이진 마스크. 8bit 흑백, 원본과 같은 크기.
- `255` = `|편차| ≤ 허용오차` 인 영역
- `0` = 그 외

허용오차에 따라 넓이가 달라진다. 임계값과 무관한 결과가 필요하면
`zero_line_crossing.png`(부호 경계선)를 쓴다.

### `deviation_points.csv` — [3] 산출

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `point_id` | int | 1부터. 최상단에서 시계 방향 |
| `x`, `y` | int | 지시선이 가리키는 지점의 픽셀 좌표 |
| `value` | float | 라벨에 적힌 편차값 |
| `label_x`, `label_y` | int | 라벨 박스 중심 |
| `confidence` | float | OCR 신뢰도 0~1 |

### `depth_measurements.csv` — [5] 산출

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `point_id` | int | `deviation_points.csv` 와 대응 |
| `x`, `y` | int | 픽셀 좌표 |
| `depth_rel` | float | 상대 깊이 0~1 |
| `curvature` | float | 국소 곡률 |
| `feature` | str | flat / fillet / step / hole |

### `result.json` — UI 입력
`pipeline/run_demo.py` 가 만든다. 없는 산출물은 키가 빠지고 `warnings` 에 기록된다.

```json
{
  "part_no": "",
  "source_image": "deviation_map.png",
  "generated_at": "2026-08-22T19:40:00",
  "images": {
    "deviation_map": "data/intermediate/deviation_map.png",
    "zero_line_mask": "data/intermediate/zero_line_mask.png",
    "zero_line_overlay": "data/intermediate/zero_line_overlay.png"
  },
  "tables": {
    "zero_line_regions": "data/intermediate/zero_line_regions.csv"
  },
  "zero_line": {
    "total_zero_px": 53047,
    "part_px": 167875,
    "zero_ratio": 0.316,
    "tolerance": 0.1,
    "tolerance_unit": "normalized",
    "region_count": 11,
    "colorbar": { "side": "right", "vmin_at": "bottom" },
    "warnings": []
  },
  "warnings": ["없음: deviation_points.csv ([3] 편차값·좌표)"]
}
```

경로는 **프로젝트 루트 기준 상대경로**다. UI 는 루트에서 실행한다고 가정한다.

---

## 3. 파트 2의 부가 산출물

필수는 아니지만 검증·발표에 쓴다.

| 파일 | 내용 |
|---|---|
| `zero_line_crossing.png` | 부호 경계선. 허용오차와 무관하게 결정되는 진짜 0-Line |
| `zero_line_tolerance_sweep.csv` | 허용오차별 0 영역 면적 (민감도 분석) |
| `zero_line_overlay.png` | 원본 위에 0-Line 을 얹은 이미지. 발표 화면용 |
| `zero_line_centerline.png` | 0 밴드의 중심선 (세선화 결과) |
| `zero_line_regions.csv` | 영역별 면적·중심좌표·외접사각형·평균편차 |
| `zero_line_contours.json` | 영역 윤곽 폴리라인 좌표 |
| `zero_line_report.json` | 처리 파라미터, 컬러바 정보, 통계, 경고 |

### `zero_line_regions.csv` 컬럼

| 컬럼 | 설명 |
|---|---|
| `region_id` | 영역 번호 |
| `area_px` | 픽셀 면적 |
| `centroid_x`, `centroid_y` | 중심 좌표 |
| `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` | 외접 사각형 |
| `perimeter_px` | 둘레 |
| `mean_value` | 영역 평균 편차 |
| `unit` | `mm` 또는 `normalized` |

---

## 4. 단위 규약 — 중요

파트 2의 편차값 단위는 **컬러바 최소·최대값을 알려줬는지에 따라 달라진다.**

| 조건 | `unit` | 의미 |
|---|---|---|
| `--vmin`, `--vmax` 를 준 경우 | `mm` | 실제 편차 mm |
| 주지 않은 경우 | `normalized` | -1 ~ +1 로 정규화된 값 |

정규화 모드에서도 **0의 위치는 정확하다.** 0-Line 검출만 목적이면 그대로 써도 된다.
mm 단위 수치가 필요하면 컬러바에 적힌 값을 `--vmin/--vmax` 로 넘긴다.

---

## 5. 좌표계

모든 좌표는 **원본 이미지 픽셀 좌표**다. 좌상단이 (0, 0), x 는 오른쪽, y 는 아래.
리사이즈·회전한 좌표를 내보내지 않는다. 파트 간 좌표가 어긋나는 가장 흔한 원인이다.

---

## 6. 아직 정하지 못한 것

| 항목 | 상태 | 확인 경로 |
|---|---|---|
| 실제 스캔 이미지의 mm 대비 픽셀 축척 | 미정 | CAD 파일 확보 시 |
| 0 판정 허용오차의 현장 기준 | 미정 (현재 컬러바 반경의 10%) | 멘토 확인 |
| `point_id` 시계 방향 시작점의 정확한 규칙 | 대략 합의 | 파트 3 구현 시 확정 |
| 보정시트의 `"0" 라인` 과 스캔 0 영역의 관계 | **미확인** | 아래 참고 |

마지막 항목은 프로젝트 전체에 영향이 있다. `zero_line_detection/README.md` 의
"알려진 한계" 를 참고할 것.
