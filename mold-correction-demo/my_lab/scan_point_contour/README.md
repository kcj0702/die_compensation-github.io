# Scan Point Contour

라벨의 연결선 끝에서 검출한 스캔포인트를 제품별 닫힌 도형으로 연결합니다.

```powershell
conda activate AJ_ENV
cd C:\ajin\my_lab\scan_point_contour
python run.py
```

결과는 `output/<이미지 이름>/`에 저장됩니다.

- `01_detected_scan_points.png`: 검출된 전체 스캔포인트
- `02_connected_closed_shapes.png`: 포인트를 연결한 닫힌 도형
- `scan_point_loops.json`: 도형별 순서가 정리된 포인트 좌표

제품별 기본 구조:

- `JD_64XX2`: 제품 외곽 1개, 큰 라운드 네모 구멍 1개, 작은 라운드 네모 구멍 1개
- `JD_67XX6`: 바깥에서 안쪽 순서의 중첩 도형 3개
- `JD_71XX2`: 제품 외곽을 따라가는 도형 1개

JD_64XX2에서 연결선이 너무 짧아 HSV 추적에서 빠지는 포인트는 숫자 상자와 목표 구멍 윤곽의 최근접 위치로 보완하며, JSON의 `inferred_short_leader_point_count`에 별도로 기록합니다.

`01_detected_scan_points.png`의 색상은 노란색이 HSV 직접 검출점, 주황색이 짧은 연결선 보완점, 회색이 도형 연결에서 제외한 점입니다.

`scan_point_loops.json`에서는 원본·보완 측정점이 `points`, 실제로 그린 폐곡선 좌표가 `connection_path`로 분리되어 있습니다. JD_64XX2의 내부 구멍은 검출·보완된 스캔포인트를 각 구멍 윤곽에 개별 배정하고, 윤곽 거리의 중앙값과 MAD를 이용해 튀는 점만 제외한 뒤 연결합니다. JD_67XX6은 제품 외곽과의 거리에서 첫 번째 점 껍질을 분리하고, 남은 점들의 외곽 껍질에서 두 번째 층을 다시 분리합니다. 모든 분리 기준은 거리 분포에서 자동 계산되며 포인트 번호를 사용하지 않습니다.
