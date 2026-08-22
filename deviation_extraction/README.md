# [파트 3] 편차값·좌표 추출

## 역할
편차 이미지에서 라벨 박스의 수치와 지시선이 가리키는 포인트 좌표를 뽑는다.
포인트 번호는 회의 결정에 따라 **최상단부터 시계 방향**으로 매긴다.

## 입력
- `data/intermediate/deviation_map.png` 또는 `clean_deviation_map.png`

## 출력 — `data/intermediate/deviation_points.csv`
| 컬럼 | 설명 |
|---|---|
| `point_id` | 1부터. 최상단 시계 방향 |
| `x`, `y` | 지시선이 가리키는 지점의 픽셀 좌표 |
| `value` | 라벨에 적힌 편차값 |
| `label_x`, `label_y` | 라벨 박스 중심 좌표 |
| `confidence` | OCR 신뢰도 0~1 |

## 파트 2와의 연계
이 CSV 가 있으면 파트 2의 색상→값 보정을 검증·교정할 수 있다.
라벨 값과 그 좌표의 색상 판독값을 비교하면 컬러바 보정이 맞는지 바로 확인된다.
특히 컬러바가 잘린 이미지(JD_64XX2)에서 유용하다.

## 할 일
- [ ] 라벨 박스 검출 (`zero_line_detection/annotations.py` 재사용 가능)
- [ ] 박스 안 숫자 OCR
- [ ] 지시선 추적 → 끝점 좌표
- [ ] 시계 방향 번호 부여
