# 실제 보정시트 기준 제로라인 평가 도구

실제 보정시트 PNG를 보면서 스캔 이미지 좌표 위에 작업자의 제로라인을 한 번
표시한 뒤, 현재 하이브리드 검출 결과와 수치로 비교하는 독립 도구입니다.

보정시트의 좁은 `0` 박스와 지시선, 제품 외곽 정합을 이용해 후보를 자동 생성합니다.
다만 보정치 점·지시선·문자가 같은 색인 시트가 있어 자동 후보를 곧바로 정답으로
간주하지 않고, 작업자가 확인해 저장한 JSON만 최종 정답으로 사용합니다.

## 가장 간단한 실행

프로젝트 최상위 폴더에서 아래 중 하나를 실행합니다.

```bat
cd /d "C:\Users\KDT013\Desktop\금형보정치\die_compensation-github.io"
.venv\Scripts\python.exe zero_line_evaluation\prepare_examples.py --force-auto
.venv\Scripts\python.exe zero_line_evaluation\review_sample.py 64XX2
.venv\Scripts\python.exe zero_line_evaluation\review_sample.py 67XX6
.venv\Scripts\python.exe zero_line_evaluation\review_sample.py 71XX2
```

자동 후보가 맞으면 창에서 `S`, `Q`만 누르면 됩니다. 잘못된 후보는 해당 점에
오른쪽 클릭하여 삭제하고, 빠진 위치는 왼쪽 클릭으로 추가합니다. 창을 닫으면 평가와
HTML 보고서 생성까지 자동으로 이어집니다.

## 1. 정답 표시

```bat
.venv\Scripts\python.exe zero_line_evaluation\annotate_ground_truth.py --scan "경북대KDT(14기) 자료\3D 스캔 이미지 및 보정치 시트_예시\JD_64XX2-DR000 3D 스캔.png" --sheet "경북대KDT(14기) 자료\3D 스캔 이미지 및 보정치 시트_예시\JD_64XX2-DR000 보정시트.png" --output "zero_line_evaluation\annotations\JD_64XX2-DR000.json" --kind points
```

창 왼쪽은 실제 보정시트, 오른쪽은 원본 스캔입니다. 보정시트를 참고하여 오른쪽
스캔의 같은 위치를 클릭합니다.

- 왼쪽 클릭: 점 추가
- `points` 모드에서 후보에 오른쪽 클릭: 가장 가까운 후보 삭제
- `line` 모드에서 오른쪽 클릭 또는 `N`: 현재 선 완료
- `U`: 마지막 점 취소
- `D`: 마지막 선 삭제
- `C`: 전부 지우기
- `M`: `line`/`points` 모드 전환
- `S`: 저장
- `Q` 또는 `Esc`: 종료

시트에 연속된 `"0" LINE`이 있으면 `line`, 제로 위치만 점으로 표시되어 있으면
`points`를 사용합니다. `points` 모드에서는 오른쪽 스캔을 클릭할 때마다 점 하나가
바로 확정되므로 `N`을 누를 필요가 없습니다.

## 2. 평가 및 결과 생성

```bat
.venv\Scripts\python.exe zero_line_evaluation\evaluate_zero_line.py --annotation "zero_line_evaluation\annotations\JD_64XX2-DR000.json" --output "zero_line_evaluation\results\JD_64XX2-DR000" --tolerance-px 8
```

실제 축척을 알면 `--mm-per-pixel 0.5 --tolerance-mm 2`처럼 mm 기준으로 평가할
수 있습니다. 축척이 없으면 픽셀 기준임을 보고서에 명시합니다.

생성 파일:

- `comparison.png`: 정답·검출·일치 구간 오버레이
- `prediction_overlay.png`: 현재 검출기가 만든 원본 오버레이
- `reference_sheet.png`: 평가에 사용한 실제 보정시트
- `metrics.json`, `metrics.csv`: 기계 판독용 수치
- `report.html`: 이미지와 지표를 한 화면에서 보는 결과 보고서

## 지표 해석

- `Line F1@거리`: 검출선 정확도와 실제선 재현율의 조화평균
- `평균 거리 오차`: 두 선 사이의 대칭 평균 거리
- `95% 거리 오차`: 크게 벗어난 일부 구간까지 확인하는 거리
- `Band IoU`: 양쪽 선을 허용거리만큼 확장했을 때 겹치는 비율
- `제로 포인트 적중률`: 실제 시트가 점만 제공할 때 각 점이 검출선 근처에 있는 비율

보정시트와 스캔 이미지의 확대·회전·투영이 서로 다르므로, 반드시 스캔 좌표 위에
정답을 옮겨 표시해야 합니다. 보정시트 PNG의 픽셀과 검출 마스크 픽셀을 직접
비교하면 의미 없는 숫자가 됩니다.

## 예시 3건 준비

아래 명령은 제공된 폴더의 스캔/보정시트를 자동으로 짝지어 현재 검출 결과와 실제
시트를 나란히 놓은 `review_board.png` 및 미검토 `annotation.json`을 만듭니다.

```bat
.venv\Scripts\python.exe zero_line_evaluation\prepare_examples.py --force-auto
```

`annotation.json`은 실제선 확인 전에는 `reviewed: false`이므로 평가기가 숫자를
만들지 않습니다. 잘못 추출된 빨간 지시선을 정답으로 간주해 높은 점수를 꾸며내는
것을 방지하기 위한 장치입니다.
