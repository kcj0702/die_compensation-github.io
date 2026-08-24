# Zero Line Advance

기존 팀원의 `zero_line_detection` 폴더를 수정하지 않고 작업용 제로라인을 실험하는 폴더입니다.

## 현재 기준

제로라인은 색상 노이즈를 그대로 따라가는 수학적 등고선이 아니라, 작업자가 보정시트에서 쉽게 사용할 수 있는 단순한 직선 경계로 생성합니다.

1. 라벨의 숫자와 측정 포인트를 검출합니다.
2. 같은 형상 윤곽선에서 `음수 → 양수` 또는 `양수 → 음수`로 바뀌는 구간을 선형 보간하여 0점을 구합니다.
3. 숫자가 직접 `0`으로 표시된 포인트도 확실한 0점으로 사용합니다.
4. 같은 부호 안에서 값이 증가했다가 감소하는 경우는 0점으로 추정하지 않습니다.
5. 확실한 0점 중 작업 영역의 시작점과 끝점을 선택합니다.
6. 시작점과 끝점이 부품의 같은 면에 있으면 주요 개구부의 위·아래에 4개 꼭짓점짜리 경로 후보를 만듭니다.
7. 각 후보의 평균 편차, `-0.5~+0.5` 이탈 비율, 구멍·부품 밖 통과 비율을 비교하여 가장 안전한 위치를 선택합니다.
8. 시작점과 끝점이 서로 다른 면에 있으면 구멍과 편차 색상을 고려해 경로를 찾은 뒤 최대 6개 꼭짓점으로 단순화합니다.

`JD_67XX`는 영역 특성이 다르므로 기본 실행 대상에서 제외합니다.

## 실행

```powershell
conda activate AJ_ENV
cd C:\ajin\die_compensation-zero-test\mold-correction-demo\zero_line_advance
python run.py
```

현재 입력 파일이 다른 폴더에 있다면 다음처럼 실행할 수 있습니다.

```powershell
python run.py `
  --input "C:\ajin\die_compensation-repo\mold-correction-demo\label_removal\input\JD_64XX2-DR000 3D 스캔.png" `
  --clean-dir "C:\ajin\die_compensation-repo\mold-correction-demo\label_removal\output\2_labels_inpainted"
```

## 주요 결과

| 파일 | 내용 |
|---|---|
| `04_work_zero_line.png` | 최종 작업용 제로라인 |
| `06_numeric_zero_points.png` | 숫자로 확정한 전체 0점 후보 |
| `07_operator_vertices.png` | 최종선의 꼭짓점 위치 |
| `zero_line_mask.png` | 다른 기능에서 사용할 이진 선 마스크 |
| `result.json` | 0점, 꼭짓점 및 실행 설정 |

최종 확인 대상은 `04_work_zero_line.png`입니다.
