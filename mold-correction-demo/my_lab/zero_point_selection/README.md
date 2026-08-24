# Zero Point Selection

`contour_graph`의 라벨 편차값과 스캔포인트 윤곽선을 이용해 0포인트를 선정합니다.

0포인트 선정 기준은 다음과 같습니다.

- 서로 이웃한 두 라벨값의 부호가 바뀌면 선형 보간으로 편차값이 0이 되는 윤곽선 위치를 계산합니다.
- 라벨값이 정확히 `0.0`이면 해당 스캔포인트 자체를 0포인트로 선정합니다.
- 폐곡선의 마지막 포인트와 첫 번째 포인트 사이도 검사합니다.
- 코너의 그래프 꼬임처럼 편차값과 무관하게 생긴 기하학적 교차는 제외합니다.

```powershell
conda activate AJ_ENV
cd C:\ajin\my_lab\zero_point_selection
python run.py
```

결과는 다음 위치에 생성됩니다.

```text
output/<원본 이미지 이름>/01_zero_points.png
output/<원본 이미지 이름>/zero_points.json
```

확인 이미지에서 노란색은 부호 변화로 보간한 0포인트이고, 초록색은 라벨에 `0.0`으로 표시된 0포인트입니다.
