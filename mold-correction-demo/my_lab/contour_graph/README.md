# Contour Graph

`scan_point_contour`에서 생성한 폐곡선의 각 점을 원본 3D 스캔 이미지의 숫자 라벨과 1:1로 연결하고, 라벨에 인쇄된 편차값(mm)으로 윤곽선 기준 편차 그래프를 그립니다.

숫자 판독은 외부 OCR 서비스 없이 라벨의 부호, 정수 한 자리, 소수점, 소수 한 자리를 분리해 로컬에서 처리합니다. 주변 스캔 색상은 편차값으로 사용하지 않습니다.

- 빨강: 양수 편차
- 초록: 0에 가까운 편차
- 파랑: 음수 편차
- 흰색 선: 스캔포인트 윤곽선 기준선

아직 그래프와 기준선의 교차점 및 제로포인트는 계산하지 않습니다.

```powershell
conda activate AJ_ENV
cd C:\ajin\my_lab\contour_graph
python run.py
```

결과는 `output/<원본 이미지 이름>/01_deviation_graph.png`와 `deviation_graph.json`에 저장됩니다.
