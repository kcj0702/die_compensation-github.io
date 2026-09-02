# 면적 우선 + 모서리 추종 제로라인 판정용 프로토타입

현재 제로라인 엔진과 분리된 검토용 실험입니다. 엔진 파일이나 서비스 흐름은 수정하지 않습니다.

## 판정 규칙

1. 각 스캔과 같은 품번의 별도 컬러맵 PNG에서 실제 색상 램프를 읽습니다.
2. 컬러맵 눈금 범위로 색상을 mm 편차로 변환합니다.
   - `JD_64XX2-DR000`: `-1.5 ~ +2.0 mm`
   - `JD_67XX6-DR000`: `-3.0 ~ +3.0 mm`
   - `JD_71XX2-DR000`: `-2.0 ~ +2.0 mm`
3. `> +0.5 mm`와 `< -0.5 mm`를 서로 다른 연결영역으로 구합니다.
4. 하나의 연결영역이 전체 파트 면적의 `5% 이상`일 때만 보정 영역으로 채택합니다.
5. 채택 영역과 비보정 유효면이 맞닿는 **내부 경계**만 남깁니다. 파트 외곽, 관통 홀, 색상 매핑 불가 회색 면과 맞닿은 경계는 제외합니다.
6. 내부 경계의 12 px 이내에서 원본 스캔의 영상 모서리를 검출해 제로라인 후보로 표시합니다.

파트 면적은 흰 배경과 실제 관통 홀을 제외한 최대 연결 재료 영역입니다. 색상 매핑이 불가능한 회색 면도 파트 면적에는 포함하지만, 편차 임계 영역에는 포함하지 않습니다.

## 결과 색상

- 빨강: `> +0.5 mm` 보정 후보/영역
- 파랑: `< -0.5 mm` 보정 후보/영역
- 흰색: 5% 조건을 통과한 보정 영역의 원래 임계 경계
- 노랑: 원래 경계 주변의 영상 모서리를 따라 검출한 제로라인 후보

각 품번 폴더의 `review_board.png`에서 원본, 임계 후보, 5% 채택 영역, 제로라인 후보를 한 화면에서 비교할 수 있습니다. `zero_line_overlay.png`는 최종 판단용 고해상도 오버레이입니다.

## 다시 생성

```powershell
& .venv\Scripts\python.exe mold-correction-demo\experiments\zero_line_area_edge_preview\generate_preview.py
```

판정값을 바꿔 비교하려면 다음 옵션을 사용할 수 있습니다.

```powershell
& .venv\Scripts\python.exe mold-correction-demo\experiments\zero_line_area_edge_preview\generate_preview.py `
  --threshold-mm 0.5 --min-area-ratio 0.05 --edge-search-radius 12
```
