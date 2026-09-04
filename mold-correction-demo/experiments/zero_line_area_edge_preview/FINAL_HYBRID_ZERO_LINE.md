# 단일 하이브리드 제로라인 엔진

`zero_line_detection/generate_final_hybrid_zero_line.py`가 외부에서 호출하는 단일 실행 파일입니다.
엔진 내부의 `detect_zero_line()`이 공통 보정영역을 입력받아 1번 또는 2번
방식을 자동으로 선택합니다. 현재는 검토용 엔진이며 정식 UI 엔진에는 연결하지
않았습니다.

## 코드 책임 구분

- 공통 보정영역 및 1·2번 판정: KDT013 기준
- 1번 영역형 제로라인: KDT013 로직
- 2번 경로형 제로라인: 보존된 원본 1~6단계 전체 로직
- 두 로직의 입력 형식 변환과 공통 추가 조건: 내부 어댑터

`case2_original_pipeline/`에는 라벨 제거, 외곽 그래프, 제로점 선정,
`±0.7mm` 보정영역, 병합 소스를 원본 그대로 보존합니다.
`case2_route_selector.py`도 원본 `select_nearest_zero_points.py`
로직을 수정하지 않고 보존합니다. `case2_route_adapter.py`가 이 단계들을
메모리 안에서 순서대로 호출하고 엔진 공통 규칙인 최소 경로 길이 100px를
최종 결과에 적용합니다. 어댑터는 별도 프로그램이나 서버가 아니라 단일 엔진
내부에서 호출되는 함수입니다.

## 공통 판정

1. 보정영역은 `+0.6mm 초과 / -0.6mm 미만`, 연결영역 면적 `2% 초과`,
   17px 원형 opening 결과를 사용합니다.
2. 제로라인 가능 영역은 `파트 - 최종 +/- 보정영역`입니다.
3. 파트 면적의 1% 미만인 제로라인 가능 연결영역은 제거합니다.
4. 남은 제로라인 가능 영역이 파트의 40% 미만이면서 연결영역이 2개 이상이면
   1번, 그 외에는 2번으로 판정합니다.

## 단일 엔진 흐름

```text
입력 이미지와 컬러맵
  -> 공통 보정영역 검출
  -> 제로라인 가능 영역 및 방식 판정
     -> 1번: KDT013 영역형 제로라인
     -> 2번: 어댑터 -> 보존된 원본 1~6단계 전체 파이프라인
  -> 공통 마스크와 summary.json 출력
```

## 실행

```powershell
cd mold-correction-demo
..\.venv\Scripts\python.exe `
  zero_line_detection\generate_final_hybrid_zero_line.py
```

특정 이미지만 실행하려면 다음과 같이 지정합니다.

```powershell
..\.venv\Scripts\python.exe `
  zero_line_detection\generate_final_hybrid_zero_line.py `
  --spec JD_67XX6-DR000
```

결과는 `experiments/zero_line_area_edge_preview/results_final_hybrid_zero_line/<제품 키>/`에 생성됩니다.

- `review_board.png`: 보정영역, 방식 판정, 구성 과정, 최종 결과
- `final_zero_line_overlay.png`: 원본 위 최종 제로라인
- `final_zero_line_mask.png`: 최종 이진 마스크
- `summary.json`: 선택 방식, 경로, 검증 수치 및 어댑터 적용 내역
