# 1차 데모 체크리스트

## 완료 기준

편차 맵에서 라벨별 편차값과 지시선 끝점 좌표를 추출하고 CSV와 검토 이미지를 생성한다.
보정량 계산, 0라인 자동 선정, 보정 시트 생성은 이번 범위에 포함하지 않는다.

## 실행 전

- [ ] `transformers>=4.49.0`을 포함한 의존성을 설치했다.
- [ ] 승인된 로컬 경로에 VLM 모델을 준비했다.
- [ ] 회사 이미지와 산출물 경로가 Git 제외 대상인지 확인했다.
- [ ] 합성 테스트가 모두 통과했다.

```powershell
.venv\Scripts\python.exe -m pip install -r deviation_extraction/requirements.txt
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
```

## 사내 이미지 실행

입력과 출력은 저장소 밖의 보안 경로를 직접 지정한다.

```powershell
.venv\Scripts\python.exe deviation_extraction/run.py `
  --image "D:/secure/deviation_map.png" `
  --model "D:/secure/models/Qwen2.5-VL-3B-Instruct" `
  --out "D:/secure/result/deviation_points.csv" `
  --debug `
  --debug-out "D:/secure/result/deviation_points_debug.png" `
  --offline
```

## 결과 확인

- [ ] 화면의 라벨 수와 CSV 행 수가 일치한다.
- [ ] `value_mm`가 라벨 숫자와 일치한다.
- [ ] 디버그 이미지의 초록색 끝점이 실제 지시 위치와 일치한다.
- [ ] 실패한 값과 좌표가 빈 값과 상태 코드로 구분된다.
- [ ] 다른 형식의 이미지 2~3장에서도 같은 절차를 확인했다.

## 제출 항목

- 실행 코드와 README
- `deviation_points.csv`
- `deviation_points_debug.png`
- 이미지별 라벨 수, 값 오독 수, 좌표 오검출 수
- 현재 제약과 후속 개선 항목
