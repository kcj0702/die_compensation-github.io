"""제품데이터 정렬 단계에서 공유하는 경로와 경험적 임계값."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PRODUCT_DIR = ROOT_DIR / "data" / "product"
ALIGNMENT_DIR = ROOT_DIR / "data" / "alignment"
# CATIA 에서 export 한 STEP/STL 을 품번당 한 파일로 보관한다. PNG 가 없어도
# 여기 파일이 있으면 스캔 방향에 맞춰 뷰를 즉석에서 렌더한다.
MESH_DIR = ROOT_DIR / "data" / "product_mesh"

# 제품데이터는 초록/파랑 CAD 렌더에 검은 외곽선이 얹힌 흰 배경 이미지다.
# label_removal.build_scan_mask와 같은 "흰색에서 얼마나 떨어졌는가" 기준을 쓴다.
PRODUCT_FOREGROUND_THRESHOLD = 20
# CAD 외곽선이 부품 내부를 여러 조각으로 끊어 놓는 경우가 있어 닫기 연산을
# 먼저 적용한다. 커널이 커지면 얇은 구멍이 메워지므로 3px로 제한한다.
PRODUCT_CLOSE_KERNEL = 3

# 방향 후보는 좌우/상하 반전 네 가지뿐이다. 두 이미지 모두 축에 정렬된
# 정투영 export라 임의 각도 회전은 다루지 않는다.
FLIP_CANDIDATES = ((False, False), (False, True), (True, False), (True, True))

# 점수는 외곽 실루엣 + 내부 구멍 + 경계 밴드의 IoU 합이다. 경계 밴드는 작은
# 노치처럼 방향을 가르는 특징을 키워 주므로 가중치를 2배로 준다.
BOUNDARY_BAND_WIDTH = 3
BOUNDARY_BAND_WEIGHT = 2.0

# bbox끼리 맞추면 두 마스크의 경계 두께 차이가 그대로 배율에 섞여 부품이 조금
# 축소된다. 실측에서 구멍 중심이 반경 방향으로 최대 3.4px 안쪽으로 당겨졌다.
# 그래서 방향을 정한 뒤 겹침이 가장 큰 배율·이동을 다시 찾는다.
REFINE_STEPS = ((0.010, 2.0), (0.004, 0.8), (0.0015, 0.3))
REFINE_MAX_ROUNDS = 6
REFINE_MIN_GAIN = 1e-6

# 외곽 IoU가 이 값에 못 미치면 같은 부품이 아니거나 제품데이터가 잘못 등록된
# 것으로 본다. 실측 3개 품번은 정방향에서 0.973 ~ 0.990이었다.
MIN_OUTLINE_IOU = 0.90
# 1위와 2위 방향의 점수 차. RING SUNROOF처럼 상하좌우가 거의 대칭인 부품은
# 0.018까지 붙어 자동 판정을 신뢰할 수 없다. 나머지 두 품번은 0.85 이상이었다.
MIN_DECISION_MARGIN = 0.20

# 품번 형식: 영숫자 5자리 + 하이픈 + 영문 2자리 + 숫자 3자리 (예: 64XX2-DR000).
# OOO/파일명 예시.xlsx의 "차종_품번_품명_공정_날짜" 규칙에서 품번 부분이다.
PART_NUMBER_PATTERN = r"[0-9A-Z]{5}-[A-Z]{2}[0-9]{3}"

# 렌더링: 제품데이터 원본이 작아(653x260) 라벨을 그대로 얹으면 읽을 수 없다.
COMPOSE_MIN_WIDTH = 1600
COMPOSE_MAX_SCALE = 4
# 품번마다 원본 해상도가 달라 마커와 글자는 결과 이미지 너비 비율로 정한다.
# 그래야 어떤 품번이든 부품 대비 같은 크기로 보인다.
COMPOSE_MARKER_RADIUS_RATIO = 0.0028
COMPOSE_MARKER_RADIUS_MIN = 3
COMPOSE_LABEL_OFFSET_RATIO = 0.015
COMPOSE_LABEL_OFFSET_MIN = 14
COMPOSE_FONT_SCALE_RATIO = 0.00028
COMPOSE_FONT_SCALE_MIN = 0.35
COMPOSE_FONT_THICKNESS_RATIO = 0.0006

