"""
AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼
================================================

현재 데이터
- 충청북도_청주시_버스정보시스템_20250401.csv
  · 서비스ID
  · 정류소명
  · 좌표(X)
  · 좌표(Y)

현재 CSV만으로 확인 가능한 것은 '실제 정류장 위치'입니다.
노선번호와 정류장 순서가 없으므로 실제 버스노선은 임의로 만들지 않고,
별도의 bus_routes.csv가 추가되었을 때 실제 노선 연동이 가능하도록 구성합니다.
"""

from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 미래예측형 대중교통 의사결정 지원",
    page_icon="🚌",
    layout="wide",
)

DATA_FILE = Path("충청북도_청주시_버스정보시스템_20250401.csv")
ROUTE_FILE = Path("bus_routes.csv")

REQUIRED_COLUMNS = {
    "서비스ID",
    "정류소명",
    "좌표(X)",
    "좌표(Y)",
}

FUTURE_FACTORS = {
    "신규 공동주택 개발": 8,
    "산업단지 조성·확장": 7,
    "철도 개통·환승거점 강화": 6,
    "대규모 상업시설 개발": 5,
    "대규모 행사·관광객 증가": 4,
}


# ============================================================
# 2. 데이터 처리 함수
# ============================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    """한글 CSV의 대표적인 인코딩을 순서대로 시도합니다."""
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as error:
            last_error = error

    raise last_error


@st.cache_data
def load_stops() -> tuple[pd.DataFrame | None, str | None]:
    """청주시 실제 정류장 데이터를 읽고 필요한 컬럼을 정리합니다."""

    if not DATA_FILE.exists():
        return None, (
            f"'{DATA_FILE.name}' 파일을 찾을 수 없습니다.\n"
            "app.py와 같은 폴더에 CSV 파일을 넣어주세요."
        )

    try:
        data = read_csv_auto(DATA_FILE)
    except Exception as error:
        return None, f"CSV 읽기 오류: {error}"

    missing = REQUIRED_COLUMNS - set(data.columns)

    if missing:
        return None, (
            "필수 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    data = data.copy()

    # 문자열 정리
    data["서비스ID"] = (
        data["서비스ID"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["정류소명"] = (
        data["정류소명"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # 좌표 숫자 변환
    data["경도"] = pd.to_numeric(
        data["좌표(X)"],
        errors="coerce",
    )

    data["위도"] = pd.to_numeric(
        data["좌표(Y)"],
        errors="coerce",
    )

    # 좌표가 없는 행 제거
    data = data.dropna(
        subset=["위도", "경도"]
    ).copy()

    # 대한민국 주변의 비정상 좌표 제거
    data = data[
        data["위도"].between(33, 39)
        & data["경도"].between(124, 132)
    ].copy()

    return data.reset_index(drop=True), None


@st.cache_data
def load_route_data() -> pd.DataFrame | None:
    """실제 노선 데이터가 있으면 읽습니다."""
    if not ROUTE_FILE.exists():
        return None

    try:
        return read_csv_auto(ROUTE_FILE)
    except Exception:
        return None


def haversine_km(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    """두 좌표 사이의 직선거리를 km로 계산합니다."""

    lat1, lon1 = point_a
    lat2, lon2 = point_b

    earth_radius = 6371.0

    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)

    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(d_lon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    return 2 * earth_radius * atan2(
        sqrt(a),
        sqrt(1 - a),
    )


def nearest_stops(
    data: pd.DataFrame,
    latitude: float,
    longitude: float,
    count: int = 6,
) -> pd.DataFrame:
    """특정 좌표 주변의 정류장 후보를 찾습니다."""

    result = data.copy()

    # 빠른 후보 검색용 좌표 거리
    result["_좌표거리"] = (
        (result["위도"] - latitude) ** 2
        + (result["경도"] - longitude) ** 2
    ) ** 0.5

    return result.nsmallest(
        count,
        "_좌표거리",
    ).drop(columns="_좌표거리")


def calculate_future_impact(
    selected_factors: list[str],
) -> int:
    """선택한 미래 도시변화 요소를 PoC용 영향지수로 계산합니다."""

    # 기본 +5%를 두고 선택 요소의 가중치를 합산
    impact = 5 + sum(
        FUTURE_FACTORS[factor]
        for factor in selected_factors
    )

    return min(impact, 50)


# ============================================================
# 3. 지도 함수
# ============================================================

def create_stop_map(
    data: pd.DataFrame,
    center: tuple[float, float],
    zoom: int = 11,
) -> folium.Map:
    """정류장 위치를 Folium 지도에 표시합니다."""

    map_object = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    for _, row in data.iterrows():
        popup_html = (
            f"<b>{row['정류소명']}</b><br>"
            f"서비스ID: {row['서비스ID']}<br>"
            f"경도: {row['경도']:.6f}<br>"
            f"위도: {row['위도']:.6f}"
        )

        folium.CircleMarker(
            location=[
                row["위도"],
                row["경도"],
            ],
            radius=4,
            color="black",
            fill=True,
            fill_opacity=0.65,
            tooltip=row["정류소명"],
            popup=folium.Popup(
                popup_html,
                max_width=280,
            ),
        ).add_to(map_object)

    return map_object


def create_candidate_route_map(
    points: list[tuple[float, float]],
) -> folium.Map:
    """AI 후보 노선을 지도에 표시합니다."""

    center_lat = sum(
        point[0] for point in points
    ) / len(points)

    center_lon = sum(
        point[1] for point in points
    ) / len(points)

    map_object = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    last_index = len(points) - 1

    for index, point in enumerate(points):
        if index == 0:
            label = "출발"
        elif index == last_index:
            label = "도착"
        else:
            label = f"중간 후보 {index}"

        folium.Marker(
            location=point,
            tooltip=label,
        ).add_to(map_object)

    folium.PolyLine(
        points,
        color="red",
        weight=6,
        opacity=0.85,
        tooltip="AI 노선 후보",
    ).add_to(map_object)

    return map_object


# ============================================================
# 4. 데이터 불러오기
# ============================================================

stops, load_error = load_stops()

if stops is None:
    st.error("청주시 정류장 데이터를 불러오지 못했습니다.")
    st.code(load_error)
    st.stop()


# ============================================================
# 5. 사이드바
# ============================================================

st.sidebar.title("🚌 분석 설정")

st.sidebar.metric(
    "실제 정류장 데이터",
    f"{len(stops):,}개",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 정류장 검색")

search_keyword = st.sidebar.text_input(
    "정류소명",
    placeholder="예: KBS / 가경 / 내덕",
)

if search_keyword.strip():
    filtered_stops = stops[
        stops["정류소명"].str.contains(
            search_keyword.strip(),
            case=False,
            na=False,
        )
    ].copy()
else:
    filtered_stops = stops.copy()

st.sidebar.write(
    f"검색 결과: **{len(filtered_stops):,}개**"
)


st.sidebar.markdown("---")
st.sidebar.subheader("🏙️ 미래 도시변화")

prediction_year = st.sidebar.selectbox(
    "예측 연도",
    [2027, 2028, 2029, 2030],
    index=1,
)

selected_factors = []

for factor in FUTURE_FACTORS:
    if st.sidebar.checkbox(factor):
        selected_factors.append(factor)

future_impact = calculate_future_impact(
    selected_factors
)

st.sidebar.metric(
    "미래 이동수요 영향",
    f"+{future_impact}%",
)


st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ 지도 설정")

map_mode = st.sidebar.radio(
    "표시 범위",
    [
        "검색 결과",
        "청주시 전체",
    ],
    index=0 if search_keyword.strip() else 1,
)

max_markers = st.sidebar.slider(
    "최대 정류장 표시 수",
    min_value=100,
    max_value=1000,
    value=500,
    step=100,
)


# ============================================================
# 6. 메인 화면
# ============================================================

st.title(
    "🚌 AI 기반 미래예측형 "
    "대중교통 의사결정 지원 플랫폼"
)

st.caption(
    "공모전용 PoC · 청주시 실제 정류장 데이터 연동"
)

st.info(
    "현재 플랫폼은 청주시 버스정보시스템의 실제 정류장 "
    "위치 데이터를 기반으로 미래 도시변화와 "
    "대중교통 정책 시나리오를 검토합니다."
)


# ============================================================
# 7. KPI
# ============================================================

st.markdown("## ① 현황 데이터")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "전체 정류장",
    f"{len(stops):,}개",
)

kpi2.metric(
    "검색 정류장",
    f"{len(filtered_stops):,}개",
)

kpi3.metric(
    "예측 연도",
    f"{prediction_year}년",
)

kpi4.metric(
    "미래 수요 영향",
    f"+{future_impact}%",
)

st.caption(
    "원본 데이터의 주요 항목: "
    "서비스ID · 정류소명 · 좌표(X) · 좌표(Y)"
)


# ============================================================
# 8. 정류장 검색 결과
# ============================================================

st.markdown("### 🔎 실제 정류장 검색")

display_table = filtered_stops[
    [
        "서비스ID",
        "정류소명",
        "경도",
        "위도",
    ]
].head(100)

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 9. 실제 정류장 지도
# ============================================================

st.markdown("---")
st.markdown("## ② 실제 청주시 정류장 지도")

if map_mode == "검색 결과":
    map_data = filtered_stops.copy()
else:
    map_data = stops.copy()

if len(map_data) > max_markers:
    map_data = map_data.sample(
        n=max_markers,
        random_state=42,
    )

center = (
    float(stops["위도"].mean()),
    float(stops["경도"].mean()),
)

stop_map = create_stop_map(
    map_data,
    center=center,
)

st_folium(
    stop_map,
    width=None,
    height=600,
)


# ============================================================
# 10. 미래 도시변화 기반 AI 노선 후보
# ============================================================

st.markdown("---")
st.markdown("## ③ 미래 도시변화 기반 AI 노선 후보")

st.warning(
    "현재 정류장 CSV에는 노선번호와 정류장 순서가 없습니다. "
    "따라서 아래 결과는 실제 운행노선이 아니라 "
    "'실제 정류장 위치를 활용한 AI 노선 후보 시뮬레이션'입니다."
)


if len(stops) < 2:
    st.error("노선 후보 분석에 필요한 정류장 데이터가 부족합니다.")
else:
    # 정류장 선택용 표시명
    stop_options = (
        stops[
            [
                "서비스ID",
                "정류소명",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    stop_options["표시명"] = (
        stop_options["정류소명"]
        + "  | ID: "
        + stop_options["서비스ID"]
    )

    origin_label, destination_label = st.columns(2)

    with origin_label:
        origin_option = st.selectbox(
            "출발 정류장",
            stop_options["표시명"].tolist(),
            index=0,
        )

    with destination_label:
        destination_index = (
            1 if len(stop_options) > 1 else 0
        )

        destination_option = st.selectbox(
            "도착 정류장",
            stop_options["표시명"].tolist(),
            index=destination_index,
        )

    origin_id = origin_option.split(" | ID: ")[-1]
    destination_id = destination_option.split(" | ID: ")[-1]

    origin = stops[
        stops["서비스ID"] == origin_id
    ].iloc[0]

    destination = stops[
        stops["서비스ID"] == destination_id
    ].iloc[0]

    origin_point = (
        float(origin["위도"]),
        float(origin["경도"]),
    )

    destination_point = (
        float(destination["위도"]),
        float(destination["경도"]),
    )

    direct_distance = haversine_km(
        origin_point,
        destination_point,
    )

    # PoC용 수요지수
    demand_index = round(
        100 * (1 + future_impact / 100),
        1,
    )

    # 평균 25km/h라는 PoC 가정
    estimated_minutes = round(
        direct_distance / 25 * 60,
        1,
    )

    result1, result2, result3 = st.columns(3)

    result1.metric(
        "직선거리",
        f"{direct_distance:.2f} km",
    )

    result2.metric(
        "예상 통행시간",
        f"{estimated_minutes:.1f}분",
    )

    result3.metric(
        "미래 수요지수",
        f"{demand_index:.1f}",
    )

    # 출발·도착 중간 지점 주변의 정류장 후보
    middle_latitude = (
        origin["위도"]
        + destination["위도"]
    ) / 2

    middle_longitude = (
        origin["경도"]
        + destination["경도"]
    ) / 2

    nearby = nearest_stops(
        stops,
        middle_latitude,
        middle_longitude,
        count=6,
    )

    st.markdown("### 🧠 AI 후보 노선")

    st.write(
        "출발·도착 정류장 사이의 생활권 정류장을 "
        "중간 후보로 선택하여 경로를 구성합니다."
    )

    candidate_points = [origin_point]

    for _, candidate in nearby.iterrows():
        candidate_id = str(candidate["서비스ID"])

        if candidate_id in {
            origin_id,
            destination_id,
        }:
            continue

        candidate_points.append(
            (
                float(candidate["위도"]),
                float(candidate["경도"]),
            )
        )

        # 출발 + 중간 후보 2개 + 도착 구조
        if len(candidate_points) >= 3:
            break

    candidate_points.append(destination_point)

    candidate_distance = sum(
        haversine_km(
            candidate_points[index],
            candidate_points[index + 1],
        )
        for index in range(
            len(candidate_points) - 1
        )
    )

    st.metric(
        "AI 후보 노선 총 거리",
        f"{candidate_distance:.2f} km",
    )

    candidate_map = create_candidate_route_map(
        candidate_points
    )

    st_folium(
        candidate_map,
        width=None,
        height=500,
    )


# ============================================================
# 11. 실제 버스노선 데이터 연동
# ============================================================

st.markdown("---")
st.markdown("## ④ 실제 버스노선 연동")

route_data = load_route_data()

if route_data is None:
    st.info(
        f"현재 '{ROUTE_FILE.name}' 파일이 없습니다. "
        "실제 노선을 연결하려면 아래 형태의 CSV를 추가하세요."
    )

    example_route = pd.DataFrame(
        {
            "노선번호": ["101", "101", "101"],
            "정류장순번": [1, 2, 3],
            "서비스ID": [
                "1001",
                "1002",
                "1003",
            ],
        }
    )

    st.dataframe(
        example_route,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "서비스ID는 현재 청주시 정류장 데이터의 "
        "서비스ID와 연결합니다."
    )

else:
    st.success(
        f"'{ROUTE_FILE.name}'을 발견했습니다."
    )

    st.dataframe(
        route_data.head(100),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 12. 행정 의사결정 지원
# ============================================================

st.markdown("---")
st.markdown("## ⑤ 행정 의사결정 지원")

decision_left, decision_right = st.columns(2)

with decision_left:
    st.subheader("정책 대안")

    policy = st.radio(
        "검토할 정책을 선택하세요.",
        [
            "현행 노선 유지",
            "노선 변경",
            "배차 간격 조정",
            "환승체계 개선",
            "신규 개발지역 연계",
        ],
    )

with decision_right:
    st.subheader("행정기관 판단")

    decision = st.selectbox(
        "최종 판단",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 데이터 검토",
        ],
    )


if decision == "정책 대안 채택":
    st.success(
        f"선택된 정책 대안: **{policy}**"
    )

elif decision == "추가 데이터 검토":
    st.warning(
        "교통카드·BIS·도시계획 등 추가 데이터를 "
        "검토한 후 재평가합니다."
    )

else:
    st.info(
        "AI 분석 결과는 의사결정 지원 자료이며, "
        "최종 정책은 행정기관이 결정합니다."
    )


# ============================================================
# 13. 시민 피드백
# ============================================================

st.markdown("---")
st.markdown("## ⑥ 정책 시행 후 시민 피드백")

feedback1, feedback2, feedback3 = st.columns(3)

with feedback1:
    satisfaction = st.slider(
        "전체 만족도",
        min_value=1,
        max_value=5,
        value=4,
    )

with feedback2:
    waiting = st.slider(
        "대기시간 만족도",
        min_value=1,
        max_value=5,
        value=4,
    )

with feedback3:
    transfer = st.slider(
        "환승 편의 만족도",
        min_value=1,
        max_value=5,
        value=4,
    )

citizen_comment = st.text_area(
    "시민 의견",
    placeholder="예: 출퇴근 시간 배차 간격을 줄여주세요.",
)

feedback_average = round(
    (
        satisfaction
        + waiting
        + transfer
    ) / 3,
    2,
)

st.metric(
    "시민 종합 만족도",
    f"{feedback_average:.2f} / 5",
)

if st.button("📊 시민 피드백 분석"):
    if feedback_average >= 4:
        st.success(
            "시민 만족도가 높습니다. "
            "현 정책의 유지·확대를 검토할 수 있습니다."
        )

    elif feedback_average >= 3:
        st.warning(
            "시민 만족도가 보통입니다. "
            "대기시간과 환승체계 등의 부분 개선을 검토합니다."
        )

    else:
        st.error(
            "시민 만족도가 낮습니다. "
            "노선 및 배차 운영의 재검토가 필요합니다."
        )

    if citizen_comment.strip():
        st.write(
            f"**시민 의견:** {citizen_comment}"
        )


# ============================================================
# 14. 전체 정책 순환 구조
# ============================================================

st.markdown("---")
st.markdown("## ⑦ 데이터 기반 정책 순환")

st.markdown(
    """
### 정책 시행
↓
### 운영 데이터 수집
↓
### 시민 피드백 수집
↓
### AI 분석·학습
↓
### 미래 수요 예측
↓
### 노선·배차 대안 개선
↓
### 정책 재시행
"""
)


# ============================================================
# 15. 데이터 한계 및 향후 확장
# ============================================================

with st.expander("📌 현재 데이터의 한계와 향후 확장"):

    st.write(
        """
현재 제공된 청주시 CSV는 실제 정류장의
서비스ID, 정류소명, 좌표(X/Y)를 제공합니다.

따라서 실제 정류장 위치를 지도에 표시하고
정류장 기반의 후보 분석을 수행할 수 있습니다.

다만 현재 CSV만으로는 다음 정보가 없습니다.

- 노선번호
- 노선별 정류장 순서
- 교통카드 승하차량
- 시간대별 수요
- 환승량
- 배차간격
- 통행시간

따라서 현재 버전은 '정류장 위치 기반 PoC'입니다.

향후 실제 노선 및 수요 데이터를 추가하면
다음 단계로 확장할 수 있습니다.

실제 노선
→ 승하차 수요 분석
→ 미래 개발지역 수요 예측
→ 노선 대안 생성
→ 기존 노선과 대안 비교
→ 시민 피드백
→ 정책 재평가
"""
    )
