import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

# ============================================================
# AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼
# v3 - 청주시 실제 정류장 데이터 연동 PoC
#
# 기본 데이터:
# 충청북도_청주시_버스정보시스템_20250401.csv
#
# 이 CSV에는 정류장 ID/명칭/좌표가 있으므로 실제 정류장 위치를
# 지도에 표시할 수 있습니다.
#
# 주의:
# 현재 CSV에는 "노선번호 + 노선별 정류장 순서"가 없으므로
# 실제 버스노선을 임의로 만들어 표시하지 않습니다.
# 실제 노선 데이터가 추가되면 route.csv를 통해 연결할 수 있습니다.
# ============================================================

st.set_page_config(
    page_title="AI 미래예측형 대중교통 의사결정 지원",
    page_icon="🚌",
    layout="wide",
)

DATA_FILE = "충청북도_청주시_버스정보시스템_20250401.csv"
ROUTE_FILE = "bus_routes.csv"

st.title("🚌 AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼")
st.caption("공모전용 PoC · 청주시 실제 정류장 데이터 연동")

st.info(
    "현재 플랫폼은 GitHub에 등록한 청주시 실제 버스정보시스템 정류장 데이터를 "
    "기본 지도 데이터로 사용합니다. 정류장 데이터에는 서비스ID, 정류소명, "
    "X/Y 좌표가 포함되어 있습니다."
)


# ============================================================
# 기본 함수
# ============================================================

def haversine_km(a, b):
    lat1, lon1 = a
    lat2, lon2 = b

    r = 6371.0

    p1 = radians(lat1)
    p2 = radians(lat2)

    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    value = (
        sin(dphi / 2) ** 2
        + cos(p1)
        * cos(p2)
        * sin(dlambda / 2) ** 2
    )

    value = max(0.0, min(1.0, value))

    return (
        2
        * r
        * atan2(
            sqrt(value),
            sqrt(1 - value),
        )
    )


def read_csv_auto(path):
    encodings = [
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "utf-8",
    ]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
            )
        except Exception as exc:
            last_error = exc

    raise last_error


def load_real_stops():
    # GitHub/Streamlit Cloud의 현재 앱 폴더에서 찾음
    path = Path(DATA_FILE)

    if not path.exists():
        return None, (
            f"'{DATA_FILE}' 파일을 app.py와 같은 폴더에 "
            "업로드해야 합니다."
        )

    try:
        data = read_csv_auto(path)
    except Exception as exc:
        return None, f"CSV 읽기 오류: {exc}"

    required = {
        "서비스ID",
        "정류소명",
        "좌표(X)",
        "좌표(Y)",
    }

    missing = required - set(data.columns)

    if missing:
        return None, (
            "필수 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    data = data.copy()

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

    data["경도"] = pd.to_numeric(
        data["좌표(X)"],
        errors="coerce",
    )

    data["위도"] = pd.to_numeric(
        data["좌표(Y)"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "위도",
            "경도",
        ]
    ).copy()

    # 대한민국 인근 좌표만 남겨 비정상 데이터 방지
    data = data[
        data["위도"].between(33, 39)
        & data["경도"].between(124, 132)
    ].copy()

    data = data.reset_index(drop=True)

    return data, None


def load_route_file():
    path = Path(ROUTE_FILE)

    if not path.exists():
        return None

    try:
        return read_csv_auto(path)
    except Exception:
        return None


def nearest_stops(df, lat, lon, n=8):
    temp = df.copy()

    temp["_거리(km)"] = (
        (
            (temp["위도"] - lat) ** 2
            + (temp["경도"] - lon) ** 2
        ) ** 0.5
    )

    return temp.nsmallest(
        n,
        "_거리(km)",
    ).copy()


# ============================================================
# 실제 청주시 정류장 데이터 로드
# ============================================================

stops, load_error = load_real_stops()

if stops is None:
    st.error(
        "청주시 실제 정류장 CSV를 불러오지 못했습니다."
    )
    st.code(load_error)
    st.stop()


# ============================================================
# 사이드바
# ============================================================

st.sidebar.header("🗺️ 실제 청주시 정류장")

st.sidebar.metric(
    "불러온 정류장 데이터",
    f"{len(stops):,}개",
)


search = st.sidebar.text_input(
    "정류소명 검색",
    placeholder="예: KBS, 가경, 내덕",
)


if search.strip():
    filtered = stops[
        stops["정류소명"]
        .str.contains(
            search.strip(),
            case=False,
            na=False,
        )
    ].copy()
else:
    filtered = stops.copy()


st.sidebar.write(
    f"검색 결과: **{len(filtered):,}개**"
)


# ============================================================
# 미래 도시변화 시나리오
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🏙️ 미래 도시변화")

year = st.sidebar.selectbox(
    "예측 연도",
    [2027, 2028, 2029, 2030],
    index=1,
)

future_factors = {
    "신규 공동주택 개발": 8,
    "산업단지 조성·확장": 7,
    "철도 개통·환승거점 강화": 6,
    "대규모 상업시설 개발": 5,
    "대규모 행사·관광객 증가": 4,
}

selected_factors = []

for factor, impact in future_factors.items():

    if st.sidebar.checkbox(
        factor,
        value=False,
    ):
        selected_factors.append(
            (factor, impact)
        )


future_impact = 5 + sum(
    impact
    for _, impact in selected_factors
)

future_impact = min(
    future_impact,
    50,
)

st.sidebar.metric(
    "미래 이동수요 영향",
    f"+{future_impact}%",
)


# ============================================================
# 지도 설정
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📍 지도 표시")

display_mode = st.sidebar.radio(
    "정류장 표시 범위",
    [
        "검색 결과",
        "청주시 전체",
    ],
    index=0 if search.strip() else 1,
)

max_markers = st.sidebar.slider(
    "최대 표시 정류장 수",
    100,
    1000,
    500,
    100,
)


# ============================================================
# 메인 KPI
# ============================================================

st.markdown("## ① 실제 청주시 정류장 데이터")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "전체 정류장",
    f"{len(stops):,}개",
)

c2.metric(
    "검색 정류장",
    f"{len(filtered):,}개",
)

c3.metric(
    "예측 연도",
    f"{year}년",
)

c4.metric(
    "미래 도시변화 영향",
    f"+{future_impact}%",
)


st.caption(
    "※ 현재 업로드된 원본 CSV의 컬럼은 "
    "서비스ID·정류소명·좌표(X)·좌표(Y)입니다."
)


# ============================================================
# 정류장 검색 결과
# ============================================================

st.markdown("### 🔎 실제 정류장 검색")

show_table = filtered[
    [
        "서비스ID",
        "정류소명",
        "경도",
        "위도",
    ]
].head(100)

st.dataframe(
    show_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 지도
# ============================================================

st.markdown("---")
st.markdown("## ② 실제 청주시 정류장 지도")


if display_mode == "검색 결과":
    map_data = filtered.copy()
else:
    map_data = stops.copy()


if len(map_data) > max_markers:

    # 검색 결과를 우선적으로 표시하고,
    # 전체 데이터일 경우 균등하게 샘플링
    map_data = map_data.sample(
        n=max_markers,
        random_state=42,
    )


center_lat = float(
    stops["위도"].mean()
)

center_lon = float(
    stops["경도"].mean()
)


m = folium.Map(
    location=[
        center_lat,
        center_lon,
    ],
    zoom_start=11,
    tiles="CartoDB positron",
)


for _, row in map_data.iterrows():

    popup = folium.Popup(
        (
            f"<b>{row['정류소명']}</b><br>"
            f"서비스ID: {row['서비스ID']}<br>"
            f"경도: {row['경도']:.6f}<br>"
            f"위도: {row['위도']:.6f}"
        ),
        max_width=280,
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
        popup=popup,
        tooltip=row["정류소명"],
    ).add_to(m)


st_folium(
    m,
    width=None,
    height=600,
)


# ============================================================
# 정류장 기반 AI 후보 분석
# ============================================================

st.markdown("---")
st.markdown("## ③ 미래 도시변화 기반 노선 후보 분석")

st.warning(
    "현재 CSV에는 노선번호와 노선별 정류장 순서가 포함되어 있지 않습니다. "
    "따라서 아래 결과는 '실제 버스노선'이 아니라 "
    "실제 정류장 위치를 활용한 **AI 노선 후보 시뮬레이션**입니다."
)


if len(stops) >= 2:

    col1, col2 = st.columns(2)

    with col1:

        origin_name = st.selectbox(
            "출발 정류장",
            stops["정류소명"].tolist(),
            index=0,
        )

    with col2:

        destination_name = st.selectbox(
            "도착 정류장",
            stops["정류소명"].tolist(),
            index=min(
                1,
                len(stops) - 1,
            ),
        )


    origin_candidates = stops[
        stops["정류소명"]
        == origin_name
    ]

    destination_candidates = stops[
        stops["정류소명"]
        == destination_name
    ]


    origin = origin_candidates.iloc[0]
    destination = destination_candidates.iloc[0]


    origin_coord = (
        float(origin["위도"]),
        float(origin["경도"]),
    )

    destination_coord = (
        float(destination["위도"]),
        float(destination["경도"]),
    )


    direct_distance = haversine_km(
        origin_coord,
        destination_coord,
    )


    # 미래 도시변화가 클수록 잠재적 수요가 증가하는
    # PoC용 가정
    demand_index = round(
        100 * (
            1
            + future_impact / 100
        ),
        1,
    )


    estimated_time = round(
        direct_distance
        / 25
        * 60,
        1,
    )


    if direct_distance < 1:

        st.info(
            "출발지와 도착지가 매우 가깝습니다."
        )

    else:

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "직선거리",
            f"{direct_distance:.2f} km",
        )

        p2.metric(
            "예상 통행시간",
            f"{estimated_time:.1f}분",
        )

        p3.metric(
            "미래 수요지수",
            f"{demand_index:.1f}",
        )


    nearby = nearest_stops(
        stops,
        (
            origin["위도"]
            + destination["위도"]
        ) / 2,
        (
            origin["경도"]
            + destination["경도"]
        ) / 2,
        n=6,
    )


    st.markdown(
        "### AI 후보 노선"
    )

    st.write(
        "선택한 실제 정류장을 출발·도착 거점으로 설정하고 "
        "중간 생활권 정류장을 후보로 탐색하는 PoC입니다."
    )


    candidate_points = [
        origin_coord,
    ]


    # 중간 후보 중 출발/도착과 동일하지 않은 정류장 추가
    for _, candidate in nearby.iterrows():

        if candidate["정류소명"] in [
            origin_name,
            destination_name,
        ]:
            continue

        candidate_points.append(
            (
                float(candidate["위도"]),
                float(candidate["경도"]),
            )
        )

        if len(candidate_points) >= 4:
            break


    candidate_points.append(
        destination_coord
    )


    candidate_distance = sum(
        haversine_km(
            candidate_points[i],
            candidate_points[i + 1],
        )
        for i in range(
            len(candidate_points) - 1
        )
    )


    st.write(
        f"**AI 후보 노선 거리:** "
        f"{candidate_distance:.2f} km"
    )


    route_map = folium.Map(
        location=[
            (
                origin["위도"]
                + destination["위도"]
            ) / 2,
            (
                origin["경도"]
                + destination["경도"]
            ) / 2,
        ],
        zoom_start=12,
        tiles="CartoDB positron",
    )


    for idx, point in enumerate(
        candidate_points
    ):

        label = (
            "출발"
            if idx == 0
            else (
                "도착"
                if idx == len(candidate_points) - 1
                else f"중간 후보 {idx}"
            )
        )

        folium.Marker(
            location=point,
            tooltip=label,
        ).add_to(route_map)


    folium.PolyLine(
        candidate_points,
        color="red",
        weight=6,
        opacity=0.85,
        tooltip="AI 노선 후보",
    ).add_to(route_map)


    st_folium(
        route_map,
        width=None,
        height=500,
    )


# ============================================================
# 실제 노선 데이터 연동 준비
# ============================================================

st.markdown("---")
st.markdown("## ④ 실제 버스노선 연동 상태")


route_data = load_route_file()


if route_data is None:

    st.info(
        "현재 GitHub 저장소에는 "
        f"'{ROUTE_FILE}'이 없어 실제 노선선형은 표시하지 않습니다."
    )

    st.write(
        "실제 노선을 연결하려면 다음 구조의 CSV를 추가하면 됩니다."
    )

    example = pd.DataFrame({
        "노선번호": ["101", "101", "101"],
        "정류장순번": [1, 2, 3],
        "서비스ID": [
            "1001",
            "1002",
            "1003",
        ],
    })

    st.dataframe(
        example,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "서비스ID는 현재 청주시 정류장 CSV의 서비스ID와 연결합니다."
    )

else:

    st.success(
        f"실제 노선 데이터 '{ROUTE_FILE}'를 발견했습니다."
    )

    st.dataframe(
        route_data.head(100),
        use_container_width=True,
    )


# ============================================================
# 행정 의사결정 지원
# ============================================================

st.markdown("---")
st.markdown("## ⑤ 행정 의사결정 지원")

decision_col1, decision_col2 = st.columns(2)


with decision_col1:

    st.subheader(
        "정책 대안"
    )

    policy = st.radio(
        "검토할 정책",
        [
            "현행 노선 유지",
            "노선 변경",
            "배차 간격 조정",
            "환승체계 개선",
            "신규 개발지역 연계",
        ],
    )


with decision_col2:

    st.subheader(
        "행정기관 판단"
    )

    decision = st.selectbox(
        "최종 결정",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 데이터 검토",
        ],
    )


if decision == "정책 대안 채택":

    st.success(
        f"최종 선택 정책: **{policy}**"
    )

elif decision == "추가 데이터 검토":

    st.warning(
        "교통카드·BIS·도시계획 등 추가 데이터 "
        "검토 후 재평가합니다."
    )

else:

    st.info(
        "AI 분석 결과를 참고하여 "
        "행정기관이 최종 결정합니다."
    )


# ============================================================
# 시민 피드백
# ============================================================

st.markdown("---")
st.markdown("## ⑥ 정책 시행 후 시민 피드백")

s1, s2, s3 = st.columns(3)

with s1:
    satisfaction = st.slider(
        "전체 만족도",
        1,
        5,
        4,
    )

with s2:
    waiting = st.slider(
        "대기시간 만족도",
        1,
        5,
        4,
    )

with s3:
    transfer = st.slider(
        "환승 편의 만족도",
        1,
        5,
        4,
    )


comment = st.text_area(
    "시민 의견",
    placeholder="예: 출퇴근 시간 배차를 줄여주세요.",
)


average_feedback = round(
    (
        satisfaction
        + waiting
        + transfer
    ) / 3,
    2,
)


st.metric(
    "시민 종합 만족도",
    f"{average_feedback:.2f} / 5",
)


if st.button(
    "📊 시민 피드백 분석"
):

    if average_feedback >= 4:

        st.success(
            "시민 만족도가 높습니다. "
            "현 정책의 유지·확대를 검토합니다."
        )

    elif average_feedback >= 3:

        st.warning(
            "시민 만족도가 보통입니다. "
            "대기시간·환승체계 등의 부분 개선을 검토합니다."
        )

    else:

        st.error(
            "시민 만족도가 낮습니다. "
            "노선·배차 운영의 재검토가 필요합니다."
        )

    if comment.strip():

        st.write(
            f"**시민 의견:** {comment}"
        )


# ============================================================
# 정책 순환
# ============================================================

st.markdown("---")
st.markdown("## ⑦ 정책 시행 → 데이터 → AI 개선")

st.markdown(
    """
**정책 시행**
→ **운영 데이터 수집**
→ **시민 피드백 수집**
→ **AI 분석·학습**
→ **다음 수요예측**
→ **노선·배차 대안 개선**
"""
)


with st.expander(
    "📌 데이터 및 모델의 한계"
):

    st.write(
        """
현재 제공된 청주시 CSV는 실제 정류장의 서비스ID,
정류소명, 좌표(X/Y)를 제공하므로 실제 정류장 위치를
지도에 반영할 수 있습니다.

반면 이 파일 자체에는 노선번호, 노선별 정류장 순서,
교통카드 승하차량, 환승량 등의 정보가 없습니다.

따라서 본 PoC에서는 실제 정류장 위치와
미래 도시변화 시나리오를 먼저 연결하고,
실제 노선·수요 데이터가 확보되면 노선 단위
수요예측 및 정책 시뮬레이션으로 확장하는 구조입니다.
        """
    )
