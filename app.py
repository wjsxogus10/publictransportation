"""
🚌 AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼
청주시 실제 정류장 데이터 기반 공모전용 PoC

페이지 구성
1. 🏠 대시보드
2. 🗺️ 청주시 정류장
3. 🏙️ 미래 도시변화
4. 🤖 AI 수요·노선 분석
5. 📊 정책 대안 비교
6. 🏛️ 행정 의사결정
7. 💬 시민 피드백

실행:
streamlit run app.py
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
    initial_sidebar_state="expanded",
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
# 2. 공통 함수
# ============================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    """한글 CSV 인코딩을 자동으로 확인합니다."""
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as error:
            last_error = error

    raise last_error


@st.cache_data
def load_stops():
    """청주시 실제 정류장 데이터를 불러옵니다."""

    if not DATA_FILE.exists():
        return None, (
            f"'{DATA_FILE.name}' 파일을 찾을 수 없습니다."
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
        subset=["위도", "경도"]
    ).copy()

    data = data[
        data["위도"].between(33, 39)
        & data["경도"].between(124, 132)
    ].reset_index(drop=True)

    return data, None


@st.cache_data
def load_routes():
    """선택적으로 실제 노선 데이터를 불러옵니다."""
    if not ROUTE_FILE.exists():
        return None

    try:
        return read_csv_auto(ROUTE_FILE)
    except Exception:
        return None


def haversine_km(a, b):
    """두 좌표의 직선거리를 km로 계산합니다."""

    lat1, lon1 = a
    lat2, lon2 = b

    radius = 6371.0

    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)

    value = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(d_lon / 2) ** 2
    )

    value = max(0, min(1, value))

    return 2 * radius * atan2(
        sqrt(value),
        sqrt(1 - value),
    )


def future_impact(selected):
    """선택한 미래 도시변화의 PoC 영향지수를 계산합니다."""
    return min(
        50,
        5 + sum(
            FUTURE_FACTORS[item]
            for item in selected
        ),
    )


def nearest_stops(data, lat, lon, count=5):
    """특정 좌표 주변 정류장을 찾습니다."""

    temp = data.copy()

    temp["_distance"] = (
        (temp["위도"] - lat) ** 2
        + (temp["경도"] - lon) ** 2
    ) ** 0.5

    return temp.nsmallest(
        count,
        "_distance",
    ).drop(columns="_distance")


def inject_css():
    """공모전 발표용 웹 UI 스타일."""

    st.markdown(
        """
        <style>

        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .sub-title {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .section-card {
            padding: 1.2rem 1.4rem;
            border-radius: 14px;
            border: 1px solid #e6e6e6;
            background: #ffffff;
            margin-bottom: 1rem;
        }

        .step-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #e8e8e8;
            background: #fafafa;
            text-align: center;
            min-height: 110px;
        }

        .small-text {
            color: #777;
            font-size: 0.9rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(number, title, description):
    """각 페이지 공통 제목."""
    st.markdown(
        f'<div class="main-title">{number}. {title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-title">{description}</div>',
        unsafe_allow_html=True,
    )
    st.divider()


# ============================================================
# 3. 데이터 로드
# ============================================================

stops, error_message = load_stops()

if stops is None:
    st.error("청주시 정류장 데이터를 불러올 수 없습니다.")
    st.code(error_message)
    st.stop()


# ============================================================
# 4. 세션 상태
# ============================================================

if "future_factors" not in st.session_state:
    st.session_state.future_factors = []

if "prediction_year" not in st.session_state:
    st.session_state.prediction_year = 2028

if "selected_policy" not in st.session_state:
    st.session_state.selected_policy = "현행 노선 유지"


# ============================================================
# 5. CSS
# ============================================================

inject_css()


# ============================================================
# 6. 사이드바 - 페이지 이동
# ============================================================

st.sidebar.markdown("## 🚌 AI 대중교통 플랫폼")
st.sidebar.caption("청주시 적용 공모전 PoC")

page = st.sidebar.radio(
    "페이지",
    [
        "🏠 대시보드",
        "🗺️ 청주시 정류장",
        "🏙️ 미래 도시변화",
        "🤖 AI 수요·노선 분석",
        "📊 정책 대안 비교",
        "🏛️ 행정 의사결정",
        "💬 시민 피드백",
    ],
)

st.sidebar.divider()

st.sidebar.metric(
    "실제 정류장",
    f"{len(stops):,}개",
)

st.sidebar.caption(
    "실제 정류장 CSV 기반"
)


# ============================================================
# PAGE 1. 대시보드
# ============================================================

if page == "🏠 대시보드":

    page_header(
        "01",
        "대시보드",
        "청주시의 현재 교통 인프라와 미래 정책 분석 현황을 한눈에 확인합니다.",
    )

    impact = future_impact(
        st.session_state.future_factors
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "실제 정류장",
        f"{len(stops):,}개",
    )

    k2.metric(
        "예측 연도",
        f"{st.session_state.prediction_year}년",
    )

    k3.metric(
        "미래 수요 영향",
        f"+{impact}%",
    )

    k4.metric(
        "분석 단계",
        "PoC",
    )

    st.markdown("### 🔄 플랫폼 분석 흐름")

    flow = [
        ("01", "현재 데이터", "실제 정류장·공간정보"),
        ("02", "미래 변화", "개발·철도·산업·상업"),
        ("03", "AI 분석", "미래 수요·노선 후보"),
        ("04", "정책 비교", "효과·비용·편의"),
        ("05", "행정 결정", "최종 정책 선택"),
    ]

    cols = st.columns(5)

    for column, item in zip(cols, flow):
        with column:
            st.markdown(
                f"""
                <div class="step-card">
                    <b>{item[0]}</b><br>
                    <strong>{item[1]}</strong><br>
                    <span class="small-text">{item[2]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 💡 핵심 아이디어")

    st.markdown(
        """
        > **과거·현재·미래의 도시 데이터를 통합하고,  
        > AI를 통해 미래 이동수요와 대중교통 정책 대안을 분석하여  
        > 행정기관의 의사결정을 지원한다.**
        """
    )

    st.markdown("### 📌 현재 구현 범위")

    left, right = st.columns(2)

    with left:
        st.success(
            """
            **현재 구현**

            - 청주시 실제 정류장 데이터
            - 정류장 검색
            - 실제 좌표 기반 지도
            - 미래 도시변화 시나리오
            - AI 노선 후보 분석
            """
        )

    with right:
        st.info(
            """
            **향후 연계**

            - 실제 버스노선
            - 교통카드 승하차 데이터
            - BIS 운행정보
            - 환승 데이터
            - 도시계획·개발사업 데이터
            """
        )


# ============================================================
# PAGE 2. 청주시 정류장
# ============================================================

elif page == "🗺️ 청주시 정류장":

    page_header(
        "02",
        "청주시 실제 정류장",
        "실제 청주시 버스정보시스템 정류장 위치를 검색하고 지도에서 확인합니다.",
    )

    search = st.text_input(
        "🔎 정류소 검색",
        placeholder="예: KBS / 가경 / 내덕 / 터미널",
    )

    if search.strip():
        result = stops[
            stops["정류소명"].str.contains(
                search.strip(),
                case=False,
                na=False,
            )
        ].copy()
    else:
        result = stops.copy()

    c1, c2 = st.columns(2)

    c1.metric(
        "검색 결과",
        f"{len(result):,}개",
    )

    c2.metric(
        "전체 정류장",
        f"{len(stops):,}개",
    )

    st.markdown("### 📋 정류장 목록")

    st.dataframe(
        result[
            [
                "서비스ID",
                "정류소명",
                "경도",
                "위도",
            ]
        ].head(200),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🗺️ 지도")

    max_count = st.slider(
        "지도에 표시할 최대 정류장",
        100,
        1000,
        500,
        100,
    )

    map_data = result.copy()

    if len(map_data) > max_count:
        map_data = map_data.sample(
            max_count,
            random_state=42,
        )

    center = (
        float(stops["위도"].mean()),
        float(stops["경도"].mean()),
    )

    m = folium.Map(
        location=center,
        zoom_start=11,
        tiles="CartoDB positron",
    )

    for _, row in map_data.iterrows():

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
                f"""
                <b>{row['정류소명']}</b><br>
                서비스ID: {row['서비스ID']}<br>
                위도: {row['위도']:.6f}<br>
                경도: {row['경도']:.6f}
                """,
                max_width=280,
            ),
        ).add_to(m)

    st_folium(
        m,
        width=None,
        height=600,
    )


# ============================================================
# PAGE 3. 미래 도시변화
# ============================================================

elif page == "🏙️ 미래 도시변화":

    page_header(
        "03",
        "미래 도시변화 시나리오",
        "도시계획 및 개발사업에 따른 미래 이동수요 변화를 가정합니다.",
    )

    st.markdown(
        """
        ### 왜 미래 도시변화를 보는가?

        기존 대중교통 계획은 과거 이용수요를 중심으로 분석하는 경우가 많습니다.
        이 플랫폼은 **미래에 발생할 도시 변화를 함께 입력하여**
        대중교통 수요와 정책 대안을 사전에 검토하는 것을 목표로 합니다.
        """
    )

    year = st.selectbox(
        "📅 분석 대상 연도",
        [2027, 2028, 2029, 2030],
        index=[2027, 2028, 2029, 2030].index(
            st.session_state.prediction_year
        ),
    )

    st.session_state.prediction_year = year

    st.markdown("### 🏗️ 미래 도시변화 선택")

    selected = []

    cols = st.columns(2)

    factors = list(FUTURE_FACTORS.keys())

    for index, factor in enumerate(factors):

        with cols[index % 2]:

            checked = st.checkbox(
                factor,
                value=factor in st.session_state.future_factors,
            )

            if checked:
                selected.append(factor)

    st.session_state.future_factors = selected

    impact = future_impact(selected)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "예측 연도",
        f"{year}년",
    )

    c2.metric(
        "선택된 도시변화",
        f"{len(selected)}개",
    )

    c3.metric(
        "PoC 미래수요 영향",
        f"+{impact}%",
    )

    st.markdown("### 📊 시나리오 결과")

    if selected:

        scenario_table = pd.DataFrame(
            {
                "미래 도시변화": selected,
                "PoC 영향지수": [
                    f"+{FUTURE_FACTORS[item]}%"
                    for item in selected
                ],
            }
        )

        st.dataframe(
            scenario_table,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "미래 도시변화 요소를 하나 이상 선택하세요."
        )


# ============================================================
# PAGE 4. AI 수요·노선 분석
# ============================================================

elif page == "🤖 AI 수요·노선 분석":

    page_header(
        "04",
        "AI 수요·노선 분석",
        "실제 청주시 정류장 위치를 기반으로 미래 도시변화에 따른 노선 후보를 분석합니다.",
    )

    st.warning(
        "현재 정류장 CSV에는 노선번호와 정류장 순서가 없습니다. "
        "따라서 아래 결과는 실제 운행노선이 아닌 "
        "**AI 노선 후보 시뮬레이션**입니다."
    )

    options = (
        stops[
            [
                "서비스ID",
                "정류소명",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    options["표시명"] = (
        options["정류소명"]
        + " | ID: "
        + options["서비스ID"]
    )

    c1, c2 = st.columns(2)

    with c1:
        origin_label = st.selectbox(
            "출발 정류장",
            options["표시명"].tolist(),
        )

    with c2:
        destination_label = st.selectbox(
            "도착 정류장",
            options["표시명"].tolist(),
            index=min(1, len(options) - 1),
        )

    origin_id = origin_label.split(" | ID: ")[-1]
    destination_id = destination_label.split(" | ID: ")[-1]

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

    impact = future_impact(
        st.session_state.future_factors
    )

    demand_index = round(
        100 * (1 + impact / 100),
        1,
    )

    estimated_time = round(
        direct_distance / 25 * 60,
        1,
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "직선거리",
        f"{direct_distance:.2f} km",
    )

    r2.metric(
        "예상 통행시간",
        f"{estimated_time:.1f}분",
    )

    r3.metric(
        "미래 수요지수",
        demand_index,
    )

    middle_lat = (
        origin["위도"]
        + destination["위도"]
    ) / 2

    middle_lon = (
        origin["경도"]
        + destination["경도"]
    ) / 2

    candidates = nearest_stops(
        stops,
        middle_lat,
        middle_lon,
        count=5,
    )

    points = [origin_point]

    for _, candidate in candidates.iterrows():

        if candidate["서비스ID"] in {
            origin_id,
            destination_id,
        }:
            continue

        points.append(
            (
                float(candidate["위도"]),
                float(candidate["경도"]),
            )
        )

        if len(points) >= 3:
            break

    points.append(destination_point)

    candidate_distance = sum(
        haversine_km(
            points[i],
            points[i + 1],
        )
        for i in range(len(points) - 1)
    )

    st.markdown("### 🧠 AI 후보 노선")

    st.metric(
        "후보 노선 총 거리",
        f"{candidate_distance:.2f} km",
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

    for i, point in enumerate(points):

        if i == 0:
            label = "출발 정류장"
        elif i == len(points) - 1:
            label = "도착 정류장"
        else:
            label = f"중간 후보 {i}"

        folium.Marker(
            location=point,
            tooltip=label,
        ).add_to(route_map)

    folium.PolyLine(
        points,
        color="red",
        weight=6,
        opacity=0.85,
        tooltip="AI 노선 후보",
    ).add_to(route_map)

    st_folium(
        route_map,
        width=None,
        height=550,
    )


# ============================================================
# PAGE 5. 정책 대안 비교
# ============================================================

elif page == "📊 정책 대안 비교":

    page_header(
        "05",
        "정책 대안 비교",
        "여러 정책 대안을 동일한 기준으로 비교하여 행정기관의 검토를 지원합니다.",
    )

    impact = future_impact(
        st.session_state.future_factors
    )

    # PoC 시뮬레이션 지표
    scenarios = pd.DataFrame(
        {
            "정책 대안": [
                "현행 노선 유지",
                "거점 직결형",
                "간선 연장형",
                "환승 최적화형",
            ],
            "예상 수요 변화": [
                0,
                12 + impact // 3,
                7 + impact // 4,
                10 + impact // 3,
            ],
            "통행시간 변화": [
                0,
                -18,
                -10,
                -15,
            ],
            "환승시간 변화": [
                0,
                -12,
                -7,
                -25,
            ],
            "운영비 변화": [
                0,
                8,
                5,
                4,
            ],
        }
    )

    # 간단한 종합점수
    scenarios["종합점수"] = (
        70
        + scenarios["예상 수요 변화"] * 0.8
        - scenarios["운영비 변화"] * 0.5
        - scenarios["통행시간 변화"].abs() * 0.3
    ).round(1)

    best = scenarios.loc[
        scenarios["종합점수"].idxmax(),
        "정책 대안",
    ]

    st.success(
        f"🤖 PoC 분석상 우선 검토 대안: **{best}**"
    )

    st.dataframe(
        scenarios,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 📌 해석")

    st.markdown(
        f"""
        현재 선택된 미래 도시변화 시나리오의 영향은
        **+{impact}%**로 설정되어 있습니다.

        PoC에서는 각 정책 대안의 수요·시간·운영비 변화를
        단순화한 가정으로 비교합니다.

        실제 서비스에서는 교통카드 승하차 데이터,
        BIS, 도로소통정보, 도시계획 정보를 활용하여
        검증된 교통수요예측 모델로 대체합니다.
        """
    )


# ============================================================
# PAGE 6. 행정 의사결정
# ============================================================

elif page == "🏛️ 행정 의사결정":

    page_header(
        "06",
        "행정 의사결정",
        "AI가 정책을 자동 결정하는 것이 아니라 행정기관의 정책 판단을 지원합니다.",
    )

    st.markdown(
        """
        ### 🧑‍💼 의사결정 구조

        **AI 분석**
        → **정책 대안 비교**
        → **전문가 검토**
        → **행정기관 판단**
        → **정책 시행**
        """
    )

    st.divider()

    policies = [
        "현행 노선 유지",
        "노선 변경",
        "배차 간격 조정",
        "환승체계 개선",
        "신규 개발지역 연계",
    ]

    selected_policy = st.radio(
        "검토할 정책 대안",
        policies,
    )

    st.session_state.selected_policy = selected_policy

    decision = st.selectbox(
        "행정기관 최종 판단",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 데이터 검토",
        ],
    )

    if decision == "정책 대안 채택":
        st.success(
            f"최종 선택안: **{selected_policy}**"
        )

    elif decision == "추가 데이터 검토":
        st.warning(
            "추가 교통자료와 전문가 검토 후 "
            "정책을 재평가합니다."
        )

    else:
        st.info(
            "AI 분석 결과는 참고자료이며 "
            "최종 결정권은 행정기관에 있습니다."
        )

    st.markdown("### 🔁 정책 시행 이후")

    st.markdown(
        """
        정책 시행
        ↓
        운영 데이터 수집
        ↓
        시민 피드백
        ↓
        AI 분석
        ↓
        다음 정책 개선
        """
    )


# ============================================================
# PAGE 7. 시민 피드백
# ============================================================

elif page == "💬 시민 피드백":

    page_header(
        "07",
        "시민 피드백",
        "정책 시행 이후 시민의 실제 경험을 다음 정책 수립에 반영합니다.",
    )

    st.markdown(
        """
        ### 시민 의견이 왜 필요한가?

        교통정책은 데이터만으로 모든 이용자의 경험을 설명하기 어렵습니다.
        따라서 정책 시행 이후 시민의 만족도와 의견을 수집하고,
        이를 다음 정책 개선의 입력자료로 활용하는 구조를 제안합니다.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        satisfaction = st.slider(
            "전체 만족도",
            1,
            5,
            4,
        )

    with c2:
        waiting = st.slider(
            "대기시간 만족도",
            1,
            5,
            4,
        )

    with c3:
        transfer = st.slider(
            "환승 편의 만족도",
            1,
            5,
            4,
        )

    comment = st.text_area(
        "시민 의견",
        placeholder="예: 출퇴근 시간 배차 간격을 줄여주세요.",
    )

    average = round(
        (
            satisfaction
            + waiting
            + transfer
        ) / 3,
        2,
    )

    st.metric(
        "시민 종합 만족도",
        f"{average:.2f} / 5",
    )

    if st.button(
        "📊 피드백 분석",
        use_container_width=True,
    ):

        if average >= 4:
            st.success(
                "만족도가 높습니다. "
                "현 정책의 유지·확대를 검토합니다."
            )

        elif average >= 3:
            st.warning(
                "만족도가 보통입니다. "
                "대기시간과 환승체계 개선을 검토합니다."
            )

        else:
            st.error(
                "만족도가 낮습니다. "
                "노선·배차 정책의 재검토가 필요합니다."
            )

        if comment.strip():
            st.write(
                f"**시민 의견:** {comment}"
            )

    st.markdown("---")

    st.markdown(
        """
        ### 🔄 최종 정책 순환 구조

        | 단계 | 주요 내용 |
        |---|---|
        | ① 데이터 | 실제 교통·도시 데이터 |
        | ② 예측 | 미래 이동수요 분석 |
        | ③ 시뮬레이션 | 노선·배차 대안 생성 |
        | ④ 비교 | 효과·비용 비교 |
        | ⑤ 결정 | 행정기관 최종 판단 |
        | ⑥ 시행 | 실제 정책 적용 |
        | ⑦ 피드백 | 시민·운영 데이터 수집 |
        | ⑧ 개선 | 다음 정책에 반영 |
        """
    )


# ============================================================
# 7. 하단 공통 안내
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "※ 현재 버전은 실제 청주시 정류장 위치를 활용한 PoC입니다."
)

st.sidebar.caption(
    "※ 실제 노선번호·노선순서·교통카드 수요 데이터가 "
    "추가되면 노선 단위 분석으로 확장할 수 있습니다."
)
