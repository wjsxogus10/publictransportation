"""
🚌 AI 기반 미래예측형 대중교통 계획·의사결정 지원 플랫폼
청주시 적용 공모전용 PoC

핵심 흐름
① 기초현황 → ② 이용현황 → ③ 수요분석 → ④ 문제진단
→ ⑤ 장래여건 → ⑥ 장래수요 → ⑦ 노선대안
→ ⑧ 배차계획 → ⑨ 환승체계 → ⑩ 대안평가
→ ⑪ 행정결정 → ⑫ 시행·모니터링

실행:
streamlit run app.py
"""

from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 미래예측형 대중교통 계획",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

STOP_FILE = Path("충청북도_청주시_버스정보시스템_20250401.csv")

STOP_COLUMNS = {
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

POLICIES = [
    "현행 유지",
    "노선 신설",
    "노선 연장",
    "노선 조정",
    "배차 증회",
    "환승 최적화",
]


# ============================================================
# 2. 공통 함수
# ============================================================

def read_csv_auto(path_or_file):
    """한글 CSV를 대표적인 인코딩으로 읽습니다."""
    errors = []

    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path_or_file, encoding=encoding)
        except Exception as error:
            errors.append(error)

    raise errors[-1]


@st.cache_data
def load_stops():
    """청주시 실제 정류장 데이터를 불러옵니다."""

    if not STOP_FILE.exists():
        return None, f"'{STOP_FILE.name}' 파일이 없습니다."

    try:
        data = read_csv_auto(STOP_FILE)
    except Exception as error:
        return None, f"CSV 읽기 오류: {error}"

    missing = STOP_COLUMNS - set(data.columns)

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


def load_population(uploaded_file):
    """지역별 인구 CSV를 읽습니다."""

    if uploaded_file is None:
        return None, None

    try:
        data = read_csv_auto(uploaded_file)
    except Exception as error:
        return None, f"인구 CSV 오류: {error}"

    required = {"지역명", "인구수(명)"}

    if not required.issubset(data.columns):
        return None, (
            "인구 CSV에는 '지역명'과 '인구수(명)' "
            "컬럼이 필요합니다."
        )

    data = data.copy()

    data["지역명"] = (
        data["지역명"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["인구수(명)"] = pd.to_numeric(
        data["인구수(명)"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["지역명", "인구수(명)"]
    ).reset_index(drop=True)

    return data, None


def load_demand(uploaded_file):
    """
    선택적 교통수요 CSV.
    기본 형식:
    정류소명, 일평균 승하차(건)
    """

    if uploaded_file is None:
        return None, None

    try:
        data = read_csv_auto(uploaded_file)
    except Exception as error:
        return None, f"수요 CSV 오류: {error}"

    required = {"정류소명", "일평균 승하차(건)"}

    if not required.issubset(data.columns):
        return None, (
            "수요 CSV에는 '정류소명'과 "
            "'일평균 승하차(건)' 컬럼이 필요합니다."
        )

    data = data.copy()

    data["정류소명"] = (
        data["정류소명"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["일평균 승하차(건)"] = pd.to_numeric(
        data["일평균 승하차(건)"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["정류소명", "일평균 승하차(건)"]
    ).reset_index(drop=True)

    return data, None


def haversine_km(a, b):
    """두 좌표 사이 직선거리를 km로 계산합니다."""

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

    value = max(0.0, min(1.0, value))

    return 2 * radius * atan2(
        sqrt(value),
        sqrt(1 - value),
    )


def future_impact(selected):
    """미래 도시변화의 PoC 수요 영향지수."""

    return min(
        50,
        5 + sum(FUTURE_FACTORS[item] for item in selected),
    )


def nearest_stops(data, lat, lon, count=5):
    """중간 생활권 주변 정류장 후보."""

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
    """웹페이지 가독성 개선."""

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: .2rem;
        }

        .sub-title {
            color: #666;
            font-size: 1.02rem;
            margin-bottom: 1rem;
        }

        .plan-card {
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1.1rem;
            background: #ffffff;
            margin-bottom: 1rem;
        }

        .step-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: .9rem;
            text-align: center;
            background: #fafafa;
            min-height: 105px;
        }

        .muted {
            color: #777;
            font-size: .88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(number, title, description):
    st.markdown(
        f'<div class="main-title">{number}. {title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-title">{description}</div>',
        unsafe_allow_html=True,
    )
    st.divider()


def build_policy_table(
    base_demand,
    future_rate,
    base_time,
):
    """대중교통 계획 대안 비교표를 생성합니다."""

    rows = []

    policy_settings = {
        "현행 유지": (0.00, 1.00, 1.00, 0),
        "노선 신설": (0.18, 0.88, 1.18, 10),
        "노선 연장": (0.12, 0.92, 1.10, 7),
        "노선 조정": (0.10, 0.90, 1.05, 4),
        "배차 증회": (0.08, 0.82, 1.15, 6),
        "환승 최적화": (0.11, 0.84, 0.90, 3),
    }

    for policy, setting in policy_settings.items():

        demand_gain, time_factor, operation_factor, cost = setting

        demand = base_demand * (
            1 + future_rate / 100 + demand_gain
        )

        travel_time = base_time * time_factor

        wait_time = (
            10
            * operation_factor
            * (1 - min(future_rate, 40) / 200)
        )

        transfer_time = (
            5
            * (
                0.75
                if policy == "환승 최적화"
                else 1.0
            )
        )

        accessibility = min(
            100,
            70
            + demand_gain * 100
            + (1 - time_factor) * 30,
        )

        score = (
            0.35 * (demand / max(base_demand, 1))
            + 0.25 * (1 / max(travel_time, 1))
            + 0.20 * (1 / max(wait_time, 1))
            + 0.20 * (1 / (1 + cost))
        )

        rows.append(
            [
                policy,
                int(round(demand)),
                round(travel_time, 1),
                round(wait_time, 1),
                round(transfer_time, 1),
                cost,
                round(accessibility, 1),
                round(score, 4),
            ]
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "정책 대안",
            "예상 일평균 수요(건)",
            "평균 통행시간(분)",
            "평균 대기시간(분)",
            "평균 환승시간(분)",
            "연간 추가 운영비(억원)",
            "접근성 지수",
            "종합점수",
        ],
    )

    return result


# ============================================================
# 3. 데이터 로드
# ============================================================

stops, stop_error = load_stops()

if stops is None:
    st.error(stop_error)
    st.stop()


# ============================================================
# 4. Session State
# ============================================================

defaults = {
    "population": None,
    "demand": None,
    "population_source": "미연결",
    "demand_source": "미연결",
    "future_factors": [],
    "prediction_year": 2030,
    "selected_policy": "현행 유지",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 5. CSS
# ============================================================

inject_css()


# ============================================================
# 6. 사이드바
# ============================================================

st.sidebar.markdown("## 🚌 대중교통 계획 플랫폼")
st.sidebar.caption("AI 기반 미래예측형 대중교통 계획·의사결정 지원")

page = st.sidebar.radio(
    "계획 단계",
    [
        "🏠 종합 대시보드",
        "01. 기초현황 분석",
        "02. 이용현황 분석",
        "03. 수요·문제 분석",
        "04. 장래여건 분석",
        "05. 장래수요 예측",
        "06. 노선 대안 계획",
        "07. 배차·환승 계획",
        "08. 대안 시뮬레이션",
        "09. 종합평가",
        "10. 행정 의사결정",
        "11. 시행·모니터링",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### 📂 데이터 입력")

population_upload = st.sidebar.file_uploader(
    "① 인구 데이터 CSV",
    type=["csv"],
    key="population_upload",
)

if population_upload is not None:

    population, error = load_population(
        population_upload
    )

    if population is not None:
        st.session_state.population = population
        st.session_state.population_source = (
            population_upload.name
        )
        st.sidebar.success("인구 데이터 연결 완료")
    else:
        st.sidebar.error(error)


demand_upload = st.sidebar.file_uploader(
    "② 승하차 데이터 CSV",
    type=["csv"],
    key="demand_upload",
)

if demand_upload is not None:

    demand, error = load_demand(
        demand_upload
    )

    if demand is not None:
        st.session_state.demand = demand
        st.session_state.demand_source = (
            demand_upload.name
        )
        st.sidebar.success("승하차 데이터 연결 완료")
    else:
        st.sidebar.error(error)


st.sidebar.divider()

st.sidebar.metric(
    "정류장",
    f"{len(stops):,}개",
)

st.sidebar.metric(
    "인구 데이터",
    "연결" if st.session_state.population is not None else "미연결",
)

st.sidebar.metric(
    "승하차 데이터",
    "연결" if st.session_state.demand is not None else "미연결",
)

st.sidebar.caption(
    "※ 실제 정류장 데이터는 CSV 파일을 기준으로 합니다."
)


# ============================================================
# PAGE 0. 종합 대시보드
# ============================================================

if page == "🏠 종합 대시보드":

    page_header(
        "00",
        "대중교통 계획 종합 대시보드",
        "도시·인구·대중교통 데이터를 종합하여 미래 대중교통 정책을 계획합니다.",
    )

    impact = future_impact(
        st.session_state.future_factors
    )

    population = st.session_state.population
    demand = st.session_state.demand

    total_population = (
        int(population["인구수(명)"].sum())
        if population is not None
        else 0
    )

    total_demand = (
        int(demand["일평균 승하차(건)"].sum())
        if demand is not None
        else 0
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "실제 정류장",
        f"{len(stops):,}개",
    )

    b.metric(
        "총 인구",
        f"{total_population:,}명"
        if total_population
        else "미입력",
    )

    c.metric(
        "일평균 승하차",
        f"{total_demand:,}건"
        if total_demand
        else "미입력",
    )

    d.metric(
        "미래수요 영향",
        f"+{impact}%",
    )

    st.markdown("### 🧭 대중교통 계획 프로세스")

    steps = [
        ("01", "기초현황", "인구·시설·정류장"),
        ("02", "이용현황", "승하차·시간대"),
        ("03", "수요·문제", "수요·취약지역"),
        ("04", "장래여건", "개발·철도·산업"),
        ("05", "장래수요", "AI 미래수요"),
        ("06", "노선계획", "신설·연장·조정"),
        ("07", "운영계획", "배차·환승"),
        ("08", "시뮬레이션", "대안 비교"),
        ("09", "종합평가", "효과·비용"),
        ("10", "행정결정", "최종 정책"),
    ]

    cols = st.columns(5)

    for i, step in enumerate(steps):

        with cols[i % 5]:

            st.markdown(
                f"""
                <div class="step-card">
                    <b>{step[0]}</b><br>
                    <strong>{step[1]}</strong><br>
                    <span class="muted">{step[2]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 🔄 플랫폼의 핵심")

    st.markdown(
        """
        **현재 도시를 분석하고 → 미래 도시를 예측하고 →
        대중교통 대안을 만들고 → 효과를 비교하고 →
        행정기관의 최종 결정을 지원합니다.**
        """
    )


# ============================================================
# PAGE 1. 기초현황
# ============================================================

elif page == "01. 기초현황 분석":

    page_header(
        "01",
        "기초현황 분석",
        "청주시의 대중교통 공간구조와 정류장 현황을 분석합니다.",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "정류장 수",
        f"{len(stops):,}개",
    )

    c2.metric(
        "위도 범위",
        f"{stops['위도'].min():.3f} ~ {stops['위도'].max():.3f}",
    )

    c3.metric(
        "경도 범위",
        f"{stops['경도'].min():.3f} ~ {stops['경도'].max():.3f}",
    )

    st.markdown("### 🗺️ 정류장 공간분포")

    m = folium.Map(
        location=[
            stops["위도"].mean(),
            stops["경도"].mean(),
        ],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    sample = stops

    if len(sample) > 700:
        sample = sample.sample(
            700,
            random_state=42,
        )

    for _, row in sample.iterrows():

        folium.CircleMarker(
            location=[
                row["위도"],
                row["경도"],
            ],
            radius=3,
            color="black",
            fill=True,
            fill_opacity=0.5,
            tooltip=row["정류소명"],
        ).add_to(m)

    st_folium(
        m,
        width=None,
        height=600,
    )


# ============================================================
# PAGE 2. 이용현황
# ============================================================

elif page == "02. 이용현황 분석":

    page_header(
        "02",
        "대중교통 이용현황 분석",
        "승하차 데이터를 업로드하여 정류장별 이용수요를 분석합니다.",
    )

    demand = st.session_state.demand

    if demand is None:

        st.warning(
            "승하차 데이터를 업로드하면 정류장별 이용수요를 분석할 수 있습니다."
        )

        st.markdown(
            """
            ### 📄 권장 CSV 형식

            | 정류소명 | 일평균 승하차(건) |
            |---|---:|
            | 청주터미널 | 12500 |
            | 사창사거리 | 8700 |
            | 내덕동 | 4300 |
            """
        )

    else:

        total = int(
            demand["일평균 승하차(건)"].sum()
        )

        avg = demand[
            "일평균 승하차(건)"
        ].mean()

        max_stop = demand.loc[
            demand["일평균 승하차(건)"].idxmax(),
            "정류소명",
        ]

        a, b, c = st.columns(3)

        a.metric(
            "총 일평균 승하차",
            f"{total:,}건",
        )

        b.metric(
            "정류장 평균",
            f"{avg:,.0f}건",
        )

        c.metric(
            "최대 수요 정류장",
            max_stop,
        )

        st.markdown("### 📊 수요 상위 정류장")

        top = (
            demand
            .sort_values(
                "일평균 승하차(건)",
                ascending=False,
            )
            .head(20)
            .set_index("정류소명")
        )

        st.bar_chart(
            top["일평균 승하차(건)"]
        )

        st.dataframe(
            demand.sort_values(
                "일평균 승하차(건)",
                ascending=False,
            ).head(100),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 3. 수요·문제 분석
# ============================================================

elif page == "03. 수요·문제 분석":

    page_header(
        "03",
        "수요·문제 분석",
        "수요가 높은 곳과 대중교통 공급이 필요한 지역을 찾아 계획 문제를 도출합니다.",
    )

    population = st.session_state.population
    demand = st.session_state.demand

    if demand is None:

        st.info(
            "승하차 데이터를 연결하면 실제 수요 기반 문제분석이 가능합니다."
        )

    else:

        threshold = st.slider(
            "고수요 정류장 기준(일평균 승하차)",
            1000,
            20000,
            5000,
            500,
        )

        high = demand[
            demand["일평균 승하차(건)"] >= threshold
        ].copy()

        c1, c2 = st.columns(2)

        c1.metric(
            "고수요 정류장",
            f"{len(high):,}개",
        )

        c2.metric(
            "고수요 총 승하차",
            f"{int(high['일평균 승하차(건)'].sum()):,}건",
        )

        st.markdown("### 🔍 주요 계획 이슈")

        issues = [
            "고수요 정류장의 혼잡 및 대기시간 증가 가능성",
            "수요가 집중되는 생활권의 배차·노선 공급 검토 필요",
            "인구 및 개발지역의 장래 수요 변화 검토 필요",
            "환승거점 중심의 연계체계 검토 필요",
        ]

        for issue in issues:
            st.markdown(f"- {issue}")

        st.dataframe(
            high.sort_values(
                "일평균 승하차(건)",
                ascending=False,
            ).head(50),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 4. 장래여건
# ============================================================

elif page == "04. 장래여건 분석":

    page_header(
        "04",
        "장래여건 분석",
        "미래 도시개발과 교통체계 변화를 입력하여 장래 교통환경을 설정합니다.",
    )

    year = st.selectbox(
        "📅 목표연도",
        [2027, 2028, 2029, 2030, 2035],
        index=3,
    )

    st.session_state.prediction_year = year

    st.markdown("### 🏗️ 미래 도시변화")

    selected = []

    for factor in FUTURE_FACTORS:

        checked = st.checkbox(
            factor,
            value=factor in st.session_state.future_factors,
        )

        if checked:
            selected.append(factor)

    st.session_state.future_factors = selected

    impact = future_impact(selected)

    a, b, c = st.columns(3)

    a.metric(
        "목표연도",
        f"{year}년",
    )

    b.metric(
        "선택 개발요소",
        f"{len(selected)}개",
    )

    c.metric(
        "PoC 수요 영향",
        f"+{impact}%",
    )

    if selected:

        st.markdown("### 📋 적용 시나리오")

        st.dataframe(
            pd.DataFrame(
                {
                    "미래요소": selected,
                    "영향 가중치": [
                        f"+{FUTURE_FACTORS[item]}%"
                        for item in selected
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 5. 장래수요
# ============================================================

elif page == "05. 장래수요 예측":

    page_header(
        "05",
        "AI 장래수요 예측",
        "현재 인구·승하차 자료와 미래 도시변화를 활용하여 장래 대중교통 수요를 추정합니다.",
    )

    population = st.session_state.population
    demand = st.session_state.demand

    impact = future_impact(
        st.session_state.future_factors
    )

    if population is None:

        st.warning(
            "인구 데이터를 먼저 업로드하세요."
        )

    else:

        total_population = int(
            population["인구수(명)"].sum()
        )

        base_population = max(
            total_population,
            1,
        )

        growth_rate = st.slider(
            "추가 장래 인구 증가율(%)",
            0,
            30,
            5,
            1,
        )

        future_population = round(
            total_population
            * (1 + growth_rate / 100)
        )

        future_demand = round(
            future_population
            / base_population
            * (1 + impact / 100)
            * (
                demand["일평균 승하차(건)"].sum()
                if demand is not None
                else 100000
            )
        )

        a, b, c = st.columns(3)

        a.metric(
            "현재 총인구",
            f"{total_population:,}명",
        )

        b.metric(
            f"{st.session_state.prediction_year}년 추정인구",
            f"{future_population:,}명",
        )

        c.metric(
            "장래 추정 수요",
            f"{future_demand:,}건",
        )

        st.markdown("### 📈 장래수요 산정 구조")

        st.markdown(
            """
            **현재 인구·현재 교통수요**
            ↓
            **장래 인구 변화**
            ↓
            **도시개발 영향**
            ↓
            **장래 대중교통 수요 추정**
            """
        )

        st.info(
            "현재 계산은 공모전용 PoC입니다. "
            "실증 단계에서는 정류장 단위 교통카드 자료와 "
            "검증된 수요예측 모델을 적용합니다."
        )


# ============================================================
# PAGE 6. 노선 대안
# ============================================================

elif page == "06. 노선 대안 계획":

    page_header(
        "06",
        "노선 대안 계획",
        "실제 정류장 위치를 활용하여 노선 신설·연장·조정 후보를 검토합니다.",
    )

    st.warning(
        "현재 정류장 데이터에는 실제 노선번호와 운행순서가 없으므로 "
        "아래 노선은 '계획 후보노선'입니다."
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
        + " | "
        + options["서비스ID"]
    )

    c1, c2 = st.columns(2)

    with c1:
        origin_label = st.selectbox(
            "기점 후보",
            options["표시명"].tolist(),
        )

    with c2:
        destination_label = st.selectbox(
            "종점 후보",
            options["표시명"].tolist(),
            index=min(1, len(options) - 1),
        )

    origin_id = origin_label.split(" | ")[-1]
    destination_id = destination_label.split(" | ")[-1]

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

    middle_lat = (
        origin["위도"]
        + destination["위도"]
    ) / 2

    middle_lon = (
        origin["경도"]
        + destination["경도"]
    ) / 2

    middle_stops = nearest_stops(
        stops,
        middle_lat,
        middle_lon,
        5,
    )

    points = [origin_point]

    for _, row in middle_stops.iterrows():

        if row["서비스ID"] in {
            origin_id,
            destination_id,
        }:
            continue

        points.append(
            (
                float(row["위도"]),
                float(row["경도"]),
            )
        )

        if len(points) >= 3:
            break

    points.append(destination_point)

    distance = sum(
        haversine_km(
            points[i],
            points[i + 1],
        )
        for i in range(len(points) - 1)
    )

    a, b = st.columns(2)

    a.metric(
        "후보노선 거리",
        f"{distance:.2f} km",
    )

    b.metric(
        "예상 주행시간",
        f"{distance / 25 * 60:.1f}분",
    )

    m = folium.Map(
        location=[
            middle_lat,
            middle_lon,
        ],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    for i, point in enumerate(points):

        folium.Marker(
            point,
            tooltip=(
                "기점"
                if i == 0
                else "종점"
                if i == len(points) - 1
                else f"경유 후보 {i}"
            ),
        ).add_to(m)

    folium.PolyLine(
        points,
        color="red",
        weight=6,
        opacity=.85,
        tooltip="계획 후보노선",
    ).add_to(m)

    st_folium(
        m,
        width=None,
        height=550,
    )


# ============================================================
# PAGE 7. 배차·환승
# ============================================================

elif page == "07. 배차·환승 계획":

    page_header(
        "07",
        "배차·환승 계획",
        "노선 대안에 따른 운행횟수와 환승체계의 변화를 검토합니다.",
    )

    st.markdown("### 🚌 배차계획")

    vehicle_count = st.number_input(
        "투입 차량 대수",
        min_value=1,
        max_value=50,
        value=8,
    )

    cycle_time = st.number_input(
        "왕복 운행시간(분)",
        min_value=20,
        max_value=240,
        value=80,
    )

    operating_hours = st.number_input(
        "운행시간(시간/일)",
        min_value=1,
        max_value=24,
        value=16,
    )

    headway = cycle_time / max(
        vehicle_count,
        1,
    )

    daily_runs = (
        operating_hours
        * 60
        / max(headway, 1)
    )

    a, b = st.columns(2)

    a.metric(
        "계산 배차간격",
        f"{headway:.1f}분",
    )

    b.metric(
        "예상 일일 운행횟수",
        f"{daily_runs:.0f}회",
    )

    st.markdown("### 🔄 환승체계")

    transfer_penalty = st.slider(
        "평균 환승 추가시간",
        0,
        20,
        5,
    )

    transfer_quality = max(
        0,
        100 - transfer_penalty * 5,
    )

    st.metric(
        "환승 편의 지수",
        f"{transfer_quality}점",
    )

    if transfer_quality >= 80:
        st.success("환승체계가 양호한 시나리오입니다.")
    elif transfer_quality >= 60:
        st.warning("환승 개선 여지가 있습니다.")
    else:
        st.error("환승체계 개선이 필요합니다.")


# ============================================================
# PAGE 8. 시뮬레이션
# ============================================================

elif page == "08. 대안 시뮬레이션":

    page_header(
        "08",
        "정책 대안 시뮬레이션",
        "현행안과 여러 대안을 동일한 기준으로 비교합니다.",
    )

    demand = st.session_state.demand
    impact = future_impact(
        st.session_state.future_factors
    )

    if demand is not None:
        base_demand = int(
            demand["일평균 승하차(건)"].sum()
        )
    else:
        base_demand = 100000

    base_time = st.slider(
        "현재 평균 통행시간(분)",
        10,
        90,
        40,
    )

    table = build_policy_table(
        base_demand,
        impact,
        base_time,
    )

    best = table.loc[
        table["종합점수"].idxmax(),
        "정책 대안",
    ]

    st.success(
        f"🤖 PoC 추천 우선검토안: **{best}**"
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 📊 대안별 종합점수")

    chart = table.set_index(
        "정책 대안"
    )["종합점수"]

    st.bar_chart(chart)

    st.caption(
        "※ 종합점수는 공모전 PoC를 위한 예시 계산이며 "
        "실제 정책 결정에는 검증된 교통분석 모델이 필요합니다."
    )


# ============================================================
# PAGE 9. 종합평가
# ============================================================

elif page == "09. 종합평가":

    page_header(
        "09",
        "대중교통 대안 종합평가",
        "수요·접근성·통행시간·환승·운영비를 종합하여 대안을 평가합니다.",
    )

    demand = st.session_state.demand

    base_demand = (
        int(demand["일평균 승하차(건)"].sum())
        if demand is not None
        else 100000
    )

    impact = future_impact(
        st.session_state.future_factors
    )

    table = build_policy_table(
        base_demand,
        impact,
        40,
    )

    criteria = st.multiselect(
        "중요하게 평가할 항목",
        [
            "수요 대응",
            "통행시간",
            "대기시간",
            "환승 편의",
            "운영비",
            "접근성",
        ],
        default=[
            "수요 대응",
            "통행시간",
            "환승 편의",
            "운영비",
        ],
    )

    st.markdown("### ⚖️ 평가기준")

    st.dataframe(
        table[
            [
                "정책 대안",
                "예상 일평균 수요(건)",
                "평균 통행시간(분)",
                "평균 대기시간(분)",
                "평균 환승시간(분)",
                "연간 추가 운영비(억원)",
                "접근성 지수",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    best = table.loc[
        table["종합점수"].idxmax(),
        "정책 대안",
    ]

    st.markdown("### 🏆 우선 검토 대안")

    st.success(
        f"**{best}**을 우선 검토 대상으로 제시합니다."
    )

    st.info(
        "AI는 여러 대안을 비교·제시하며 "
        "최종 결정은 행정기관과 교통계획 담당자가 수행합니다."
    )


# ============================================================
# PAGE 10. 행정 의사결정
# ============================================================

elif page == "10. 행정 의사결정":

    page_header(
        "10",
        "행정 의사결정",
        "AI 분석결과를 행정기관의 실제 정책 결정 과정과 연결합니다.",
    )

    st.markdown(
        """
        ### 🏛️ 의사결정 구조

        **데이터 분석**
        → **AI 예측**
        → **대안 생성**
        → **효과 비교**
        → **전문가 검토**
        → **행정기관 결정**
        → **정책 시행**
        """
    )

    policy = st.radio(
        "최종 검토 정책",
        POLICIES,
        index=POLICIES.index(
            st.session_state.selected_policy
        ),
    )

    st.session_state.selected_policy = policy

    decision = st.selectbox(
        "행정기관 판단",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 분석 요청",
            "보류",
        ],
    )

    if decision == "정책 대안 채택":
        st.success(
            f"최종 정책: **{policy}**"
        )

    elif decision == "추가 분석 요청":
        st.warning(
            "추가 수요·교통·재정 데이터를 반영하여 "
            "대안을 다시 분석합니다."
        )

    elif decision == "보류":
        st.error(
            "현재 단계에서는 정책을 보류합니다."
        )

    else:
        st.info(
            "AI 분석 결과를 검토하고 "
            "행정기관이 최종 판단합니다."
        )


# ============================================================
# PAGE 11. 시행·모니터링
# ============================================================

elif page == "11. 시행·모니터링":

    page_header(
        "11",
        "정책 시행·모니터링",
        "정책 시행 이후 운영자료와 시민 의견을 다시 분석하여 다음 계획에 반영합니다.",
    )

    st.markdown(
        """
        ### 🔄 선순환 구조

        **정책 시행**
        ↓
        **운영 데이터 수집**
        ↓
        **승하차·배차·환승 분석**
        ↓
        **시민 피드백**
        ↓
        **AI 재분석**
        ↓
        **다음 대중교통 계획 개선**
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        satisfaction = st.slider(
            "시민 만족도",
            1,
            5,
            4,
        )

    with c2:
        punctuality = st.slider(
            "정시성 만족도",
            1,
            5,
            4,
        )

    with c3:
        transfer = st.slider(
            "환승 만족도",
            1,
            5,
            4,
        )

    average = round(
        (
            satisfaction
            + punctuality
            + transfer
        ) / 3,
        2,
    )

    st.metric(
        "정책 시행 후 종합 만족도",
        f"{average:.2f} / 5",
    )

    comment = st.text_area(
        "시민 의견",
        placeholder="예: 출퇴근 시간 배차간격을 줄여주세요.",
    )

    if st.button(
        "🔄 다음 정책에 반영",
        use_container_width=True,
    ):

        if average >= 4:
            st.success(
                "현재 정책의 성과가 양호합니다. "
                "다음 계획에서 유지·확대를 검토합니다."
            )
        elif average >= 3:
            st.warning(
                "부분적인 운영 개선이 필요합니다."
            )
        else:
            st.error(
                "정책 재검토가 필요합니다."
            )

        if comment.strip():
            st.write(
                f"**시민 피드백:** {comment}"
            )


# ============================================================
# 7. 하단
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "AI는 정책을 자동 결정하지 않습니다."
)

st.sidebar.caption(
    "AI 분석 → 대안 제시 → 전문가 검토 → 행정기관 최종 결정"
)
