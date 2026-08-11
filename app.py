import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from math import radians, sin, cos, sqrt, atan2

# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 미래예측형 대중교통 의사결정 지원",
    page_icon="🚌",
    layout="wide",
)

st.title("🚌 AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼")
st.caption("공모전용 PoC · 청주시 적용 시나리오")

st.info(
    "※ 본 프로토타입은 정책 시나리오 실험을 위한 PoC입니다. "
    "기준 데이터와 정책 변수를 사용자가 직접 조정하여 "
    "미래 이동수요와 대중교통 대안을 비교할 수 있습니다."
)

# ============================================================
# 기본 데이터
# ============================================================

def default_population():
    return pd.DataFrame({
        "지역명": [
            "상당구",
            "서원구",
            "흥덕구",
            "청원구",
            "오창읍",
            "오송읍",
            "가경동",
            "복대동",
            "성안동",
            "내덕동",
        ],
        "기준인구(명)": [
            194551,
            182689,
            292051,
            186861,
            67832,
            48348,
            53342,
            52000,
            15000,
            30000,
        ],
        "위도": [
            36.633,
            36.628,
            36.635,
            36.665,
            36.7153,
            36.6205,
            36.6240,
            36.6355,
            36.6338,
            36.6480,
        ],
        "경도": [
            127.490,
            127.470,
            127.430,
            127.490,
            127.4258,
            127.3274,
            127.3900,
            127.4221,
            127.4879,
            127.4890,
        ],
    })


def default_demand():
    return pd.DataFrame({
        "정류장ID": [
            "S001",
            "S002",
            "S003",
            "S004",
            "S005",
            "S006",
            "S007",
            "S008",
        ],
        "정류장명": [
            "청주터미널",
            "사창사거리",
            "내덕동",
            "오창산단",
            "오송역",
            "가경동",
            "복대동",
            "성안동",
        ],
        "위도": [
            36.6271,
            36.6342,
            36.6482,
            36.7153,
            36.6205,
            36.6240,
            36.6355,
            36.6338,
        ],
        "경도": [
            127.4321,
            127.4567,
            127.4891,
            127.4258,
            127.3274,
            127.3900,
            127.4221,
            127.4879,
        ],
        "일평균 승차(건)": [
            5200,
            3400,
            1600,
            2800,
            4100,
            3000,
            3600,
            1800,
        ],
        "일평균 하차(건)": [
            5000,
            3600,
            1700,
            2700,
            4300,
            3100,
            3500,
            1900,
        ],
    })


# ============================================================
# CSV 안전 로딩
# ============================================================

def read_csv_flexible(uploaded_file):
    if uploaded_file is None:
        return None, "파일이 없습니다."

    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()

        if not raw:
            return None, "업로드된 파일이 비어 있습니다."

        text = None

        for encoding in [
            "utf-8-sig",
            "cp949",
            "euc-kr",
            "utf-8",
        ]:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            return None, "CSV 인코딩을 확인할 수 없습니다."

        if not text.strip():
            return None, "CSV 내용이 없습니다."

        from io import StringIO

        df = pd.read_csv(
            StringIO(text),
            sep=None,
            engine="python",
        )

        if df.empty:
            return None, "CSV 데이터가 없습니다."

        df.columns = (
            df.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        return df, None

    except pd.errors.EmptyDataError:
        return None, "CSV 파일에 읽을 데이터가 없습니다."

    except Exception as error:
        return None, f"CSV 읽기 오류: {error}"


def load_population(uploaded_file):
    df, error = read_csv_flexible(uploaded_file)

    if error:
        return None, error

    # 우리가 만든 단순 형식
    rename = {}

    for col in df.columns:
        normalized = (
            str(col)
            .replace(" ", "")
            .replace("_", "")
            .strip()
        )

        if normalized in {
            "지역명",
            "지역",
            "읍면동",
            "읍면동명",
            "행정동",
            "행정동명",
        }:
            rename[col] = "지역명"

        elif normalized in {
            "기준인구명",
            "인구수명",
            "인구수",
            "총인구수명",
            "인구",
        }:
            rename[col] = "기준인구(명)"

    df = df.rename(columns=rename)

    # 사용자가 제공한 시계열 형식
    # 행정구역(동읍면)별 / 항목 / 2012 ... 2025
    if (
        "행정구역(동읍면)별" in df.columns
        and "항목" in df.columns
    ):
        total_rows = df[
            df["항목"]
            .astype(str)
            .str.contains(
                "총인구수",
                na=False,
            )
        ].copy()

        if total_rows.empty:
            return None, (
                "시계열 인구 파일에서 "
                "'총인구수' 행을 찾지 못했습니다."
            )

        year_columns = [
            str(year)
            for year in range(2012, 2026)
            if str(year) in df.columns
        ]

        if not year_columns:
            return None, (
                "2012~2025 연도 열을 찾지 못했습니다."
            )

        rows = []

        for _, row in total_rows.iterrows():
            region = str(
                row["행정구역(동읍면)별"]
            ).strip()

            values = {}

            for year in year_columns:
                value = str(row[year])
                value = (
                    value
                    .replace(",", "")
                    .replace("-", "")
                    .strip()
                )

                values[year] = pd.to_numeric(
                    value,
                    errors="coerce",
                )

            valid = pd.Series(values).dropna()

            if valid.empty:
                continue

            rows.append({
                "지역명": region,
                "기준인구(명)": int(
                    round(valid.iloc[-1])
                ),
                **{
                    f"인구_{year}": values[year]
                    for year in year_columns
                },
            })

        result = pd.DataFrame(rows)

        if result.empty:
            return None, "유효한 인구 데이터가 없습니다."

        return result, None

    # 단순 지역명 + 인구수 형식
    if {
        "지역명",
        "기준인구(명)",
    }.issubset(df.columns):

        df["지역명"] = (
            df["지역명"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["기준인구(명)"] = (
            df["기준인구(명)"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("명", "", regex=False)
        )

        df["기준인구(명)"] = pd.to_numeric(
            df["기준인구(명)"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "지역명",
                "기준인구(명)",
            ]
        ).copy()

        return df.reset_index(drop=True), None

    return None, (
        "인구 CSV 형식을 인식하지 못했습니다.\n"
        "지원 형식: 지역명/기준인구(명) 또는 "
        "행정구역(동읍면)별/항목/2012~2025"
    )


def load_demand(uploaded_file):
    df, error = read_csv_flexible(uploaded_file)

    if error:
        return None, error

    rename = {}

    for col in df.columns:
        normalized = (
            str(col)
            .replace(" ", "")
            .replace("_", "")
        )

        if normalized in {
            "정류장ID",
            "정류장아이디",
        }:
            rename[col] = "정류장ID"

        elif normalized in {
            "정류장명",
            "정류소명",
            "정류소",
        }:
            rename[col] = "정류장명"

        elif normalized in {
            "일평균승차건",
            "승차인원",
            "승차",
        }:
            rename[col] = "일평균 승차(건)"

        elif normalized in {
            "일평균하차건",
            "하차인원",
            "하차",
        }:
            rename[col] = "일평균 하차(건)"

        elif normalized == "위도":
            rename[col] = "위도"

        elif normalized == "경도":
            rename[col] = "경도"

    df = df.rename(columns=rename)

    required = {
        "정류장명",
        "일평균 승차(건)",
        "일평균 하차(건)",
    }

    if not required.issubset(df.columns):
        return None, (
            "승하차 CSV에 필요한 컬럼이 없습니다.\n"
            "필수: 정류장명, 일평균 승차(건), "
            "일평균 하차(건)"
        )

    for col in [
        "일평균 승차(건)",
        "일평균 하차(건)",
    ]:
        df[col] = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0)

    return df.reset_index(drop=True), None


# ============================================================
# 거리 계산
# ============================================================

def haversine_km(a, b):
    lat1, lon1 = a
    lat2, lon2 = b

    R = 6371.0

    p1 = radians(lat1)
    p2 = radians(lat2)

    dphi = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)

    x = (
        sin(dphi / 2) ** 2
        + cos(p1)
        * cos(p2)
        * sin(dl / 2) ** 2
    )

    return (
        2
        * R
        * atan2(
            sqrt(x),
            sqrt(1 - x),
        )
    )


# ============================================================
# 세션 상태
# ============================================================

if "population" not in st.session_state:
    st.session_state.population = None

if "demand" not in st.session_state:
    st.session_state.demand = None


# ============================================================
# 사이드바
# ============================================================

st.sidebar.header("📂 기준 데이터")

population_file = st.sidebar.file_uploader(
    "인구 CSV",
    type=["csv"],
    help=(
        "시계열 형식: 행정구역(동읍면)별, "
        "항목, 2012~2025"
    ),
)

if population_file is not None:
    population, error = load_population(
        population_file
    )

    if population is not None:
        st.session_state.population = population
        st.sidebar.success(
            f"인구 데이터 {len(population):,}개 지역 연결"
        )
    else:
        st.sidebar.error(error)

demand_file = st.sidebar.file_uploader(
    "승하차 CSV",
    type=["csv"],
)

if demand_file is not None:
    demand, error = load_demand(
        demand_file
    )

    if demand is not None:
        st.session_state.demand = demand
        st.sidebar.success(
            f"승하차 데이터 {len(demand):,}개 정류장 연결"
        )
    else:
        st.sidebar.error(error)

if st.sidebar.button(
    "↺ 데이터 초기화",
    use_container_width=True,
):
    st.session_state.population = None
    st.session_state.demand = None
    st.rerun()


# ============================================================
# 기준 데이터
# ============================================================

population = st.session_state.population

if population is None:
    population = default_population()
    population_source = "기본 PoC 데이터"
else:
    population_source = "사용자 업로드 데이터"

demand = st.session_state.demand

if demand is None:
    demand = default_demand()
    demand_source = "기본 PoC 데이터"
else:
    demand_source = "사용자 업로드 데이터"


# ============================================================
# 페이지 선택
# ============================================================

page = st.sidebar.radio(
    "📑 계획 단계",
    [
        "🏠 종합 대시보드",
        "👥 인구·장래수요",
        "🚌 대중교통 이용현황",
        "🔮 미래 시나리오",
        "🛣️ 노선 대안",
        "📊 대안 비교",
        "🏛️ 행정 의사결정",
    ],
)


# ============================================================
# 1. 종합 대시보드
# ============================================================

if page == "🏠 종합 대시보드":

    st.header("🏠 대중교통 계획 종합 대시보드")

    total_population = int(
        population["기준인구(명)"].sum()
    )

    total_demand = int(
        demand["일평균 승차(건)"].sum()
        + demand["일평균 하차(건)"].sum()
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "기준 인구",
        f"{total_population:,}명",
    )

    b.metric(
        "정류장",
        f"{len(demand):,}개",
    )

    c.metric(
        "일평균 승하차",
        f"{total_demand:,}건",
    )

    d.metric(
        "데이터 상태",
        "PoC",
    )

    st.markdown("---")

    st.markdown(
        """
### 🧭 AI 대중교통 계획 프로세스

**현재 도시현황 → 미래 시나리오 설정 → 장래수요 예측
→ 노선 대안 생성 → 효과 비교 → 행정기관 의사결정**

이 플랫폼은 AI가 정책을 자동 결정하는 것이 아니라,
**여러 미래 상황을 빠르게 비교하여 행정기관의 판단을 지원하는 것**을 목표로 합니다.
"""
    )


# ============================================================
# 2. 인구·장래수요
# ============================================================

elif page == "👥 인구·장래수요":

    st.header("👥 인구 및 장래수요 분석")

    region_list = population["지역명"].tolist()

    selected_region = st.selectbox(
        "분석 지역",
        region_list,
    )

    selected_row = population[
        population["지역명"]
        == selected_region
    ].iloc[0]

    current_population = int(
        selected_row["기준인구(명)"]
    )

    st.metric(
        "기준 인구",
        f"{current_population:,}명",
    )

    year_cols = [
        col
        for col in population.columns
        if str(col).startswith("인구_")
    ]

    if year_cols:

        chart = selected_row[
            year_cols
        ].copy()

        chart.index = [
            int(str(x).replace("인구_", ""))
            for x in chart.index
        ]

        chart = pd.to_numeric(
            chart,
            errors="coerce",
        ).dropna()

        st.line_chart(chart)

        st.caption(
            "실제 업로드된 시계열 데이터가 있는 경우 "
            "과거 인구 추세를 기준으로 시나리오 분석에 활용합니다."
        )

    st.markdown("---")

    st.subheader("🔮 장래인구 시나리오")

    target_year = st.selectbox(
        "예측 연도",
        [2027, 2030, 2035, 2040],
        index=1,
    )

    future_growth = st.slider(
        "사용자 조정 장래 인구 변화율 (%)",
        -30,
        100,
        10,
        1,
    )

    future_population = round(
        current_population
        * (1 + future_growth / 100)
    )

    a, b = st.columns(2)

    a.metric(
        f"{target_year}년 시나리오 인구",
        f"{future_population:,}명",
        f"{future_growth:+d}%",
    )

    if year_cols:
        latest = pd.to_numeric(
            selected_row[year_cols],
            errors="coerce",
        ).dropna()

        if len(latest) >= 2:
            x = np.arange(len(latest))
            model = LinearRegression()
            model.fit(
                x.reshape(-1, 1),
                latest.values,
            )

            predicted = max(
                0,
                int(
                    round(
                        model.predict(
                            [[len(latest) + 4]]
                        )[0]
                    )
                ),
            )

            b.metric(
                "추세 기반 참고예측",
                f"{predicted:,}명",
            )

    st.info(
        "※ 사용자가 조정한 장래 인구 변화율은 "
        "정책 시나리오 실험값입니다."
    )


# ============================================================
# 3. 대중교통 이용현황
# ============================================================

elif page == "🚌 대중교통 이용현황":

    st.header("🚌 대중교통 이용현황")

    demand["총 승하차(건)"] = (
        demand["일평균 승차(건)"]
        + demand["일평균 하차(건)"]
    )

    st.dataframe(
        demand.sort_values(
            "총 승하차(건)",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🔥 고수요 정류장")

    top = demand.nlargest(
        10,
        "총 승하차(건)",
    )

    st.bar_chart(
        top.set_index("정류장명")[
            "총 승하차(건)"
        ]
    )


# ============================================================
# 4. 미래 시나리오
# ============================================================

elif page == "🔮 미래 시나리오":

    st.header("🔮 미래 도시 시나리오 설정")

    st.caption(
        "이 화면은 정책 담당자가 가상의 미래 상황을 "
        "직접 설정하고 결과를 비교하기 위한 핵심 PoC입니다."
    )

    region_list = population["지역명"].tolist()

    selected_region = st.selectbox(
        "시나리오 대상 지역",
        region_list,
    )

    selected_row = population[
        population["지역명"]
        == selected_region
    ].iloc[0]

    base_population = int(
        selected_row["기준인구(명)"]
    )

    st.markdown("### 👥 인구 시나리오")

    population_change = st.slider(
        "인구 변화율 (%)",
        -50,
        100,
        15,
        1,
    )

    st.markdown("### 🏗️ 미래 도시개발")

    development_type = st.selectbox(
        "개발 유형",
        [
            "개발 없음",
            "공동주택 개발",
            "산업단지 조성",
            "대규모 상업시설",
            "복합개발",
        ],
    )

    development_population = st.slider(
        "개발에 따른 추가 유입인구 (명)",
        0,
        30000,
        5000,
        500,
    )

    st.markdown("### 🚦 교통환경")

    congestion = st.slider(
        "교통 혼잡 증가율 (%)",
        -20,
        100,
        10,
        1,
    )

    demand_change = st.slider(
        "대중교통 이용수요 변화율 (%)",
        -30,
        100,
        15,
        1,
    )

    scenario_population = round(
        base_population
        * (1 + population_change / 100)
        + development_population
    )

    scenario_demand = round(
        scenario_population
        / max(base_population, 1)
        * (
            1
            + demand_change / 100
        )
        * (
            1
            + congestion / 200
        )
        * (
            demand["총 승하차(건)"].mean()
        )
    )

    a, b, c = st.columns(3)

    a.metric(
        "기준 인구",
        f"{base_population:,}명",
    )

    b.metric(
        "시나리오 인구",
        f"{scenario_population:,}명",
        f"{population_change:+d}% + 개발유입",
    )

    c.metric(
        "예상 수요지수",
        f"{scenario_demand:,.0f}",
    )

    st.success(
        f"현재 설정된 시나리오: "
        f"**{selected_region} / {development_type}**"
    )


# ============================================================
# 5. 노선 대안
# ============================================================

elif page == "🛣️ 노선 대안":

    st.header("🛣️ 노선 대안 계획")

    scenario = st.selectbox(
        "정책 대안",
        [
            "현행 유지",
            "대안 A · 거점 직결형",
            "대안 B · 간선 연장형",
            "대안 C · 환승 최적화형",
        ],
    )

    scenario_info = {
        "현행 유지": (
            "현재 노선을 유지하고 수요 변화만 모니터링합니다."
        ),
        "대안 A · 거점 직결형": (
            "주요 생활권·산업단지·환승거점을 직접 연결합니다."
        ),
        "대안 B · 간선 연장형": (
            "기존 간선노선을 미래 개발지역까지 연장합니다."
        ),
        "대안 C · 환승 최적화형": (
            "환승거점을 중심으로 배차와 환승체계를 조정합니다."
        ),
    }

    st.info(
        scenario_info[scenario]
    )

    st.markdown("### 🗺️ 계획 대상")

    if {
        "위도",
        "경도",
    }.issubset(demand.columns):

        center_lat = demand["위도"].mean()
        center_lon = demand["경도"].mean()

        m = folium.Map(
            location=[
                center_lat,
                center_lon,
            ],
            zoom_start=11,
            tiles="CartoDB positron",
        )

        for _, row in demand.iterrows():

            folium.CircleMarker(
                [
                    row["위도"],
                    row["경도"],
                ],
                radius=max(
                    5,
                    min(
                        25,
                        row["총 승하차(건)"]
                        / 600,
                    ),
                ),
                tooltip=(
                    f"{row['정류장명']} | "
                    f"승하차 "
                    f"{row['총 승하차(건)']:,}건"
                ),
                color="black",
                fill=True,
                fill_opacity=0.5,
            ).add_to(m)

        st_folium(
            m,
            width=None,
            height=550,
        )


# ============================================================
# 6. 대안 비교
# ============================================================

elif page == "📊 대안 비교":

    st.header("📊 대중교통 정책 대안 비교")

    base = int(
        demand["총 승하차(건)"].sum()
    )

    scenarios = pd.DataFrame({
        "정책 대안": [
            "현행 유지",
            "대안 A · 거점 직결형",
            "대안 B · 간선 연장형",
            "대안 C · 환승 최적화형",
        ],
        "예상 수요(건)": [
            base,
            round(base * 1.18),
            round(base * 1.10),
            round(base * 1.15),
        ],
        "평균 통행시간(분)": [
            45,
            31,
            36,
            34,
        ],
        "평균 대기시간(분)": [
            10,
            7,
            8,
            6,
        ],
        "환승시간(분)": [
            8,
            7,
            6,
            4,
        ],
        "추가 운영비(억원)": [
            0,
            4.5,
            2.8,
            3.6,
        ],
        "탄소배출 지수": [
            1.00,
            0.78,
            0.86,
            0.72,
        ],
    })

    # 정책 가중치
    st.sidebar.markdown("### ⚖️ 평가 가중치")

    w_demand = st.sidebar.slider(
        "수요",
        0.0,
        1.0,
        0.30,
        0.05,
    )

    w_time = st.sidebar.slider(
        "통행·대기시간",
        0.0,
        1.0,
        0.30,
        0.05,
    )

    w_cost = st.sidebar.slider(
        "운영비",
        0.0,
        1.0,
        0.20,
        0.05,
    )

    w_carbon = st.sidebar.slider(
        "탄소",
        0.0,
        1.0,
        0.20,
        0.05,
    )

    total_weight = (
        w_demand
        + w_time
        + w_cost
        + w_carbon
    )

    if total_weight == 0:
        total_weight = 1

    nd = (
        scenarios["예상 수요(건)"]
        / scenarios["예상 수요(건)"].max()
    )

    nt = 1 - (
        scenarios["평균 통행시간(분)"]
        + scenarios["평균 대기시간(분)"]
    ) / (
        scenarios["평균 통행시간(분)"]
        + scenarios["평균 대기시간(분)"]
    ).max()

    nc = 1 - (
        scenarios["추가 운영비(억원)"]
        / max(
            scenarios["추가 운영비(억원)"].max(),
            1,
        )
    )

    ncarbon = 1 - (
        scenarios["탄소배출 지수"]
        / scenarios["탄소배출 지수"].max()
    )

    scenarios["종합점수"] = (
        w_demand * nd
        + w_time * nt
        + w_cost * nc
        + w_carbon * ncarbon
    ) / total_weight

    best = scenarios.loc[
        scenarios["종합점수"].idxmax(),
        "정책 대안",
    ]

    def highlight_best(row):
        if row["정책 대안"] == best:
            return [
                "background-color: #E8F5E9"
            ] * len(row)

        return [""] * len(row)

    st.dataframe(
        scenarios.style.apply(
            highlight_best,
            axis=1,
        ).format({
            "예상 수요(건)": "{:,.0f}",
            "평균 통행시간(분)": "{:.1f}",
            "평균 대기시간(분)": "{:.1f}",
            "환승시간(분)": "{:.1f}",
            "추가 운영비(억원)": "{:.1f}",
            "탄소배출 지수": "{:.2f}",
            "종합점수": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        f"🤖 시나리오 조건에서 종합점수가 가장 높은 대안: "
        f"**{best}**"
    )

    st.caption(
        "※ 추천안은 PoC 계산 결과이며 최종 정책은 "
        "행정기관의 전문가 검토와 시민 의견을 통해 결정합니다."
    )


# ============================================================
# 7. 행정 의사결정
# ============================================================

elif page == "🏛️ 행정 의사결정":

    st.header("🏛️ 행정기관 의사결정 지원")

    st.markdown(
        """
### AI의 역할

AI는 **정책을 결정하는 주체가 아니라 의사결정을 지원하는 분석도구**입니다.

1. 미래 시나리오 설정
2. 장래 이동수요 예측
3. 노선·배차 대안 생성
4. 효과 및 비용 비교
5. 행정기관 최종 판단
6. 정책 시행
7. 운영 데이터 및 시민 피드백 반영
"""
    )

    decision = st.radio(
        "행정기관 의사결정",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 분석 필요",
        ],
    )

    if decision == "정책 대안 채택":
        st.success(
            "정책 대안을 채택합니다. "
            "시행 후 운영 데이터를 다시 수집하여 "
            "다음 정책 분석에 반영합니다."
        )

    elif decision == "추가 분석 필요":
        st.warning(
            "추가 데이터와 전문가 검토 후 "
            "다시 시나리오를 분석합니다."
        )

    else:
        st.info(
            "AI 분석 결과를 참고하여 "
            "행정기관이 최종 판단합니다."
        )

    st.markdown("---")

    st.markdown(
        """
### 🔄 정책 선순환 구조

**정책 수립 → 시행 → 운영 데이터 수집 → 시민 피드백
→ AI 분석 → 다음 정책 개선**

이를 통해 일회성 노선 개편이 아니라
**지속적으로 개선되는 데이터 기반 대중교통 행정체계**를 목표로 합니다.
"""
    )

# ============================================================
# 하단 고지
# ============================================================

st.markdown("---")

st.caption(
    "AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼 · "
    "공모전용 PoC"
)
