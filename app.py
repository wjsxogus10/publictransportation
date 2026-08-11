import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression

# ============================================================
# AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼
# 공모전용 PoC · 청주시 적용 시나리오
#
# 핵심 구조
# 1. 인구 CSV 또는 기본 PoC 데이터
# 2. 미래 인구 시나리오 직접 조정
# 3. 승하차 CSV 없음 → 수요를 직접 조정
# 4. 미래 수요 계산
# 5. 정책 대안 비교
# 6. 행정기관 최종 의사결정
# ============================================================

st.set_page_config(
    page_title="AI 미래예측형 대중교통 의사결정 지원",
    page_icon="🚌",
    layout="wide",
)

# ------------------------------------------------------------
# 기본 데이터
# ------------------------------------------------------------

def default_population():
    return pd.DataFrame({
        "지역명": [
            "상당구", "서원구", "흥덕구", "청원구",
            "오창읍", "오송읍", "가경동", "복대동",
            "성안동", "내덕동",
        ],
        "기준인구(명)": [
            194551, 182689, 292051, 186861,
            67832, 48348, 53342, 52000,
            15000, 30000,
        ],
        "위도": [
            36.633, 36.628, 36.635, 36.665,
            36.7153, 36.6205, 36.6240, 36.6355,
            36.6338, 36.6480,
        ],
        "경도": [
            127.490, 127.470, 127.430, 127.490,
            127.4258, 127.3274, 127.3900, 127.4221,
            127.4879, 127.4890,
        ],
    })


# ------------------------------------------------------------
# CSV 안전 읽기
# ------------------------------------------------------------

def read_csv_flexible(uploaded_file):
    if uploaded_file is None:
        return None, "파일이 없습니다."

    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()

        if not raw:
            return None, "업로드된 파일이 비어 있습니다."

        text = None

        for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None or not text.strip():
            return None, "CSV 내용을 읽을 수 없습니다."

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

    except Exception as e:
        return None, f"CSV 읽기 오류: {e}"


# ------------------------------------------------------------
# 인구 CSV 처리
# 지원:
# A. 행정구역(동읍면)별 + 항목 + 2012~2025
# B. 지역명 + 기준인구(명)
# ------------------------------------------------------------

def load_population(uploaded_file):
    df, error = read_csv_flexible(uploaded_file)

    if error:
        return None, error

    # 실제 시계열 형식
    if (
        "행정구역(동읍면)별" in df.columns
        and "항목" in df.columns
    ):
        total = df[
            df["항목"]
            .astype(str)
            .str.contains("총인구수", na=False)
        ].copy()

        years = [
            str(y)
            for y in range(2012, 2026)
            if str(y) in df.columns
        ]

        if total.empty:
            return None, "총인구수 행을 찾지 못했습니다."

        if not years:
            return None, "2012~2025 연도 열을 찾지 못했습니다."

        rows = []

        for _, row in total.iterrows():
            region = str(
                row["행정구역(동읍면)별"]
            ).strip()

            values = {}

            for year in years:
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
                "기준인구(명)": int(round(valid.iloc[-1])),
                **{
                    f"인구_{year}": values[year]
                    for year in years
                },
            })

        result = pd.DataFrame(rows)

        if result.empty:
            return None, "유효한 인구 데이터가 없습니다."

        return result, None

    # 단순 형식
    rename = {}

    for col in df.columns:
        normalized = (
            str(col)
            .replace(" ", "")
            .replace("_", "")
        )

        if normalized in {
            "지역명", "지역", "읍면동",
            "읍면동명", "행정동", "행정동명",
        }:
            rename[col] = "지역명"

        elif normalized in {
            "기준인구명", "인구수명",
            "인구수", "총인구수명", "인구",
        }:
            rename[col] = "기준인구(명)"

    df = df.rename(columns=rename)

    if {
        "지역명",
        "기준인구(명)",
    }.issubset(df.columns):

        df["기준인구(명)"] = pd.to_numeric(
            df["기준인구(명)"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("명", "", regex=False),
            errors="coerce",
        )

        df = df.dropna(
            subset=["지역명", "기준인구(명)"]
        ).copy()

        return df.reset_index(drop=True), None

    return None, (
        "인구 CSV 형식을 인식하지 못했습니다. "
        "행정구역(동읍면)별/항목/연도 형식 또는 "
        "지역명/기준인구(명) 형식을 사용하세요."
    )


# ------------------------------------------------------------
# 세션 상태
# ------------------------------------------------------------

if "population" not in st.session_state:
    st.session_state.population = None


# ============================================================
# 사이드바
# ============================================================

st.sidebar.title("🚌 대중교통 계획")

st.sidebar.markdown("### 📂 기준 데이터")

population_file = st.sidebar.file_uploader(
    "인구 CSV 업로드",
    type=["csv"],
    help=(
        "2012~2025 시계열 인구 CSV 또는 "
        "지역명/기준인구(명) CSV"
    ),
)

if population_file is not None:

    population, error = load_population(
        population_file
    )

    if population is not None:
        st.session_state.population = population

        st.sidebar.success(
            f"인구 {len(population):,}개 지역 연결"
        )

    else:
        st.sidebar.error(error)


if st.sidebar.button(
    "↺ 인구 데이터 초기화",
    use_container_width=True,
):
    st.session_state.population = None
    st.rerun()


population = st.session_state.population

if population is None:
    population = default_population()
    data_status = "기본 PoC 데이터"
else:
    data_status = "사용자 업로드 데이터"


# ============================================================
# 페이지
# ============================================================

page = st.sidebar.radio(
    "📑 계획 단계",
    [
        "🏠 종합 대시보드",
        "👥 인구·장래수요",
        "🔮 미래 시나리오",
        "🚌 대중교통 수요 시나리오",
        "🛣️ 노선 대안",
        "📊 대안 비교",
        "🏛️ 행정 의사결정",
    ],
)


# ============================================================
# ① 종합 대시보드
# ============================================================

if page == "🏠 종합 대시보드":

    st.title(
        "🚌 AI 기반 미래예측형 "
        "대중교통 의사결정 지원 플랫폼"
    )

    st.caption(
        "공모전용 PoC · 청주시 적용 시나리오"
    )

    st.info(
        "※ 본 프로토타입은 미래 정책 시나리오를 "
        "사용자가 직접 조정하고, 그 결과를 비교하는 "
        "의사결정 지원 구조를 보여주기 위한 PoC입니다."
    )

    total_population = int(
        population["기준인구(명)"].sum()
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "기준 인구",
        f"{total_population:,}명",
    )

    b.metric(
        "분석 지역",
        f"{len(population):,}개",
    )

    c.metric(
        "승하차 데이터",
        "직접 조정",
    )

    d.metric(
        "데이터 상태",
        data_status,
    )

    st.markdown("---")

    st.subheader("🔄 플랫폼 작동 구조")

    st.markdown(
        """
**현재 도시현황**
→ **미래 인구 시나리오**
→ **승하차 수요 시나리오**
→ **미래 이동수요 예측**
→ **노선 대안 생성**
→ **효과 비교**
→ **행정기관 최종 결정**
"""
    )

    st.success(
        "핵심: AI가 정책을 대신 결정하는 것이 아니라 "
        "다양한 미래 상황을 빠르게 비교하여 "
        "행정기관의 정책 판단을 지원합니다."
    )


# ============================================================
# ② 인구·장래수요
# ============================================================

elif page == "👥 인구·장래수요":

    st.header("👥 인구 및 장래수요")

    region = st.selectbox(
        "분석 지역",
        population["지역명"].tolist(),
    )

    row = population[
        population["지역명"] == region
    ].iloc[0]

    base_population = int(
        row["기준인구(명)"]
    )

    st.metric(
        "기준 인구",
        f"{base_population:,}명",
    )

    year_cols = [
        col
        for col in population.columns
        if str(col).startswith("인구_")
    ]

    if year_cols:

        chart = pd.to_numeric(
            row[year_cols],
            errors="coerce",
        ).dropna()

        chart.index = [
            int(
                str(x)
                .replace("인구_", "")
            )
            for x in chart.index
        ]

        st.line_chart(chart)

    st.markdown("---")

    target_year = st.selectbox(
        "예측 연도",
        [2027, 2030, 2035, 2040],
        index=1,
    )

    growth = st.slider(
        "미래 인구 변화율 (%)",
        -50,
        100,
        15,
        1,
    )

    future_population = round(
        base_population
        * (1 + growth / 100)
    )

    a, b = st.columns(2)

    a.metric(
        f"{target_year}년 시나리오 인구",
        f"{future_population:,}명",
        f"{growth:+d}%",
    )

    if year_cols and len(year_cols) >= 3:

        values = pd.to_numeric(
            row[year_cols],
            errors="coerce",
        ).dropna()

        x = np.arange(len(values))

        model = LinearRegression()

        model.fit(
            x.reshape(-1, 1),
            values.values,
        )

        reference_prediction = max(
            0,
            int(
                round(
                    model.predict(
                        [[
                            len(values)
                            + (
                                target_year
                                - int(year_cols[-1].replace("인구_", ""))
                            )
                        ]]
                    )[0]
                )
            ),
        )

        b.metric(
            "추세 기반 참고예측",
            f"{reference_prediction:,}명",
        )

    st.caption(
        "※ 사용자 조정값은 미래 정책 시나리오를 "
        "실험하기 위한 PoC 변수입니다."
    )


# ============================================================
# ③ 미래 시나리오
# ============================================================

elif page == "🔮 미래 시나리오":

    st.header("🔮 미래 도시 시나리오 설정")

    st.write(
        "정책 담당자가 미래 상황을 직접 설정하고 "
        "그 변화가 대중교통 수요에 미치는 영향을 "
        "실시간으로 확인합니다."
    )

    region = st.selectbox(
        "시나리오 대상 지역",
        population["지역명"].tolist(),
    )

    row = population[
        population["지역명"] == region
    ].iloc[0]

    base_population = int(
        row["기준인구(명)"]
    )

    st.markdown("### 👥 ① 인구 변화")

    population_change = st.slider(
        "인구 변화율 (%)",
        -50,
        100,
        15,
        1,
    )

    st.markdown("### 🏗️ ② 도시개발")

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

    scenario_population = round(
        base_population
        * (1 + population_change / 100)
        + development_population
    )

    st.markdown("### 🚦 ③ 교통환경")

    congestion = st.slider(
        "교통 혼잡 변화율 (%)",
        -30,
        100,
        10,
        1,
    )

    st.markdown("### 📅 ④ 예측연도")

    target_year = st.selectbox(
        "예측 연도",
        [2027, 2030, 2035, 2040],
        index=1,
    )

    a, b, c = st.columns(3)

    a.metric(
        "현재 인구",
        f"{base_population:,}명",
    )

    b.metric(
        "미래 시나리오 인구",
        f"{scenario_population:,}명",
    )

    c.metric(
        "예측 연도",
        f"{target_year}년",
    )

    st.success(
        f"**{region} / {development_type} / "
        f"{target_year}년** 시나리오가 설정되었습니다."
    )


# ============================================================
# ④ 대중교통 수요 시나리오
# ============================================================

elif page == "🚌 대중교통 수요 시나리오":

    st.header("🚌 대중교통 수요 시나리오")

    st.info(
        "승하차 CSV는 사용하지 않습니다. "
        "공모전 PoC에서는 정책 담당자가 기준 수요와 "
        "미래 수요 변화율을 직접 조정할 수 있도록 구성했습니다."
    )

    st.markdown("### ① 현재 기준 수요")

    a, b = st.columns(2)

    with a:
        base_boarding = st.number_input(
            "일평균 승차 수요 (건)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000,
        )

    with b:
        base_alighting = st.number_input(
            "일평균 하차 수요 (건)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000,
        )

    current_total = (
        base_boarding
        + base_alighting
    )

    st.metric(
        "현재 일평균 총 승하차",
        f"{current_total:,}건",
    )

    st.markdown("---")

    st.markdown("### ② 미래 수요 변화")

    demand_change = st.slider(
        "대중교통 수요 변화율 (%)",
        -50,
        150,
        15,
        1,
    )

    development_effect = st.slider(
        "개발사업에 따른 추가 수요 영향 (%)",
        0,
        100,
        10,
        1,
    )

    congestion_effect = st.slider(
        "승용차 혼잡 증가에 따른 대중교통 전환 (%)",
        0,
        50,
        5,
        1,
    )

    future_boarding = round(
        base_boarding
        * (
            1
            + demand_change / 100
            + development_effect / 100
            + congestion_effect / 100
        )
    )

    future_alighting = round(
        base_alighting
        * (
            1
            + demand_change / 100
            + development_effect / 100
            + congestion_effect / 100
        )
    )

    future_total = (
        future_boarding
        + future_alighting
    )

    st.markdown("### ③ 미래 수요 결과")

    x, y, z = st.columns(3)

    x.metric(
        "미래 승차",
        f"{future_boarding:,}건",
        f"{future_boarding - base_boarding:+,}",
    )

    y.metric(
        "미래 하차",
        f"{future_alighting:,}건",
        f"{future_alighting - base_alighting:+,}",
    )

    z.metric(
        "미래 총 승하차",
        f"{future_total:,}건",
        f"{future_total - current_total:+,}",
    )

    st.markdown("---")

    comparison = pd.DataFrame({
        "구분": [
            "현재",
            "미래 시나리오",
        ],
        "승차": [
            base_boarding,
            future_boarding,
        ],
        "하차": [
            base_alighting,
            future_alighting,
        ],
        "총 승하차": [
            current_total,
            future_total,
        ],
    })

    st.dataframe(
        comparison.style.format({
            "승차": "{:,.0f}",
            "하차": "{:,.0f}",
            "총 승하차": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        comparison.set_index("구분")[
            ["승차", "하차"]
        ]
    )


# ============================================================
# ⑤ 노선 대안
# ============================================================

elif page == "🛣️ 노선 대안":

    st.header("🛣️ 노선 대안 시뮬레이션")

    st.caption(
        "미래 수요 시나리오에 따라 정책 대안을 비교하기 위한 PoC입니다."
    )

    scenario = st.selectbox(
        "정책 대안",
        [
            "현행 유지",
            "대안 A · 거점 직결형",
            "대안 B · 간선 연장형",
            "대안 C · 환승 최적화형",
        ],
    )

    info = {
        "현행 유지":
            "현재 노선을 유지합니다.",
        "대안 A · 거점 직결형":
            "주요 생활권·산업·환승거점을 직접 연결합니다.",
        "대안 B · 간선 연장형":
            "기존 간선노선을 미래 개발지역까지 연장합니다.",
        "대안 C · 환승 최적화형":
            "환승 결절점을 중심으로 배차와 환승을 최적화합니다.",
    }

    st.info(info[scenario])

    st.markdown("### 🚌 운영 변수")

    frequency = st.slider(
        "배차간격 (분)",
        5,
        60,
        15,
        1,
    )

    speed = st.slider(
        "평균 운행속도 (km/h)",
        10,
        50,
        25,
        1,
    )

    transfer = st.slider(
        "평균 환승시간 (분)",
        0,
        20,
        5,
        1,
    )

    congestion = st.slider(
        "교통 혼잡 영향 (%)",
        0,
        100,
        15,
        1,
    )

    estimated_wait = (
        frequency / 2
    ) * (
        1 + congestion / 100
    )

    estimated_time = (
        30
        * (25 / speed)
        * (1 + congestion / 100)
    )

    st.markdown("### 📊 예상 운영지표")

    a, b, c = st.columns(3)

    a.metric(
        "평균 대기시간",
        f"{estimated_wait:.1f}분",
    )

    b.metric(
        "예상 통행시간",
        f"{estimated_time:.1f}분",
    )

    c.metric(
        "환승시간",
        f"{transfer:.1f}분",
    )


# ============================================================
# ⑥ 대안 비교
# ============================================================

elif page == "📊 대안 비교":

    st.header("📊 정책 대안 종합 비교")

    st.markdown(
        "기준 수요와 미래 수요 변수를 직접 조정하여 "
        "정책 대안의 상대적인 효과를 비교합니다."
    )

    st.markdown("### 🚌 미래 수요 설정")

    base_demand = st.number_input(
        "기준 일평균 총 승하차 (건)",
        0,
        1000000,
        100000,
        1000,
    )

    demand_growth = st.slider(
        "미래 수요 증가율 (%)",
        -50,
        150,
        20,
        1,
    )

    future_demand = round(
        base_demand
        * (1 + demand_growth / 100)
    )

    st.metric(
        "미래 예상 총 승하차",
        f"{future_demand:,}건",
        f"{demand_growth:+d}%",
    )

    st.markdown("---")

    scenarios = pd.DataFrame({
        "정책 대안": [
            "현행 유지",
            "대안 A · 거점 직결형",
            "대안 B · 간선 연장형",
            "대안 C · 환승 최적화형",
        ],
        "예상 일일 수요": [
            future_demand,
            round(future_demand * 1.18),
            round(future_demand * 1.10),
            round(future_demand * 1.15),
        ],
        "평균 통행시간": [
            45,
            31,
            36,
            34,
        ],
        "평균 대기시간": [
            10,
            7,
            8,
            6,
        ],
        "환승시간": [
            8,
            7,
            6,
            4,
        ],
        "추가 운영비": [
            0.0,
            4.5,
            2.8,
            3.6,
        ],
        "탄소지수": [
            1.00,
            0.78,
            0.86,
            0.72,
        ],
    })

    st.sidebar.markdown("### ⚖️ 정책 평가 가중치")

    w_demand = st.sidebar.slider(
        "수요 효과",
        0.0, 1.0, 0.30, 0.05,
    )

    w_time = st.sidebar.slider(
        "시간 절감",
        0.0, 1.0, 0.30, 0.05,
    )

    w_cost = st.sidebar.slider(
        "운영비",
        0.0, 1.0, 0.20, 0.05,
    )

    w_carbon = st.sidebar.slider(
        "탄소",
        0.0, 1.0, 0.20, 0.05,
    )

    total = (
        w_demand
        + w_time
        + w_cost
        + w_carbon
    )

    if total == 0:
        total = 1

    demand_score = (
        scenarios["예상 일일 수요"]
        / scenarios["예상 일일 수요"].max()
    )

    time_value = (
        scenarios["평균 통행시간"]
        + scenarios["평균 대기시간"]
    )

    time_score = (
        1
        - time_value / time_value.max()
    )

    cost_score = (
        1
        - scenarios["추가 운영비"]
        / max(
            scenarios["추가 운영비"].max(),
            1,
        )
    )

    carbon_score = (
        1
        - scenarios["탄소지수"]
        / scenarios["탄소지수"].max()
    )

    scenarios["종합점수"] = (
        w_demand * demand_score
        + w_time * time_score
        + w_cost * cost_score
        + w_carbon * carbon_score
    ) / total

    best = scenarios.loc[
        scenarios["종합점수"].idxmax(),
        "정책 대안",
    ]

    def highlight(row):
        if row["정책 대안"] == best:
            return [
                "background-color: #E8F5E9"
            ] * len(row)
        return [""] * len(row)

    st.dataframe(
        scenarios.style.apply(
            highlight,
            axis=1,
        ).format({
            "예상 일일 수요": "{:,.0f}",
            "평균 통행시간": "{:.1f}",
            "평균 대기시간": "{:.1f}",
            "환승시간": "{:.1f}",
            "추가 운영비": "{:.1f}",
            "탄소지수": "{:.2f}",
            "종합점수": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        f"🤖 현재 정책 가중치 기준 추천안: **{best}**"
    )

    st.caption(
        "※ 추천 결과는 정책 시나리오 비교를 위한 "
        "PoC 계산값이며, 최종 정책 결정은 행정기관이 수행합니다."
    )


# ============================================================
# ⑦ 행정 의사결정
# ============================================================

elif page == "🏛️ 행정 의사결정":

    st.header("🏛️ 행정기관 최종 의사결정")

    st.markdown(
        """
### AI의 역할

AI는 정책을 대신 결정하지 않습니다.

**① 미래 시나리오 설정**
→ **② 미래 이동수요 예측**
→ **③ 정책 대안 비교**
→ **④ 효과·비용 분석**
→ **⑤ 행정기관 최종 판단**
"""
    )

    decision = st.radio(
        "최종 정책 상태",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 분석",
        ],
    )

    if decision == "정책 대안 채택":
        st.success(
            "정책 대안을 채택합니다. "
            "정책 시행 이후 운영 데이터와 시민 피드백을 "
            "다음 정책 분석에 반영합니다."
        )

    elif decision == "추가 분석":
        st.warning(
            "추가적인 미래 시나리오와 "
            "전문가 검토가 필요합니다."
        )

    else:
        st.info(
            "AI 분석 결과를 참고하여 "
            "행정기관이 최종 판단합니다."
        )

    st.markdown("---")

    st.subheader("🔄 정책 선순환")

    st.markdown(
        """
**정책 수립**
↓  
**정책 시행**
↓  
**운영 데이터 수집**
↓  
**시민 피드백**
↓  
**AI 분석 및 재학습**
↓  
**다음 정책 개선**
"""
    )

    st.success(
        "목표: 일회성 노선 개편이 아니라 "
        "지속적으로 개선되는 데이터 기반 대중교통 행정체계"
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼 · 공모전용 PoC"
)
