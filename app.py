
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
# 핵심
# - 인구 CSV 선택적 업로드
# - 승하차 CSV 없음 → 사용자가 직접 수요 조정
# - 정류장/노선 CSV 업로드
# - 실제 정류장 순서 기반 노선 지도
# - 시민 피드백 입력 및 정책 재평가
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
            "오창읍", "오송읍", "가경동", "복대동",
            "성안동", "내덕동"
        ],
        "기준인구(명)": [
            71000, 31000, 52000, 53000, 15000, 30000
        ],
        "위도": [
            36.7153, 36.6205, 36.6240,
            36.6355, 36.6338, 36.6480
        ],
        "경도": [
            127.4258, 127.3274, 127.3900,
            127.4221, 127.4879, 127.4890
        ],
    })


def default_stops():
    """
    데모용 fallback.
    실제 공모전 제출에서는 청주시 공식 BIS CSV를
    '청주시 정류장 CSV' 업로더에 넣어 사용하는 것을 권장합니다.
    """
    return pd.DataFrame({
        "노선ID": (
            ["P001"] * 6
            + ["P002"] * 5
            + ["P003"] * 5
        ),
        "정류장순번": (
            list(range(1, 7))
            + list(range(1, 6))
            + list(range(1, 6))
        ),
        "정류장ID": [
            "DEMO001", "DEMO002", "DEMO003", "DEMO004",
            "DEMO005", "DEMO006", "DEMO007", "DEMO008",
            "DEMO009", "DEMO010", "DEMO011", "DEMO012",
            "DEMO013", "DEMO014", "DEMO015", "DEMO016",
        ],
        "정류장명": [
            "오창산단", "오창읍사무소", "청주대학교",
            "내덕동", "성안길", "청주터미널",
            "오송역", "가경동", "청주터미널",
            "사창사거리", "충북대학교",
            "오창산단", "내수역", "청주대학교",
            "성안길", "용암동",
        ],
        "위도": [
            36.7153, 36.7050, 36.6500,
            36.6480, 36.6338, 36.6271,
            36.6205, 36.6240, 36.6271,
            36.6342, 36.6280, 36.7153,
            36.6900, 36.6500, 36.6338, 36.6080,
        ],
        "경도": [
            127.4258, 127.4400, 127.4950,
            127.4890, 127.4879, 127.4321,
            127.3274, 127.3900, 127.4321,
            127.4567, 127.4580, 127.4258,
            127.5050, 127.4950, 127.4879, 127.5100,
        ],
    })


def normalize_xy(value):
    """WGS84 decimal degree 또는 단순 숫자 문자열을 안전하게 변환."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(",", "")

    try:
        return float(text)
    except ValueError:
        pass

    # DMS: 127°25'12.3" 형태
    import re

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(nums) >= 3:
        deg = float(nums[0])
        minute = float(nums[1])
        second = float(nums[2])
        sign = -1 if deg < 0 else 1
        return sign * (
            abs(deg)
            + minute / 60
            + second / 3600
        )

    return np.nan


def load_stops(uploaded):
    df, error = read_csv(uploaded)

    if error:
        return None, error

    rename = {}

    for col in df.columns:
        raw = str(col).strip()
        n = (
            raw.replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .lower()
        )

        if n in {
            "서비스id",
            "서비스아이디",
            "승강장id",
            "정류장id",
            "정류소id",
            "stationid",
            "stopid",
            "busstopid",
        }:
            rename[col] = "정류장ID"

        elif n in {
            "노선id", "노선아이디", "routeid",
            "route", "노선번호", "노선명",
        }:
            rename[col] = "노선ID"

        elif n in {
            "정류장순번", "정류소순번",
            "순번", "stopsequence",
            "stopseq", "sequence",
        }:
            rename[col] = "정류장순번"

        elif n in {
            "정류장명", "정류소명",
            "정류장", "정류소",
            "stopname",
        }:
            rename[col] = "정류장명"

        elif n in {
            "좌표x", "x좌표", "경도",
            "lon", "lng", "longitude",
        }:
            rename[col] = "경도"

        elif n in {
            "좌표y", "y좌표", "위도",
            "lat", "latitude",
        }:
            rename[col] = "위도"

    df = df.rename(columns=rename)

    # 청주시 공식 파일은 서비스ID, 정류소명, 좌표(X), 좌표(Y) 구조다.
    # 노선/순번은 공식 정류장 위치 데이터에 없을 수 있으므로
    # 노선 지도는 별도의 노선-정류장 데이터가 있을 때 활성화한다.
    if not {
        "정류장명", "위도", "경도"
    }.issubset(df.columns):
        return None, (
            "청주시 정류장 CSV를 인식하지 못했습니다.\n"
            "청주시 공식 BIS 파일은 서비스ID, 정류소명, "
            "좌표(X), 좌표(Y) 컬럼을 사용합니다."
        )

    if "정류장ID" not in df.columns:
        df["정류장ID"] = (
            "STOP_"
            + pd.Series(
                range(1, len(df) + 1),
                index=df.index,
            ).astype(str)
        )

    if "노선ID" not in df.columns:
        df["노선ID"] = "정류장 위치 데이터"

    if "정류장순번" not in df.columns:
        df["정류장순번"] = (
            df.groupby(
                "노선ID",
                sort=False,
            ).cumcount()
            + 1
        )

    df["정류장ID"] = (
        df["정류장ID"]
        .fillna("")
        .astype(str)
    )

    df["노선ID"] = (
        df["노선ID"]
        .fillna("정류장 위치 데이터")
        .astype(str)
    )

    df["정류장명"] = (
        df["정류장명"]
        .fillna("")
        .astype(str)
    )

    df["위도"] = df["위도"].apply(
        normalize_xy
    )

    df["경도"] = df["경도"].apply(
        normalize_xy
    )

    df["정류장순번"] = pd.to_numeric(
        df["정류장순번"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["위도", "경도"]
    ).copy()

    # 한국 주변의 WGS84 좌표만 남긴다.
    df = df[
        df["위도"].between(33, 39)
        & df["경도"].between(124, 132)
    ].copy()

    if df.empty:
        return None, (
            "유효한 WGS84 좌표가 없습니다. "
            "좌표(X)=경도, 좌표(Y)=위도인지 확인하세요."
        )

    df["정류장순번"] = (
        df["정류장순번"]
        .fillna(0)
        .astype(int)
    )

    return df.reset_index(drop=True), None


# ------------------------------------------------------------
# CSV 공통 처리
# ------------------------------------------------------------

def read_csv(uploaded):
    if uploaded is None:
        return None, "파일이 없습니다."

    try:
        uploaded.seek(0)
        raw = uploaded.read()

        if not raw:
            return None, "파일이 비어 있습니다."

        text = None

        for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                pass

        if text is None:
            return None, "CSV 인코딩을 읽을 수 없습니다."

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

    except Exception as e:
        return None, f"CSV 읽기 오류: {e}"


def load_population(uploaded):
    df, error = read_csv(uploaded)

    if error:
        return None, error

    # 단순형
    rename = {}

    for col in df.columns:
        n = (
            str(col)
            .replace(" ", "")
            .replace("_", "")
        )

        if n in {
            "지역명", "지역", "읍면동",
            "읍면동명", "행정동", "행정동명"
        }:
            rename[col] = "지역명"

        if n in {
            "기준인구명", "인구수명",
            "인구수", "총인구수명", "인구"
        }:
            rename[col] = "기준인구(명)"

    df = df.rename(columns=rename)

    if {"지역명", "기준인구(명)"}.issubset(df.columns):
        df["기준인구(명)"] = pd.to_numeric(
            df["기준인구(명)"]
            .astype(str)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )

        df = df.dropna(
            subset=["지역명", "기준인구(명)"]
        ).copy()

        return df.reset_index(drop=True), None

    # 통계청형 / 행정구역 시계열
    if (
        "행정구역(동읍면)별" in df.columns
        and "항목" in df.columns
    ):
        total = df[
            df["항목"].astype(str).str.contains(
                "총인구수",
                na=False,
            )
        ].copy()

        years = [
            str(y)
            for y in range(2012, 2026)
            if str(y) in df.columns
        ]

        if total.empty or not years:
            return None, "인구 시계열 형식을 확인해주세요."

        rows = []

        for _, row in total.iterrows():
            values = {}

            for year in years:
                values[year] = pd.to_numeric(
                    str(row[year]).replace(",", ""),
                    errors="coerce",
                )

            valid = pd.Series(values).dropna()

            if valid.empty:
                continue

            rows.append({
                "지역명": str(
                    row["행정구역(동읍면)별"]
                ).strip(),
                "기준인구(명)": float(valid.iloc[-1]),
            })

        result = pd.DataFrame(rows)

        if result.empty:
            return None, "유효한 인구 데이터가 없습니다."

        return result.reset_index(drop=True), None

    return None, (
        "인구 CSV 형식을 인식하지 못했습니다. "
        "지역명/기준인구(명) 형식을 권장합니다."
    )


def load_stops(uploaded):
    df, error = read_csv(uploaded)

    if error:
        return None, error

    rename = {}

    for col in df.columns:
        n = (
            str(col)
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .lower()
        )

        if n in {
            "노선id", "노선아이디", "routeid",
            "route", "노선번호", "노선명"
        }:
            rename[col] = "노선ID"

        elif n in {
            "정류장순번", "정류소순번",
            "순번", "stopsequence",
            "stopseq", "sequence"
        }:
            rename[col] = "정류장순번"

        elif n in {
            "정류장명", "정류소명",
            "정류장", "정류소",
            "stopname"
        }:
            rename[col] = "정류장명"

        elif n in {"위도", "lat", "latitude"}:
            rename[col] = "위도"

        elif n in {
            "경도", "lon", "lng", "longitude"
        }:
            rename[col] = "경도"

    df = df.rename(columns=rename)

    required = {
        "정류장명",
        "위도",
        "경도",
    }

    if not required.issubset(df.columns):
        return None, (
            "정류장 CSV에는 최소한 "
            "정류장명, 위도, 경도가 필요합니다. "
            "노선 경로 표현에는 노선ID와 정류장순번도 필요합니다."
        )

    if "노선ID" not in df.columns:
        df["노선ID"] = "P001"

    if "정류장순번" not in df.columns:
        df["정류장순번"] = (
            df.groupby(
                "노선ID",
                sort=False,
            ).cumcount()
            + 1
        )

    df["노선ID"] = (
        df["노선ID"]
        .fillna("P001")
        .astype(str)
    )

    df["정류장명"] = (
        df["정류장명"]
        .fillna("")
        .astype(str)
    )

    df["위도"] = pd.to_numeric(
        df["위도"],
        errors="coerce",
    )

    df["경도"] = pd.to_numeric(
        df["경도"],
        errors="coerce",
    )

    df["정류장순번"] = pd.to_numeric(
        df["정류장순번"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["위도", "경도"]
    ).copy()

    df["정류장순번"] = (
        df["정류장순번"]
        .fillna(0)
        .astype(int)
    )

    if df.empty:
        return None, "유효한 정류장 좌표가 없습니다."

    return df.reset_index(drop=True), None


# ------------------------------------------------------------
# 세션 상태
# ------------------------------------------------------------

if "population" not in st.session_state:
    st.session_state.population = None

if "stops" not in st.session_state:
    st.session_state.stops = None

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "scenario" not in st.session_state:
    st.session_state.scenario = {
        "demand_growth": 20,
        "development_effect": 10,
        "congestion_effect": 5,
    }


# ============================================================
# 사이드바
# ============================================================

st.sidebar.title("🚌 대중교통 계획")

st.sidebar.markdown("### 📂 인구 데이터")

population_file = st.sidebar.file_uploader(
    "인구 CSV",
    type=["csv"],
)

if population_file is not None:
    pop, error = load_population(
        population_file
    )

    if pop is not None:
        st.session_state.population = pop
        st.sidebar.success(
            f"{len(pop):,}개 지역 연결"
        )
    else:
        st.sidebar.error(error)

st.sidebar.markdown("### 🗺️ 청주시 전체 정류장")

st.sidebar.caption(
    "청주시 공식 BIS 정류장 데이터 권장 · "
    "CSV 업로드 시 전체 정류장을 지도에 반영합니다."
)

st.sidebar.caption(
    "공식 데이터: 청주시 버스정보시스템(BIS), "
    "2025-04 기준 3,402개 행"
)

stops_file = st.sidebar.file_uploader(
    "정류장/노선 CSV",
    type=["csv"],
    help=(
        "권장 컬럼: 노선ID, 정류장순번, "
        "정류장명, 위도, 경도"
    ),
)

if stops_file is not None:
    stop_data, error = load_stops(
        stops_file
    )

    if stop_data is not None:
        st.session_state.stops = stop_data

        st.sidebar.success(
            f"정류장 {len(stop_data):,}개 · "
            f"노선 {stop_data['노선ID'].nunique():,}개"
        )

    else:
        st.sidebar.error(error)

if st.sidebar.button(
    "↺ 데이터 초기화",
    use_container_width=True,
):
    st.session_state.population = None
    st.session_state.stops = None
    st.rerun()


population = (
    st.session_state.population
    if st.session_state.population is not None
    else default_population()
)

stops = (
    st.session_state.stops
    if st.session_state.stops is not None
    else default_stops()
)

stops_is_real = st.session_state.stops is not None

if stops_is_real:
    stops_source_label = "청주시 BIS CSV"
else:
    stops_source_label = "PoC 데모 정류장"


# ============================================================
# 페이지
# ============================================================

page = st.sidebar.radio(
    "📑 계획 단계",
    [
        "🏠 종합 대시보드",
        "👥 인구·장래수요",
        "🔮 미래 시나리오",
        "🚌 대중교통 수요",
        "🗺️ 정류장·노선 지도",
        "🛣️ 노선 대안",
        "📊 대안 비교",
        "📣 시민 피드백",
        "🏛️ 행정 의사결정",
    ],
)


# ============================================================
# 공통 시나리오 계산
# ============================================================

demand_growth = st.session_state.scenario[
    "demand_growth"
]

development_effect = st.session_state.scenario[
    "development_effect"
]

congestion_effect = st.session_state.scenario[
    "congestion_effect"
]

scenario_multiplier = (
    1
    + demand_growth / 100
    + development_effect / 100
    + congestion_effect / 100
)


# ============================================================
# ① 종합 대시보드
# ============================================================

if page == "🏠 종합 대시보드":

    st.title(
        "🚌 AI 기반 미래예측형 "
        "대중교통 의사결정 지원 플랫폼"
    )

    if stops_is_real:
        st.success(
            f"청주시 BIS 정류장 데이터 연결 완료 · "
            f"{len(stops):,}개 정류장"
        )
    else:
        st.warning(
            "현재는 PoC 데모 정류장을 사용 중입니다. "
            "청주시 공식 BIS CSV를 업로드하면 실제 전체 정류장이 반영됩니다."
        )

    st.caption(
        "공모전용 PoC · 청주시 적용 시나리오"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "기준 인구",
        f"{population['기준인구(명)'].sum():,.0f}명",
    )

    b.metric(
        "정류장",
        f"{len(stops):,}개",
    )

    c.metric(
        "노선",
        f"{stops['노선ID'].nunique():,}개",
    )

    d.metric(
        "시민 피드백",
        f"{len(st.session_state.feedback):,}건",
    )

    st.markdown("---")

    st.subheader("🔄 데이터 → 정책 → 피드백 순환")

    st.markdown(
        """
**① 도시·인구 변화**
→ **② 미래 이동수요 예측**
→ **③ 정류장·노선 분석**
→ **④ 정책 대안 생성**
→ **⑤ 효과 비교**
→ **⑥ 정책 시행**
→ **⑦ 시민 피드백**
→ **⑧ 다음 정책 재평가**
"""
    )

    st.success(
        "핵심: AI가 정책을 대신 결정하는 것이 아니라 "
        "미래 상황과 정책 대안을 비교하여 "
        "행정기관의 의사결정을 지원합니다."
    )

    st.markdown("### 🗺️ 현재 정류장 분포")

    m = folium.Map(
        location=[
            stops["위도"].mean(),
            stops["경도"].mean(),
        ],
        zoom_start=11,
        tiles="OpenStreetMap",
    )

    for _, row in stops.iterrows():
        folium.CircleMarker(
            location=[
                float(row["위도"]),
                float(row["경도"]),
            ],
            radius=3,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.65,
            tooltip=(
                f"{row['정류장명']} · "
                f"ID {row['정류장ID']}"
            ),
        ).add_to(m)

    st_folium(
        m,
        height=620,
        use_container_width=True,
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

    base = float(
        row["기준인구(명)"]
    )

    growth = st.slider(
        "미래 인구 변화율 (%)",
        -50,
        100,
        15,
        1,
    )

    target_year = st.selectbox(
        "예측 연도",
        [2027, 2030, 2035, 2040],
        index=1,
    )

    future = round(
        base * (1 + growth / 100)
    )

    a, b = st.columns(2)

    a.metric(
        "현재 인구",
        f"{base:,.0f}명",
    )

    b.metric(
        f"{target_year}년 시나리오",
        f"{future:,.0f}명",
        f"{growth:+d}%",
    )

    st.dataframe(
        population,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ③ 미래 시나리오
# ============================================================

elif page == "🔮 미래 시나리오":

    st.header("🔮 미래 도시·교통 시나리오")

    st.write(
        "정책 담당자가 미래 상황을 직접 조정하고 "
        "대중교통 수요 변화를 확인합니다."
    )

    region = st.selectbox(
        "시나리오 대상 지역",
        population["지역명"].tolist(),
    )

    base_population = float(
        population.loc[
            population["지역명"] == region,
            "기준인구(명)",
        ].iloc[0]
    )

    population_change = st.slider(
        "인구 변화율 (%)",
        -50,
        100,
        15,
        1,
    )

    development = st.selectbox(
        "도시개발 유형",
        [
            "개발 없음",
            "공동주택 개발",
            "산업단지 조성",
            "대규모 상업시설",
            "복합개발",
        ],
    )

    additional_population = st.slider(
        "개발에 따른 추가 유입인구 (명)",
        0,
        30000,
        5000,
        500,
    )

    congestion = st.slider(
        "교통 혼잡 증가율 (%)",
        0,
        100,
        10,
        1,
    )

    demand = st.slider(
        "대중교통 수요 변화율 (%)",
        -50,
        150,
        20,
        1,
    )

    future_population = round(
        base_population
        * (1 + population_change / 100)
        + additional_population
    )

    st.session_state.scenario = {
        "demand_growth": demand,
        "development_effect": (
            additional_population
            / max(base_population, 1)
            * 100
        ),
        "congestion_effect": congestion,
    }

    a, b, c = st.columns(3)

    a.metric(
        "미래 인구",
        f"{future_population:,}명",
    )

    b.metric(
        "대중교통 수요 변화",
        f"{demand:+d}%",
    )

    c.metric(
        "개발 유형",
        development,
    )

    st.success(
        "시나리오가 저장되었습니다. "
        "다음 단계에서 수요·노선 대안을 비교할 수 있습니다."
    )


# ============================================================
# ④ 대중교통 수요
# ============================================================

elif page == "🚌 대중교통 수요":

    st.header("🚌 대중교통 수요 시나리오")

    st.info(
        "승하차 CSV는 사용하지 않습니다. "
        "기준 승차·하차 수요를 직접 입력하고 "
        "미래 변화율을 조정합니다."
    )

    base_boarding = st.number_input(
        "기준 일평균 승차 수요 (건)",
        0,
        1000000,
        50000,
        1000,
    )

    base_alighting = st.number_input(
        "기준 일평균 하차 수요 (건)",
        0,
        1000000,
        50000,
        1000,
    )

    demand_change = st.slider(
        "미래 수요 변화율 (%)",
        -50,
        150,
        int(demand_growth),
        1,
    )

    development_effect_input = st.slider(
        "개발사업 추가 영향 (%)",
        0,
        100,
        int(max(0, development_effect)),
        1,
    )

    congestion_input = st.slider(
        "승용차 혼잡에 따른 전환 (%)",
        0,
        50,
        int(max(0, congestion_effect)),
        1,
    )

    multiplier = (
        1
        + demand_change / 100
        + development_effect_input / 100
        + congestion_input / 100
    )

    future_boarding = max(
        0,
        round(base_boarding * multiplier),
    )

    future_alighting = max(
        0,
        round(base_alighting * multiplier),
    )

    current_total = (
        base_boarding
        + base_alighting
    )

    future_total = (
        future_boarding
        + future_alighting
    )

    a, b, c = st.columns(3)

    a.metric(
        "현재 총 승하차",
        f"{current_total:,}건",
    )

    b.metric(
        "미래 총 승하차",
        f"{future_total:,}건",
    )

    c.metric(
        "총 수요 변화",
        f"{future_total-current_total:+,}건",
    )

    result = pd.DataFrame({
        "구분": ["현재", "미래"],
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
        result.style.format({
            "승차": "{:,.0f}",
            "하차": "{:,.0f}",
            "총 승하차": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ⑤ 정류장·노선 지도
# ============================================================

elif page == "🗺️ 정류장·노선 지도":

    st.header("🗺️ 청주시 전체 정류장 지도")

    st.caption(
        f"데이터: {stops_source_label} · "
        f"정류장 {len(stops):,}개"
    )

    search = st.text_input(
        "🔎 정류장 검색",
        placeholder="정류장명 또는 정류장ID 입력",
    )

    filtered = stops.copy()

    if search.strip():
        q = search.strip().lower()

        filtered = filtered[
            filtered["정류장명"]
            .str.lower()
            .str.contains(q, na=False)
            |
            filtered["정류장ID"]
            .str.lower()
            .str.contains(q, na=False)
        ].copy()

    a, b, c = st.columns(3)

    a.metric(
        "전체 정류장",
        f"{len(stops):,}개",
    )

    b.metric(
        "검색 결과",
        f"{len(filtered):,}개",
    )

    c.metric(
        "데이터 출처",
        "청주시 BIS" if stops_is_real else "PoC",
    )

    if filtered.empty:
        st.warning("검색 결과가 없습니다.")
    else:

        center = [
            filtered["위도"].mean(),
            filtered["경도"].mean(),
        ]

        m = folium.Map(
            location=center,
            zoom_start=11,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # 검색 결과가 많아도 지도 성능을 고려하여
        # 전체 정류장은 가벼운 CircleMarker로 표시한다.
        for _, row in filtered.iterrows():

            folium.CircleMarker(
                location=[
                    float(row["위도"]),
                    float(row["경도"]),
                ],
                radius=4 if len(filtered) < 1000 else 3,
                color="#2563eb",
                fill=True,
                fill_color="#2563eb",
                fill_opacity=0.7,
                tooltip=(
                    f"{row['정류장명']} · "
                    f"{row['정류장ID']}"
                ),
                popup=(
                    f"<b>{row['정류장명']}</b><br>"
                    f"정류장 ID: {row['정류장ID']}<br>"
                    f"위도: {row['위도']:.6f}<br>"
                    f"경도: {row['경도']:.6f}"
                ),
            ).add_to(m)

        st_folium(
            m,
            height=680,
            use_container_width=True,
        )

        st.markdown("### 📋 정류장 정보")

        st.dataframe(
            filtered[
                [
                    "정류장ID",
                    "정류장명",
                    "위도",
                    "경도",
                ]
            ].sort_values("정류장명"),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    st.subheader("🛣️ 실제 노선 경로 표시")

    has_route_sequence = (
        stops["노선ID"].nunique() > 1
        and stops["정류장순번"].max() > 1
        and stops["노선ID"].ne(
            "정류장 위치 데이터"
        ).any()
    )

    if has_route_sequence:

        route_ids = (
            stops["노선ID"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        selected_route = st.selectbox(
            "노선 선택",
            route_ids,
        )

        route = stops[
            stops["노선ID"].astype(str)
            == selected_route
        ].sort_values("정류장순번")

        points = [
            [float(x), float(y)]
            for x, y in zip(
                route["위도"],
                route["경도"],
            )
        ]

        route_map = folium.Map(
            location=[
                route["위도"].mean(),
                route["경도"].mean(),
            ],
            zoom_start=12,
            tiles="OpenStreetMap",
        )

        for _, row in route.iterrows():
            folium.CircleMarker(
                location=[
                    float(row["위도"]),
                    float(row["경도"]),
                ],
                radius=5,
                color="#dc2626",
                fill=True,
                fill_color="#dc2626",
                fill_opacity=0.8,
                tooltip=(
                    f"{int(row['정류장순번'])}. "
                    f"{row['정류장명']}"
                ),
            ).add_to(route_map)

        if len(points) >= 2:
            folium.PolyLine(
                locations=points,
                color="#dc2626",
                weight=6,
                opacity=0.85,
                tooltip=f"{selected_route} 노선",
            ).add_to(route_map)

        st_folium(
            route_map,
            height=550,
            use_container_width=True,
        )

    else:

        st.info(
            "현재 연결된 청주시 BIS 파일은 "
            "정류장 위치 데이터입니다. "
            "이 파일에는 노선별 경유 순서가 없으므로 "
            "정류장 전체 위치는 표시할 수 있지만 "
            "실제 노선 선형은 별도의 '노선-정류장 순번' "
            "데이터를 추가해야 합니다."
        )

        st.markdown(
            """
**다음 단계에서 추가할 데이터**

`노선ID + 노선번호 + 정류장ID + 정류장순번`

이 데이터를 연결하면

**실제 정류장 → 정류장 순서 → 실제 노선 경로**

를 지도에서 표현할 수 있습니다.
"""
        )


# ============================================================
# ⑥ 노선 대안
# ============================================================

elif page == "🛣️ 노선 대안":

    st.header("🛣️ 노선 대안 시뮬레이션")

    scenario = st.selectbox(
        "정책 대안",
        [
            "현행 유지",
            "대안 A · 거점 직결형",
            "대안 B · 간선 연장형",
            "대안 C · 환승 최적화형",
        ],
    )

    descriptions = {
        "현행 유지":
            "기존 노선을 유지합니다.",
        "대안 A · 거점 직결형":
            "산업·주거·환승거점을 직접 연결합니다.",
        "대안 B · 간선 연장형":
            "기존 간선노선을 미래 개발지역까지 연장합니다.",
        "대안 C · 환승 최적화형":
            "환승 결절점 중심으로 배차와 환승을 최적화합니다.",
    }

    st.info(
        descriptions[scenario]
    )

    frequency = st.slider(
        "배차간격 (분)",
        5,
        60,
        15,
    )

    congestion = st.slider(
        "교통 혼잡 영향 (%)",
        0,
        100,
        15,
    )

    transfer = st.slider(
        "환승시간 (분)",
        0,
        20,
        5,
    )

    average_wait = (
        frequency / 2
    ) * (
        1 + congestion / 100
    )

    travel_time = (
        30
        * (1 + congestion / 100)
    )

    a, b, c = st.columns(3)

    a.metric(
        "예상 평균 대기",
        f"{average_wait:.1f}분",
    )

    b.metric(
        "예상 통행시간",
        f"{travel_time:.1f}분",
    )

    c.metric(
        "환승시간",
        f"{transfer:.1f}분",
    )

    st.markdown("### 🗺️ 선택 노선")

    route_ids = (
        stops["노선ID"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    selected_route = st.selectbox(
        "기준 노선",
        route_ids,
    )

    route = stops[
        stops["노선ID"].astype(str)
        == selected_route
    ].sort_values(
        "정류장순번"
    )

    m = folium.Map(
        location=[
            route["위도"].mean(),
            route["경도"].mean(),
        ],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    points = []

    for _, stop in route.iterrows():

        point = [
            float(stop["위도"]),
            float(stop["경도"]),
        ]

        points.append(point)

        folium.Marker(
            location=point,
            tooltip=(
                f"{int(stop['정류장순번'])}. "
                f"{stop['정류장명']}"
            ),
            icon=folium.Icon(
                color="blue",
                icon="bus",
                prefix="fa",
            ),
        ).add_to(m)

    if len(points) >= 2:
        folium.PolyLine(
            locations=points,
            color="#dc2626",
            weight=7,
            opacity=0.85,
        ).add_to(m)

    st_folium(
        m,
        height=550,
        use_container_width=True,
    )


# ============================================================
# ⑦ 대안 비교
# ============================================================

elif page == "📊 대안 비교":

    st.header("📊 정책 대안 종합 비교")

    base_demand = st.number_input(
        "기준 일평균 총 승하차",
        0,
        1000000,
        100000,
        1000,
    )

    growth = st.slider(
        "미래 수요 증가율 (%)",
        -50,
        150,
        int(demand_growth),
    )

    future = round(
        base_demand
        * (1 + growth / 100)
    )

    scenarios = pd.DataFrame({
        "정책 대안": [
            "현행 유지",
            "대안 A · 거점 직결형",
            "대안 B · 간선 연장형",
            "대안 C · 환승 최적화형",
        ],
        "예상 수요": [
            future,
            future * 1.18,
            future * 1.10,
            future * 1.15,
        ],
        "통행시간": [
            45, 31, 36, 34
        ],
        "대기시간": [
            10, 7, 8, 6
        ],
        "운영비": [
            0.0, 4.5, 2.8, 3.6
        ],
        "탄소지수": [
            1.00, 0.78, 0.86, 0.72
        ],
    })

    feedback_count = len(
        st.session_state.feedback
    )

    if feedback_count > 0:
        satisfaction = np.mean([
            x["만족도"]
            for x in st.session_state.feedback
        ])

        feedback_bonus = (
            satisfaction / 5
        ) * 0.10

        st.info(
            f"시민 피드백 {feedback_count}건을 "
            f"반영 중 · 평균 만족도 "
            f"{satisfaction:.1f}/5"
        )
    else:
        feedback_bonus = 0

    demand_score = (
        scenarios["예상 수요"]
        / scenarios["예상 수요"].max()
    )

    time_score = (
        1
        - (
            scenarios["통행시간"]
            + scenarios["대기시간"]
        )
        / (
            scenarios["통행시간"]
            + scenarios["대기시간"]
        ).max()
    )

    cost_score = (
        1
        - scenarios["운영비"]
        / max(
            scenarios["운영비"].max(),
            1,
        )
    )

    carbon_score = (
        1
        - scenarios["탄소지수"]
        / scenarios["탄소지수"].max()
    )

    scenarios["종합점수"] = (
        demand_score * 0.30
        + time_score * 0.30
        + cost_score * 0.20
        + carbon_score * 0.20
        + feedback_bonus
    )

    best = scenarios.loc[
        scenarios["종합점수"].idxmax(),
        "정책 대안",
    ]

    def highlight(row):
        if row["정책 대안"] == best:
            return [
                "background-color: #DCFCE7"
            ] * len(row)
        return [""] * len(row)

    st.dataframe(
        scenarios.style.apply(
            highlight,
            axis=1,
        ).format({
            "예상 수요": "{:,.0f}",
            "통행시간": "{:.1f}",
            "대기시간": "{:.1f}",
            "운영비": "{:.1f}",
            "탄소지수": "{:.2f}",
            "종합점수": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        f"🤖 현재 조건의 추천안: **{best}**"
    )


# ============================================================
# ⑧ 시민 피드백
# ============================================================

elif page == "📣 시민 피드백":

    st.header("📣 정책 시행 후 시민 피드백")

    st.write(
        "정책 시행 이후 시민이 체감한 "
        "대기시간·환승·접근성·혼잡도 등을 입력하고 "
        "다음 정책 분석에 반영하는 단계입니다."
    )

    with st.form("feedback_form"):

        selected_policy = st.selectbox(
            "이용한 정책 대안",
            [
                "현행 유지",
                "대안 A · 거점 직결형",
                "대안 B · 간선 연장형",
                "대안 C · 환승 최적화형",
            ],
        )

        satisfaction = st.slider(
            "전체 만족도",
            1,
            5,
            3,
        )

        waiting = st.slider(
            "대기시간 만족도",
            1,
            5,
            3,
        )

        transfer = st.slider(
            "환승 편의성",
            1,
            5,
            3,
        )

        accessibility = st.slider(
            "정류장·노선 접근성",
            1,
            5,
            3,
        )

        crowding = st.slider(
            "혼잡도 만족도",
            1,
            5,
            3,
        )

        issue = st.selectbox(
            "가장 개선이 필요한 부분",
            [
                "대기시간",
                "환승",
                "노선 접근성",
                "배차간격",
                "혼잡도",
                "특별한 문제 없음",
            ],
        )

        submitted = st.form_submit_button(
            "시민 의견 등록",
            use_container_width=True,
        )

    if submitted:

        st.session_state.feedback.append({
            "정책": selected_policy,
            "만족도": satisfaction,
            "대기": waiting,
            "환승": transfer,
            "접근성": accessibility,
            "혼잡": crowding,
            "개선요구": issue,
        })

        st.success(
            "시민 의견이 등록되었습니다. "
            "다음 정책 평가에 반영됩니다."
        )

    feedback = st.session_state.feedback

    if feedback:

        st.markdown("---")
        st.subheader("📊 누적 시민 피드백")

        average_satisfaction = np.mean([
            x["만족도"]
            for x in feedback
        ])

        average_waiting = np.mean([
            x["대기"]
            for x in feedback
        ])

        average_transfer = np.mean([
            x["환승"]
            for x in feedback
        ])

        average_accessibility = np.mean([
            x["접근성"]
            for x in feedback
        ])

        a, b, c, d = st.columns(4)

        a.metric(
            "응답 수",
            f"{len(feedback):,}건",
        )

        b.metric(
            "평균 만족도",
            f"{average_satisfaction:.1f}/5",
        )

        c.metric(
            "환승 만족도",
            f"{average_transfer:.1f}/5",
        )

        d.metric(
            "접근성",
            f"{average_accessibility:.1f}/5",
        )

        issues = pd.Series([
            x["개선요구"]
            for x in feedback
        ])

        issue_counts = (
            issues.value_counts()
            .rename_axis("개선 요구")
            .reset_index(name="응답 수")
        )

        st.markdown("### 🔎 시민이 가장 많이 요구한 개선사항")

        st.dataframe(
            issue_counts,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 🔄 AI 정책 개선 반영")

        if average_satisfaction < 3:
            st.warning(
                "시민 만족도가 낮습니다. "
                "노선·배차·환승체계의 재검토가 필요합니다."
            )
        elif average_satisfaction < 4:
            st.info(
                "시민 만족도가 보통 수준입니다. "
                "주요 불편사항을 중심으로 추가 개선이 필요합니다."
            )
        else:
            st.success(
                "시민 만족도가 높습니다. "
                "현재 정책의 유지·확대를 검토할 수 있습니다."
            )

        st.markdown(
            """
**정책 시행**
→ **시민 이용**
→ **피드백 수집**
→ **불편요인 분석**
→ **정책 대안 재평가**
→ **다음 노선·배차 계획**
"""
        )

    else:

        st.info(
            "아직 시민 피드백이 없습니다. "
            "위 설문을 입력하면 정책 개선 과정이 표시됩니다."
        )


# ============================================================
# ⑨ 행정 의사결정
# ============================================================

elif page == "🏛️ 행정 의사결정":

    st.header("🏛️ 행정기관 의사결정")

    feedback_count = len(
        st.session_state.feedback
    )

    st.markdown(
        """
### 의사결정 구조

**AI 분석**
→ 미래 수요 예측
→ 노선 대안 비교
→ 운영효과 분석
→ 시민 피드백 분석
→ **행정기관 최종 판단**
"""
    )

    if feedback_count:
        satisfaction = np.mean([
            x["만족도"]
            for x in st.session_state.feedback
        ])

        st.metric(
            "누적 시민 만족도",
            f"{satisfaction:.1f}/5",
        )

    decision = st.radio(
        "최종 정책 상태",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 분석",
            "시민 의견 반영 후 재검토",
        ],
    )

    if decision == "정책 대안 채택":

        st.success(
            "정책 대안을 채택합니다. "
            "시행 후 운영 데이터와 시민 피드백을 "
            "다음 정책에 반영합니다."
        )

    elif decision == "시민 의견 반영 후 재검토":

        st.warning(
            "시민 피드백을 반영하여 "
            "노선·배차·환승 대안을 다시 검토합니다."
        )

    elif decision == "추가 분석":

        st.info(
            "추가적인 교통·도시계획 데이터와 "
            "전문가 검토가 필요합니다."
        )

    else:

        st.info(
            "AI 분석 결과를 참고하여 "
            "행정기관이 최종 판단합니다."
        )

    st.markdown("---")

    st.subheader("🔄 지속적인 정책 선순환")

    st.markdown(
        """
**도시 변화**
↓  
**미래 수요 예측**
↓  
**대중교통 계획**
↓  
**노선·배차 시행**
↓  
**시민 이용**
↓  
**시민 피드백**
↓  
**AI 분석**
↓  
**정책 재설계**
"""
    )

    st.success(
        "목표: 한 번의 노선 개편이 아니라 "
        "도시 변화와 시민 반응을 지속적으로 반영하는 "
        "데이터 기반 대중교통 행정체계"
    )


# ------------------------------------------------------------
# 하단 안내
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "※ 현재 프로토타입은 PoC용 시뮬레이션입니다. "
    "실증 단계에서는 실제 청주시 정류장·노선·BIS·도시계획 "
    "데이터와 검증된 교통수요예측 모델을 연계하는 구조를 전제로 합니다."
)
