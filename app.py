import math
from io import BytesIO

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium


# ============================================================
# 청주시 미래 대중교통 계획 의사결정 지원 플랫폼
# 독립 실행형 PoC
#
# 핵심 흐름
# 읍면동 인구 입력
#      ↓
# 미래인구 산정
#      ↓
# 정류장 데이터 업로드/샘플 사용
#      ↓
# 정류장 공급·접근성 분석
#      ↓
# 미래 교통수요 추정
#      ↓
# 우선 개선지역 선정
#      ↓
# 노선 후보축 생성
#      ↓
# 정책 시나리오 비교
#
# 주의:
# - 이 프로그램은 공모전용 PoC입니다.
# - "AI가 실제 정책을 결정한다"고 주장하지 않습니다.
# - 수요·우선순위 계산식은 투명하게 공개합니다.
# ============================================================


# ------------------------------------------------------------
# 0. 페이지 설정
# ------------------------------------------------------------

st.set_page_config(
    page_title="청주시 미래 대중교통 계획",
    page_icon="🚌",
    layout="wide",
)

st.title("🚌 청주시 미래 대중교통 계획 의사결정 지원 플랫폼")
st.caption("독립형 공모전 PoC · 인구·정류장 기반 대중교통 계획 시뮬레이터")

st.info(
    "읍면동 인구를 직접 입력하거나 CSV로 불러오고, "
    "버스정류장 위치를 업로드하면 미래인구·수요·접근성·"
    "노선개편 우선지역을 자동 분석합니다."
)


# ------------------------------------------------------------
# 1. 공통 함수
# ------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """두 좌표 사이의 직선거리를 km로 계산"""
    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * r * math.atan2(
        math.sqrt(a),
        math.sqrt(max(0, 1 - a)),
    )


def nearest_distance(area_lat, area_lon, stops):
    """읍면동 중심점에서 가장 가까운 정류장 거리"""
    if stops.empty:
        return None

    distances = [
        haversine_km(
            area_lat,
            area_lon,
            row["위도"],
            row["경도"],
        )
        for _, row in stops.iterrows()
    ]

    return min(distances)


def stops_within_radius(area_lat, area_lon, stops, radius):
    """분석반경 안의 정류장 개수"""
    if stops.empty:
        return 0

    count = 0

    for _, row in stops.iterrows():
        d = haversine_km(
            area_lat,
            area_lon,
            row["위도"],
            row["경도"],
        )

        if d <= radius:
            count += 1

    return count


def clean_population(df):
    """인구 데이터 형식 정리"""

    df = df.copy()

    required = [
        "읍면동",
        "현재인구(명)",
        "연평균증가율(%)",
        "위도",
        "경도",
    ]

    for col in required:
        if col not in df.columns:
            df[col] = 0

    df["읍면동"] = (
        df["읍면동"]
        .astype(str)
        .str.strip()
    )

    for col in required[1:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    df["현재인구(명)"] = (
        df["현재인구(명)"]
        .clip(lower=0)
    )

    df["연평균증가율(%)"] = (
        df["연평균증가율(%)"]
        .clip(-100, 100)
    )

    df["위도"] = pd.to_numeric(
        df["위도"],
        errors="coerce",
    ).fillna(0)

    df["경도"] = pd.to_numeric(
        df["경도"],
        errors="coerce",
    ).fillna(0)

    return df[
        required
    ].reset_index(drop=True)


def clean_stops(df):
    """정류장 데이터 컬럼을 표준화"""

    df = df.copy()

    # 가능한 한 다양한 CSV 컬럼명을 자동 인식
    name_candidates = [
        "정류소명",
        "정류장명",
        "정류장",
        "stop_name",
        "name",
    ]

    lat_candidates = [
        "위도",
        "lat",
        "latitude",
        "좌표(Y)",
        "Y",
    ]

    lon_candidates = [
        "경도",
        "lon",
        "longitude",
        "좌표(X)",
        "X",
    ]

    def find_column(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    name_col = find_column(name_candidates)
    lat_col = find_column(lat_candidates)
    lon_col = find_column(lon_candidates)

    if lat_col is None or lon_col is None:
        return None, (
            "정류장 CSV에서 위도·경도 컬럼을 찾지 못했습니다. "
            "예: 위도/경도 또는 좌표(Y)/좌표(X)"
        )

    if name_col is None:
        df["정류장명"] = "정류장"
        name_col = "정류장명"

    result = pd.DataFrame(
        {
            "정류장명": df[name_col].astype(str),
            "위도": pd.to_numeric(
                df[lat_col],
                errors="coerce",
            ),
            "경도": pd.to_numeric(
                df[lon_col],
                errors="coerce",
            ),
        }
    )

    result = result.dropna(
        subset=["위도", "경도"]
    )

    # 대한민국 범위를 벗어난 이상값 제거
    result = result[
        result["위도"].between(33, 39)
        & result["경도"].between(124, 132)
    ]

    result = (
        result
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return result, None


def default_population():
    """앱 첫 실행용 예시 데이터"""

    return pd.DataFrame(
        [
            ["내덕1동", 7659, 2.0, 36.6530, 127.4910],
            ["내덕2동", 10000, 1.5, 36.6550, 127.4960],
            ["우암동", 15000, 1.0, 36.6500, 127.4860],
            ["오창읍", 68000, 2.5, 36.7153, 127.4258],
            ["오송읍", 48000, 2.5, 36.6205, 127.3274],
            ["가경동", 53000, 0.8, 36.6240, 127.3900],
            ["복대동", 53000, 0.8, 36.6355, 127.4221],
            ["성안동", 15000, -0.5, 36.6338, 127.4879],
        ],
        columns=[
            "읍면동",
            "현재인구(명)",
            "연평균증가율(%)",
            "위도",
            "경도",
        ],
    )


def default_stops():
    """정류장 파일이 없을 때 사용하는 소규모 예시"""

    return pd.DataFrame(
        [
            ["내덕중학교", 36.6537, 127.4912],
            ["내덕시장", 36.6548, 127.4935],
            ["우암동주민센터", 36.6501, 127.4868],
            ["청주대학교", 36.6518, 127.4970],
            ["오창과학산업단지", 36.7160, 127.4262],
            ["오창호수공원", 36.7100, 127.4310],
            ["오송역", 36.6204, 127.3276],
            ["가경터미널", 36.6242, 127.3904],
            ["복대시장", 36.6357, 127.4224],
            ["성안길", 36.6337, 127.4875],
        ],
        columns=[
            "정류장명",
            "위도",
            "경도",
        ],
    )


# ------------------------------------------------------------
# 2. 세션 데이터
# ------------------------------------------------------------

if "population" not in st.session_state:
    st.session_state.population = default_population()

if "stops" not in st.session_state:
    st.session_state.stops = default_stops()


# ------------------------------------------------------------
# 3. 사이드바
# ------------------------------------------------------------

st.sidebar.header("⚙️ 분석 설정")

page = st.sidebar.radio(
    "분석 메뉴",
    [
        "① 종합 대시보드",
        "② 인구 시나리오",
        "③ 정류장 분석",
        "④ 노선 개편 분석",
        "⑤ 정책 시나리오",
    ],
)

target_year = st.sidebar.selectbox(
    "분석 연도",
    [2027, 2030, 2035, 2040, 2045],
    index=2,
)

radius = st.sidebar.slider(
    "정류장 접근성 분석 반경(km)",
    0.2,
    2.0,
    0.8,
    0.1,
)

demand_rate = st.sidebar.slider(
    "인구 대비 일일 대중교통 수요율(%)",
    1.0,
    30.0,
    10.0,
    0.5,
)


# ------------------------------------------------------------
# 4. 인구 데이터 입력
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("👥 읍면동 인구")

population_file = st.sidebar.file_uploader(
    "인구 CSV 업로드",
    type=["csv"],
    key="population_csv",
)

if population_file is not None:

    try:
        uploaded_population = pd.read_csv(
            population_file,
            encoding="utf-8-sig",
        )

        st.session_state.population = clean_population(
            uploaded_population
        )

        st.sidebar.success(
            "인구 CSV 적용 완료"
        )

    except UnicodeDecodeError:

        try:
            uploaded_population = pd.read_csv(
                population_file,
                encoding="cp949",
            )

            st.session_state.population = clean_population(
                uploaded_population
            )

            st.sidebar.success(
                "인구 CSV 적용 완료"
            )

        except Exception as e:
            st.sidebar.error(
                f"인구 CSV 오류: {e}"
            )

    except Exception as e:

        st.sidebar.error(
            f"인구 CSV 오류: {e}"
        )


# ------------------------------------------------------------
# 5. 정류장 데이터 입력
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("🚌 정류장 데이터")

stop_file = st.sidebar.file_uploader(
    "정류장 CSV 업로드",
    type=["csv"],
    key="stop_csv",
)

if stop_file is not None:

    try:
        uploaded_stops = pd.read_csv(
            stop_file,
            encoding="utf-8-sig",
        )

        parsed_stops, error = clean_stops(
            uploaded_stops
        )

        if error:
            st.sidebar.error(error)
        else:
            st.session_state.stops = parsed_stops

            st.sidebar.success(
                f"정류장 {len(parsed_stops):,}개 적용"
            )

    except UnicodeDecodeError:

        try:
            uploaded_stops = pd.read_csv(
                stop_file,
                encoding="cp949",
            )

            parsed_stops, error = clean_stops(
                uploaded_stops
            )

            if error:
                st.sidebar.error(error)
            else:
                st.session_state.stops = parsed_stops

        except Exception as e:
            st.sidebar.error(
                f"정류장 CSV 오류: {e}"
            )

    except Exception as e:

        st.sidebar.error(
            f"정류장 CSV 오류: {e}"
        )


population = clean_population(
    st.session_state.population
)

stops = st.session_state.stops.copy()


# ------------------------------------------------------------
# 6. 미래인구 계산
# ------------------------------------------------------------

years = target_year - 2025

population["미래인구(명)"] = (
    population["현재인구(명)"]
    * (
        1
        + population["연평균증가율(%)"] / 100
    ) ** years
).round().astype(int)


# ------------------------------------------------------------
# 7. 읍면동별 정류장 분석
# ------------------------------------------------------------

analysis_rows = []

for _, area in population.iterrows():

    name = area["읍면동"]
    current_pop = area["현재인구(명)"]
    future_pop = area["미래인구(명)"]
    lat = area["위도"]
    lon = area["경도"]

    if lat == 0 or lon == 0:
        nearest = None
        stop_count = 0
    else:
        nearest = nearest_distance(
            lat,
            lon,
            stops,
        )

        stop_count = stops_within_radius(
            lat,
            lon,
            stops,
            radius,
        )

    if nearest is None:
        nearest = 999

    # 정류장 공급량
    supply_per_1000 = (
        stop_count
        / max(future_pop / 1000, 1)
    )

    # 접근성 점수
    distance_score = max(
        0,
        min(
            100,
            (1 - nearest / max(radius, 0.1))
            * 100,
        ),
    )

    supply_score = min(
        100,
        supply_per_1000 * 100,
    )

    accessibility = (
        distance_score * 0.6
        + supply_score * 0.4
    )

    # 미래 교통수요
    demand = (
        future_pop
        * demand_rate
        / 100
    )

    # 수요가 많고 접근성이 낮을수록
    # 노선개편 필요도가 높도록 계산
    demand_pressure = min(
        100,
        demand / 1000,
    )

    improvement_priority = (
        demand_pressure * 0.55
        + (100 - accessibility) * 0.45
    )

    if stop_count == 0:
        status = "🚨 정류장 공급 없음"
    elif accessibility < 40:
        status = "🔴 접근성 취약"
    elif improvement_priority >= 60:
        status = "🟠 노선 개선 우선"
    else:
        status = "🟢 상대적 양호"

    analysis_rows.append(
        [
            name,
            current_pop,
            future_pop,
            stop_count,
            nearest,
            supply_per_1000,
            accessibility,
            demand,
            improvement_priority,
            status,
        ]
    )


analysis = pd.DataFrame(
    analysis_rows,
    columns=[
        "읍면동",
        "현재인구(명)",
        "미래인구(명)",
        "반경내 정류장",
        "최근접 정류장(km)",
        "정류장/1천명",
        "접근성 점수",
        "예상 일일 수요(건)",
        "노선개편 우선도",
        "판정",
    ],
)


# ------------------------------------------------------------
# 8. 지도
# ------------------------------------------------------------

def create_map():

    if stops.empty:
        center = [36.635, 127.49]
    else:
        center = [
            stops["위도"].mean(),
            stops["경도"].mean(),
        ]

    m = folium.Map(
        location=center,
        zoom_start=11,
        tiles="OpenStreetMap",
    )

    # 정류장
    for _, stop in stops.iterrows():

        folium.CircleMarker(
            location=[
                stop["위도"],
                stop["경도"],
            ],
            radius=3,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.55,
            tooltip=stop["정류장명"],
        ).add_to(m)

    # 읍면동
    for _, row in analysis.iterrows():

        area = population[
            population["읍면동"]
            == row["읍면동"]
        ]

        if area.empty:
            continue

        area = area.iloc[0]

        lat = area["위도"]
        lon = area["경도"]

        if lat == 0 or lon == 0:
            continue

        priority = row["노선개편 우선도"]

        if priority >= 70:
            color = "#dc2626"
        elif priority >= 50:
            color = "#f59e0b"
        else:
            color = "#16a34a"

        popup_html = f"""
        <b>{row['읍면동']}</b><br>
        현재인구: {row['현재인구(명)']:,}명<br>
        미래인구: {row['미래인구(명)']:,}명<br>
        반경내 정류장: {row['반경내 정류장']}개<br>
        최근접 정류장: {row['최근접 정류장(km)']:.2f}km<br>
        예상수요: {row['예상 일일 수요(건)']:,}건/일<br>
        접근성: {row['접근성 점수']:.1f}점<br>
        노선개편 우선도: {priority:.1f}점
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(
                7,
                min(
                    24,
                    7 + priority / 5,
                ),
            ),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=folium.Popup(
                popup_html,
                max_width=320,
            ),
            tooltip=(
                f"{row['읍면동']} "
                f"| 우선도 {priority:.1f}"
            ),
        ).add_to(m)

    # 우선순위 상위 지역을 후보축으로 연결
    top = analysis.sort_values(
        "노선개편 우선도",
        ascending=False,
    ).head(5)

    route_points = []

    for _, row in top.iterrows():

        area = population[
            population["읍면동"]
            == row["읍면동"]
        ]

        if area.empty:
            continue

        area = area.iloc[0]

        if (
            area["위도"] != 0
            and area["경도"] != 0
        ):
            route_points.append(
                [
                    area["위도"],
                    area["경도"],
                ]
            )

    if len(route_points) >= 2:

        folium.PolyLine(
            route_points,
            color="#dc2626",
            weight=6,
            opacity=0.8,
            dash_array="10,8",
            tooltip="데이터 기반 노선 개편 후보축",
        ).add_to(m)

    return m


# ------------------------------------------------------------
# 9. 페이지 ① 종합 대시보드
# ------------------------------------------------------------

if page == "① 종합 대시보드":

    st.header("① 종합 대시보드")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "분석 정류장",
        f"{len(stops):,}개",
    )

    c2.metric(
        "분석 읍면동",
        f"{len(population):,}개",
    )

    c3.metric(
        f"{target_year}년 미래인구",
        f"{population['미래인구(명)'].sum():,.0f}명",
    )

    c4.metric(
        "예상 일일 수요",
        f"{analysis['예상 일일 수요(건)'].sum():,.0f}건",
    )

    st.markdown("---")

    st.subheader(
        "🚨 노선 개편 우선지역 TOP 5"
    )

    top5 = analysis.sort_values(
        "노선개편 우선도",
        ascending=False,
    ).head(5)

    st.dataframe(
        top5[
            [
                "읍면동",
                "현재인구(명)",
                "미래인구(명)",
                "반경내 정류장",
                "최근접 정류장(km)",
                "예상 일일 수요(건)",
                "접근성 점수",
                "노선개편 우선도",
                "판정",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader(
        "🗺️ 미래 대중교통 계획 지도"
    )

    m = create_map()

    st_folium(
        m,
        height=650,
        use_container_width=True,
    )

    st.caption(
        "🔵 실제 정류장 / "
        "🔴 높은 개선 우선도 / "
        "🟠 개선 필요 / "
        "🟢 상대적으로 양호 / "
        "빨간 점선 = 데이터 기반 노선 후보축"
    )


# ------------------------------------------------------------
# 10. 페이지 ② 인구 시나리오
# ------------------------------------------------------------

elif page == "② 인구 시나리오":

    st.header(
        "② 읍면동 미래인구 시나리오"
    )

    st.write(
        "읍면동별 현재인구와 연평균 증가율을 직접 입력하면 "
        "선택한 연도의 미래인구를 자동 계산합니다."
    )

    edited = st.data_editor(
        population[
            [
                "읍면동",
                "현재인구(명)",
                "연평균증가율(%)",
                "위도",
                "경도",
            ]
        ],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="population_editor",
    )

    if st.button(
        "💾 인구 시나리오 적용",
        use_container_width=True,
    ):

        st.session_state.population = (
            clean_population(edited)
        )

        st.success(
            "인구 시나리오를 적용했습니다."
        )

        st.rerun()

    st.markdown("---")

    st.subheader(
        f"📈 {target_year}년 예측 결과"
    )

    st.dataframe(
        population[
            [
                "읍면동",
                "현재인구(명)",
                "연평균증가율(%)",
                "미래인구(명)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        population.set_index("읍면동")[
            [
                "현재인구(명)",
                "미래인구(명)",
            ]
        ]
    )


# ------------------------------------------------------------
# 11. 페이지 ③ 정류장 분석
# ------------------------------------------------------------

elif page == "③ 정류장 분석":

    st.header(
        "③ 정류장 공급·접근성 분석"
    )

    st.write(
        f"각 읍면동 중심점에서 반경 **{radius:.1f}km** "
        "안에 있는 정류장을 분석합니다."
    )

    st.metric(
        "현재 사용 중인 정류장 데이터",
        f"{len(stops):,}개",
    )

    st.dataframe(
        analysis[
            [
                "읍면동",
                "미래인구(명)",
                "반경내 정류장",
                "정류장/1천명",
                "최근접 정류장(km)",
                "접근성 점수",
                "판정",
            ]
        ].sort_values(
            "접근성 점수"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader(
        "🗺️ 정류장·읍면동 공간분석"
    )

    st_folium(
        create_map(),
        height=680,
        use_container_width=True,
    )


# ------------------------------------------------------------
# 12. 페이지 ④ 노선 개편 분석
# ------------------------------------------------------------

elif page == "④ 노선 개편 분석":

    st.header(
        "④ 데이터 기반 노선 개편 분석"
    )

    st.success(
        "핵심 판단 기준: "
        "미래수요가 많고 + 정류장 접근성이 낮은 지역을 "
        "노선 개편 우선지역으로 선정합니다."
    )

    ranked = analysis.sort_values(
        "노선개편 우선도",
        ascending=False,
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "우선순위",
        range(1, len(ranked) + 1),
    )

    st.dataframe(
        ranked[
            [
                "우선순위",
                "읍면동",
                "미래인구(명)",
                "반경내 정류장",
                "최근접 정류장(km)",
                "예상 일일 수요(건)",
                "접근성 점수",
                "노선개편 우선도",
                "판정",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader(
        "🚌 자동 노선 후보축"
    )

    top = ranked.head(5)

    st.markdown(
        " → ".join(
            top["읍면동"].tolist()
        )
    )

    st_folium(
        create_map(),
        height=700,
        use_container_width=True,
    )

    st.warning(
        "주의: 빨간 점선은 실제 운행노선이 아니라 "
        "우선 개선지역을 연결한 '노선 개편 후보축'입니다. "
        "실제 노선 확정에는 도로망·교통량·운행시간·"
        "기존 노선·환승·운영비 등을 추가 검증해야 합니다."
    )


# ------------------------------------------------------------
# 13. 페이지 ⑤ 정책 시나리오
# ------------------------------------------------------------

elif page == "⑤ 정책 시나리오":

    st.header(
        "⑤ 대중교통 정책 시나리오 비교"
    )

    top = analysis.sort_values(
        "노선개편 우선도",
        ascending=False,
    ).head(5)

    total_demand = (
        analysis["예상 일일 수요(건)"].sum()
    )

    current_access = (
        analysis["접근성 점수"].mean()
    )

    scenarios = pd.DataFrame(
        [
            [
                "현행 유지",
                total_demand,
                current_access,
                0,
                "현재 체계를 유지",
            ],
            [
                "대안 A · 우선지역 직결",
                total_demand * 1.12,
                min(
                    100,
                    current_access + 10,
                ),
                4.5,
                "상위 우선지역을 직접 연결",
            ],
            [
                "대안 B · 접근성 개선",
                total_demand * 1.08,
                min(
                    100,
                    current_access + 15,
                ),
                3.2,
                "취약지역 정류장·노선을 보완",
            ],
            [
                "대안 C · 환승 중심 개편",
                total_demand * 1.15,
                min(
                    100,
                    current_access + 8,
                ),
                3.8,
                "주요 환승거점 중심으로 체계 개편",
            ],
        ],
        columns=[
            "정책 대안",
            "예상 일일 수요",
            "접근성 점수",
            "추가 운영비(억원)",
            "설명",
        ],
    )

    # 투명한 PoC 평가점수
    scenarios["정책점수"] = (
        scenarios["예상 일일 수요"]
        / scenarios["예상 일일 수요"].max()
        * 50
        + scenarios["접근성 점수"]
        / 100
        * 35
        + (
            1
            - scenarios["추가 운영비(억원)"]
            / max(
                scenarios["추가 운영비(억원)"].max(),
                1,
            )
        )
        * 15
    ).round(1)

    best = scenarios.loc[
        scenarios["정책점수"].idxmax()
    ]

    st.dataframe(
        scenarios,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader(
        "🏆 PoC 추천 정책"
    )

    st.success(
        f"**{best['정책 대안']}** · "
        f"종합 정책점수 {best['정책점수']:.1f}점"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "예상 일일 수요",
        f"{best['예상 일일 수요']:,.0f}건",
    )

    col2.metric(
        "접근성 점수",
        f"{best['접근성 점수']:.1f}점",
    )

    col3.metric(
        "추가 운영비",
        f"{best['추가 운영비(억원)']:.1f}억원",
    )

    st.info(
        "이 추천은 공모전용 PoC의 비교 결과이며 "
        "실제 행정 의사결정을 자동으로 대신하지 않습니다."
    )


# ------------------------------------------------------------
# 14. 데이터 다운로드
# ------------------------------------------------------------

st.markdown("---")

csv_data = analysis.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "📥 분석 결과 CSV 다운로드",
    data=csv_data,
    file_name="청주시_대중교통_분석결과.csv",
    mime="text/csv",
)

with st.expander("📌 데이터·모델의 한계와 실제 적용 방향"):

    st.write(
        """
### 현재 PoC

- 읍면동 인구는 직접 입력하거나 CSV로 변경할 수 있습니다.
- 정류장 위치는 CSV로 업로드할 수 있습니다.
- 미래인구는 현재인구와 연평균 증가율을 이용해 계산합니다.
- 미래 교통수요는 미래인구 × 설정된 수요율로 추정합니다.
- 접근성은 정류장 수와 최근접 정류장 거리를 이용합니다.
- 노선개편 우선도는 미래수요와 접근성을 결합하여 계산합니다.
- 빨간 점선은 실제 버스노선이 아니라 계획 후보축입니다.

### 실제 실증 단계

향후에는 교통카드 승하차·환승 데이터, BIS 운행정보,
도로망과 통행속도, 기존 버스노선, 생활인구,
토지이용 및 도시개발사업 데이터를 연계하여
실제 교통수요예측과 네트워크 분석으로 고도화할 수 있습니다.
"""
    )
