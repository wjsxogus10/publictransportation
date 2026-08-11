import os
from pathlib import Path

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium


# ============================================================
# 0. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="청주시 대중교통 계획 플랫폼",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 1. TITLE
# ============================================================

st.title(
    "🚌 AI 기반 미래예측형 "
    "청주시 대중교통 의사결정 지원 플랫폼"
)

st.caption(
    "청주시 실제 버스정류장 데이터를 기반으로 한 "
    "공모전용 대중교통 계획 PoC"
)

st.info(
    "현재 플랫폼은 청주시 실제 정류장 위치 데이터를 기반으로 "
    "정류장 공간정보를 시각화합니다. "
    "향후 인구·수요·노선·시민피드백 데이터를 연결하여 "
    "미래 대중교통 계획으로 확장할 수 있습니다."
)


# ============================================================
# 2. FILE PATH
# ============================================================

STOP_FILE = Path(__file__).with_name(
    "충청북도_청주시_버스정보시스템_20250401.csv"
)


# ============================================================
# 3. LOAD BUS STOP DATA
# ============================================================

@st.cache_data
def load_bus_stops():

    if not STOP_FILE.exists():

        return None, (
            "청주시 정류장 CSV 파일을 찾을 수 없습니다.\n\n"
            "app.py와 같은 폴더에 다음 파일을 넣어주세요.\n\n"
            "충청북도_청주시_버스정보시스템_20250401.csv"
        )

    df = None
    last_error = None

    # 한글 CSV 인코딩 대응
    for encoding in [
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "utf-8",
    ]:

        try:

            df = pd.read_csv(
                STOP_FILE,
                encoding=encoding,
            )

            break

        except Exception as e:

            last_error = e

    if df is None:

        return None, (
            f"CSV 파일을 읽을 수 없습니다.\n\n"
            f"오류: {last_error}"
        )

    # 컬럼명 정리
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required_columns = [
        "서비스ID",
        "정류소명",
        "좌표(X)",
        "좌표(Y)",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        return None, (
            "필수 컬럼이 없습니다.\n\n"
            f"누락 컬럼: {', '.join(missing)}"
        )

    # 필요한 데이터만 사용
    df = df[
        required_columns
    ].copy()

    # 좌표 숫자 변환
    df["좌표(X)"] = pd.to_numeric(
        df["좌표(X)"],
        errors="coerce",
    )

    df["좌표(Y)"] = pd.to_numeric(
        df["좌표(Y)"],
        errors="coerce",
    )

    # 좌표 없는 데이터 제거
    df = df.dropna(
        subset=[
            "좌표(X)",
            "좌표(Y)",
        ]
    ).copy()

    # 대한민국 좌표 범위
    df = df[
        df["좌표(X)"].between(
            124,
            132,
        )
        &
        df["좌표(Y)"].between(
            33,
            39,
        )
    ].copy()

    # 서비스 ID 중복 제거
    df = df.drop_duplicates(
        subset=["서비스ID"]
    ).reset_index(drop=True)

    return df, None


stops, error = load_bus_stops()


# ============================================================
# 4. ERROR HANDLING
# ============================================================

if stops is None:

    st.error(error)

    st.markdown(
        """
        ### 📁 GitHub 폴더 구조

        ```text
        publictransportation/
        │
        ├── app.py
        ├── requirements.txt
        └── 충청북도_청주시_버스정보시스템_20250401.csv
        ```
        """
    )

    st.stop()


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title("🚌 대중교통 계획")

st.sidebar.success(
    f"실제 정류장 데이터 연결\n\n"
    f"{len(stops):,}개 정류장"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "페이지 선택",
    [
        "🏠 종합 대시보드",
        "🗺️ 정류장 지도",
        "📍 정류장 분석",
        "📊 데이터 분석",
        "🔮 미래 시나리오",
        "🚌 노선 계획",
        "📣 시민 피드백",
        "🏛️ 정책 의사결정",
    ],
)


# ============================================================
# 6. COMMON FUNCTIONS
# ============================================================

def make_map(
    data,
    zoom=12,
    height=650,
):

    if data.empty:

        return None

    center_lat = data[
        "좌표(Y)"
    ].mean()

    center_lon = data[
        "좌표(X)"
    ].mean()

    m = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    for _, row in data.iterrows():

        popup = f"""
        <div style="font-size:14px;">
            <b>{row['정류소명']}</b><br>
            서비스ID: {row['서비스ID']}<br>
            경도: {row['좌표(X)']:.6f}<br>
            위도: {row['좌표(Y)']:.6f}
        </div>
        """

        folium.CircleMarker(
            location=[
                row["좌표(Y)"],
                row["좌표(X)"],
            ],
            radius=4,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.65,
            weight=1,
            tooltip=row["정류소명"],
            popup=folium.Popup(
                popup,
                max_width=300,
            ),
        ).add_to(m)

    return m


# ============================================================
# 7. HOME
# ============================================================

if page == "🏠 종합 대시보드":

    st.header(
        "🏠 청주시 대중교통 계획 종합 대시보드"
    )

    st.markdown(
        """
        ### 플랫폼 핵심 구조

        **실제 청주시 정류장 데이터**

        ↓

        **인구·도시개발 변화**

        ↓

        **미래 교통수요 예측**

        ↓

        **정류장 접근성 분석**

        ↓

        **노선 개편 대안**

        ↓

        **시민 피드백**

        ↓

        **행정기관 정책 결정**
        """
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "전체 정류장",
        f"{len(stops):,}개",
    )

    c2.metric(
        "정류장 데이터",
        "청주시 BIS",
    )

    c3.metric(
        "공간정보",
        "X / Y 좌표",
    )

    c4.metric(
        "플랫폼 단계",
        "PoC",
    )

    st.markdown("---")

    st.subheader(
        "🗺️ 청주시 실제 정류장 분포"
    )

    m = make_map(
        stops,
        zoom=12,
        height=600,
    )

    if m:

        st_folium(
            m,
            height=600,
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader(
        "📌 현재 데이터의 역할"
    )

    st.write(
        """
        현재 연결된 데이터는 청주시 버스정류장의
        실제 위치를 나타내는 공간 데이터입니다.

        따라서 이 데이터를 대중교통 계획의
        기본 공간 DB로 활용하고,

        향후 인구·승하차·노선·도시개발 데이터를
        결합하여 정류장별 수요와 접근성을
        분석하는 구조로 확장합니다.
        """
    )


# ============================================================
# 8. BUS STOP MAP
# ============================================================

elif page == "🗺️ 정류장 지도":

    st.header(
        "🗺️ 청주시 전체 버스정류장 지도"
    )

    st.caption(
        "청주시 BIS 실제 정류장 위치 데이터"
    )

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        search = st.text_input(
            "🔎 정류장 검색",
            placeholder=(
                "정류장명 또는 서비스ID 입력"
            ),
        )

    with col2:

        show_count = st.checkbox(
            "결과 수 표시",
            value=True,
        )

    filtered = stops.copy()

    if search.strip():

        query = search.strip().lower()

        filtered = filtered[
            filtered[
                "정류소명"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False,
            )
            |
            filtered[
                "서비스ID"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False,
            )
        ].copy()

    if show_count:

        st.write(
            f"검색 결과: "
            f"**{len(filtered):,}개**"
        )

    if filtered.empty:

        st.warning(
            "검색 결과가 없습니다."
        )

    else:

        m = make_map(
            filtered,
            zoom=13 if search else 12,
            height=680,
        )

        st_folium(
            m,
            height=680,
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader(
        "📋 정류장 목록"
    )

    table = filtered[
        [
            "서비스ID",
            "정류소명",
            "좌표(X)",
            "좌표(Y)",
        ]
    ].copy()

    table = table.rename(
        columns={
            "서비스ID": "서비스 ID",
            "정류소명": "정류장명",
            "좌표(X)": "경도",
            "좌표(Y)": "위도",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 9. STOP ANALYSIS
# ============================================================

elif page == "📍 정류장 분석":

    st.header(
        "📍 정류장별 상세 분석"
    )

    search = st.text_input(
        "정류장 검색",
        placeholder="정류장명을 입력하세요",
    )

    candidate = stops.copy()

    if search:

        candidate = candidate[
            candidate["정류소명"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    if candidate.empty:

        st.warning(
            "검색된 정류장이 없습니다."
        )

    else:

        selected = st.selectbox(
            "분석할 정류장",
            candidate["정류소명"].tolist(),
        )

        row = candidate[
            candidate["정류소명"]
            == selected
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "정류장",
            row["정류소명"],
        )

        c2.metric(
            "경도",
            f"{row['좌표(X)']:.6f}",
        )

        c3.metric(
            "위도",
            f"{row['좌표(Y)']:.6f}",
        )

        st.markdown("---")

        st.subheader(
            "📌 정류장 위치"
        )

        one = pd.DataFrame(
            {
                "정류장": [
                    row["정류소명"]
                ],
                "서비스ID": [
                    row["서비스ID"]
                ],
                "경도": [
                    row["좌표(X)"]
                ],
                "위도": [
                    row["좌표(Y)"]
                ],
            }
        )

        st.dataframe(
            one,
            use_container_width=True,
            hide_index=True,
        )

        point_map = folium.Map(
            location=[
                row["좌표(Y)"],
                row["좌표(X)"],
            ],
            zoom_start=16,
            tiles="OpenStreetMap",
        )

        folium.Marker(
            location=[
                row["좌표(Y)"],
                row["좌표(X)"],
            ],
            tooltip=row["정류소명"],
            popup=folium.Popup(
                f"""
                <b>{row['정류소명']}</b><br>
                서비스ID: {row['서비스ID']}
                """,
                max_width=300,
            ),
        ).add_to(point_map)

        st_folium(
            point_map,
            height=500,
            use_container_width=True,
        )

        st.markdown("---")

        st.subheader(
            "🔮 향후 분석 가능 항목"
        )

        a, b, c = st.columns(3)

        a.info(
            "👥 주변 인구\n\n"
            "행정동·생활권 인구와 연결"
        )

        b.info(
            "🚌 미래 수요\n\n"
            "승하차 및 인구변화와 연결"
        )

        c.info(
            "🚶 접근성\n\n"
            "생활권별 정류장 접근성 분석"
        )


# ============================================================
# 10. DATA ANALYSIS
# ============================================================

elif page == "📊 데이터 분석":

    st.header(
        "📊 청주시 정류장 데이터 분석"
    )

    st.subheader(
        "기초 통계"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "정류장 수",
        f"{len(stops):,}",
    )

    c2.metric(
        "고유 정류장명",
        f"{stops['정류소명'].nunique():,}",
    )

    c3.metric(
        "최소 경도",
        f"{stops['좌표(X)'].min():.4f}",
    )

    c4.metric(
        "최대 경도",
        f"{stops['좌표(X)'].max():.4f}",
    )

    st.markdown("---")

    st.subheader(
        "정류장명 빈도"
    )

    name_count = (
        stops["정류소명"]
        .value_counts()
        .reset_index()
    )

    name_count.columns = [
        "정류장명",
        "개수",
    ]

    st.dataframe(
        name_count.head(50),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader(
        "원본 정류장 데이터"
    )

    st.dataframe(
        stops,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 11. FUTURE SCENARIO
# ============================================================

elif page == "🔮 미래 시나리오":

    st.header(
        "🔮 미래 도시·교통 시나리오"
    )

    st.caption(
        "정류장 데이터를 기반으로 향후 인구·수요 데이터를 "
        "연계하기 위한 정책 시뮬레이션 화면"
    )

    development = st.slider(
        "도시개발 영향",
        min_value=-20,
        max_value=50,
        value=15,
        step=5,
    )

    population_change = st.slider(
        "인구 변화",
        min_value=-30,
        max_value=50,
        value=10,
        step=5,
    )

    transit_change = st.slider(
        "대중교통 이용 변화",
        min_value=-30,
        max_value=50,
        value=10,
        step=5,
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "도시개발 영향",
        f"{development:+d}%",
    )

    c2.metric(
        "인구 변화",
        f"{population_change:+d}%",
    )

    c3.metric(
        "대중교통 수요 변화",
        f"{transit_change:+d}%",
    )

    st.markdown("---")

    st.subheader(
        "미래 대중교통 계획 방향"
    )

    if population_change > 10:

        st.success(
            "인구 증가 시나리오: "
            "주요 생활권 정류장 및 노선 공급 확대 검토"
        )

    elif population_change < 0:

        st.warning(
            "인구 감소 시나리오: "
            "노선 효율화 및 수요응답형 교통 검토"
        )

    else:

        st.info(
            "인구 안정 시나리오: "
            "기존 노선 효율성 개선 검토"
        )

    st.markdown("---")

    st.subheader(
        "📌 실제 적용 시"
    )

    st.write(
        """
        미래 인구 데이터를 정류장 공간정보와 결합하면
        정류장별 미래 수요를 산출하고,

        수요 증가지역은 노선·배차 확대,
        수요 감소지역은 노선 효율화,

        그리고 신규 개발지역은 신규 정류장 및
        노선 공급 후보지역으로 분석할 수 있습니다.
        """
    )


# ============================================================
# 12. ROUTE PLANNING
# ============================================================

elif page == "🚌 노선 계획":

    st.header(
        "🚌 미래 노선 계획"
    )

    st.warning(
        "현재 제공된 청주시 정류장 파일에는 "
        "정류장 위치정보만 있고 "
        "노선ID·정류장 순번 정보가 없습니다."
    )

    st.markdown(
        """
        ### 현재 데이터

        ```text
        서비스ID
        정류소명
        좌표(X)
        좌표(Y)
        ```

        ### 실제 노선 표현에 필요한 데이터

        ```text
        노선ID
        노선번호
        서비스ID
        정류장순번
        ```

        위 데이터를 연결하면 실제 노선의

        **정류장 A → 정류장 B → 정류장 C**

        순서를 지도에 연결할 수 있습니다.
        """
    )

    st.markdown("---")

    st.subheader(
        "🛣️ 노선 대안 개념"
    )

    option = st.radio(
        "정책 대안",
        [
            "현행 유지",
            "거점 직결형",
            "간선 연장형",
            "환승 최적화형",
        ],
        horizontal=True,
    )

    descriptions = {

        "현행 유지":
            "현재 대중교통 공급체계를 유지합니다.",

        "거점 직결형":
            "주요 생활권·산업·환승거점을 직접 연결합니다.",

        "간선 연장형":
            "기존 간선노선을 미래 개발지역까지 연장합니다.",

        "환승 최적화형":
            "환승 결절점을 중심으로 노선체계를 재구성합니다.",
    }

    st.info(
        descriptions[option]
    )

    st.markdown("---")

    st.subheader(
        "📍 노선 계획 후보 정류장"
    )

    selected_stops = st.multiselect(
        "노선에 포함할 정류장",
        stops["정류소명"].tolist(),
    )

    if len(selected_stops) >= 2:

        route_df = stops[
            stops["정류소명"]
            .isin(selected_stops)
        ].copy()

        route_map = folium.Map(
            location=[
                route_df["좌표(Y)"].mean(),
                route_df["좌표(X)"].mean(),
            ],
            zoom_start=12,
            tiles="OpenStreetMap",
        )

        points = []

        for _, row in route_df.iterrows():

            point = [
                row["좌표(Y)"],
                row["좌표(X)"],
            ]

            points.append(point)

            folium.CircleMarker(
                location=point,
                radius=6,
                color="#dc2626",
                fill=True,
                fill_color="#dc2626",
                fill_opacity=0.8,
                tooltip=row["정류소명"],
            ).add_to(route_map)

        # 사용자가 선택한 정류장을 연결한
        # '계획안 시각화'일 뿐 실제 기존 노선이 아님.
        folium.PolyLine(
            locations=points,
            color="#dc2626",
            weight=5,
            opacity=0.8,
            tooltip="사용자 계획 노선",
        ).add_to(route_map)

        st_folium(
            route_map,
            height=600,
            use_container_width=True,
        )

        st.caption(
            "※ 이 선은 실제 버스 노선이 아니라 "
            "선택한 정류장을 연결한 계획안 시각화입니다."
        )

    else:

        st.info(
            "최소 2개 이상의 정류장을 선택하세요."
        )


# ============================================================
# 13. CITIZEN FEEDBACK
# ============================================================

elif page == "📣 시민 피드백":

    st.header(
        "📣 시민 의견 및 이용자 피드백"
    )

    st.write(
        """
        정책 시행 이후 시민의 실제 이용 경험을
        다시 정책 계획에 반영하는 구조입니다.
        """
    )

    satisfaction = st.slider(
        "전체 대중교통 만족도",
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
        "정류장 접근성",
        1,
        5,
        3,
    )

    congestion = st.slider(
        "혼잡도 만족도",
        1,
        5,
        3,
    )

    issue = st.selectbox(
        "가장 개선이 필요한 부분",
        [
            "배차간격",
            "노선 부족",
            "환승 불편",
            "정류장 접근성",
            "혼잡",
            "통행시간",
            "기타",
        ],
    )

    opinion = st.text_area(
        "시민 의견",
        placeholder=(
            "이용하면서 불편했던 점이나 "
            "개선되었으면 하는 점을 작성해주세요."
        ),
    )

    if st.button(
        "📨 의견 제출",
        use_container_width=True,
    ):

        score = round(
            (
                satisfaction
                + waiting
                + transfer
                + accessibility
                + congestion
            )
            / 5,
            2,
        )

        st.success(
            f"의견이 접수되었습니다. "
            f"평균 만족도: {score}/5"
        )

        st.session_state[
            "last_feedback"
        ] = {
            "만족도": score,
            "주요 문제": issue,
            "의견": opinion,
        }

    if "last_feedback" in st.session_state:

        st.markdown("---")

        st.subheader(
            "🔄 정책 피드백 결과"
        )

        feedback = st.session_state[
            "last_feedback"
        ]

        st.metric(
            "평균 만족도",
            f"{feedback['만족도']}/5",
        )

        st.write(
            f"**주요 개선 요구:** "
            f"{feedback['주요 문제']}"
        )

        if feedback["의견"]:

            st.write(
                f"**시민 의견:** "
                f"{feedback['의견']}"
            )


# ============================================================
# 14. POLICY DECISION
# ============================================================

elif page == "🏛️ 정책 의사결정":

    st.header(
        "🏛️ 행정기관 정책 의사결정 지원"
    )

    st.write(
        """
        AI와 데이터 분석은 정책 대안을 제시하고,
        최종적인 정책 결정은 행정기관과 전문가가
        수행하는 구조입니다.
        """
    )

    st.markdown("---")

    option = st.selectbox(
        "검토할 정책 대안",
        [
            "현행 유지",
            "거점 직결형",
            "간선 연장형",
            "환승 최적화형",
        ],
    )

    st.subheader(
        "📊 정책 검토 항목"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "수요 효과",
            "분석 필요",
        )

        st.metric(
            "접근성 효과",
            "분석 필요",
        )

        st.metric(
            "통행시간 효과",
            "분석 필요",
        )

    with c2:

        st.metric(
            "운영비",
            "분석 필요",
        )

        st.metric(
            "환승 효과",
            "분석 필요",
        )

        st.metric(
            "시민 만족도",
            "피드백 연계",
        )

    st.markdown("---")

    decision = st.radio(
        "행정기관 의사결정",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 데이터 분석",
            "전문가 검토",
        ],
    )

    if decision == "정책 대안 채택":

        st.success(
            f"선택 정책: {option}"
        )

    elif decision == "추가 데이터 분석":

        st.warning(
            "추가 데이터 분석 후 "
            "정책 대안을 재평가합니다."
        )

    elif decision == "전문가 검토":

        st.info(
            "교통 전문가 및 관련 부서의 "
            "추가 검토가 필요합니다."
        )

    else:

        st.info(
            "현재 정책 대안을 검토 중입니다."
        )


# ============================================================
# 15. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🚌 청주시 실제 버스정류장 공간정보 기반 "
    "미래예측형 대중교통 의사결정 지원 PoC"
)
