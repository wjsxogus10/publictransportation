import os
from pathlib import Path

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="청주시 대중교통 계획 플랫폼",
    page_icon="🚌",
    layout="wide",
)

st.title("🚌 청주시 AI 기반 미래예측형 대중교통 계획 플랫폼")
st.caption(
    "청주시 실제 버스정류장 위치 데이터를 기반으로 한 공모전용 PoC"
)


# ============================================================
# 실제 청주시 정류장 데이터
# ============================================================

STOP_FILE = Path(__file__).with_name(
    "충청북도_청주시_버스정보시스템_20250401.csv"
)


@st.cache_data
def load_bus_stops():

    if not STOP_FILE.exists():
        return None, (
            "정류장 CSV 파일을 찾을 수 없습니다. "
            "app.py와 같은 폴더에 "
            "'충청북도_청주시_버스정보시스템_20250401.csv' "
            "파일을 넣어주세요."
        )

    # 한글 CSV 인코딩 자동 처리
    last_error = None

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
            df = None

    if df is None:
        return None, f"CSV 읽기 오류: {last_error}"

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required = [
        "서비스ID",
        "정류소명",
        "좌표(X)",
        "좌표(Y)",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        return None, (
            "필수 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    # 좌표 숫자 변환
    df["좌표(X)"] = pd.to_numeric(
        df["좌표(X)"],
        errors="coerce",
    )

    df["좌표(Y)"] = pd.to_numeric(
        df["좌표(Y)"],
        errors="coerce",
    )

    # 유효 좌표만 유지
    df = df.dropna(
        subset=[
            "좌표(X)",
            "좌표(Y)",
        ]
    ).copy()

    # 대한민국 범위 필터
    df = df[
        df["좌표(X)"].between(124, 132)
        & df["좌표(Y)"].between(33, 39)
    ].copy()

    # 서비스ID 중복 제거
    df = df.drop_duplicates(
        subset=["서비스ID"]
    ).reset_index(drop=True)

    return df, None


stops, load_error = load_bus_stops()


# ============================================================
# 오류 처리
# ============================================================

if stops is None:

    st.error(load_error)

    st.info(
        """
        GitHub 프로젝트 구조를 다음과 같이 만들어 주세요.

        ```
        publictransportation/
        ├── app.py
        ├── requirements.txt
        └── 충청북도_청주시_버스정보시스템_20250401.csv
        ```
        """
    )

    st.stop()


# ============================================================
# 사이드바
# ============================================================

st.sidebar.header("🚌 청주시 정류장 데이터")

st.sidebar.success(
    f"실제 정류장 데이터 연결 완료\n\n"
    f"{len(stops):,}개 정류장"
)

st.sidebar.caption(
    "정류장 위치 데이터: "
    "서비스ID / 정류소명 / 좌표(X) / 좌표(Y)"
)

st.sidebar.markdown("---")

search_text = st.sidebar.text_input(
    "🔎 정류장 검색",
    placeholder="예: 내덕, 성안길, 청주대학교",
)

show_labels = st.sidebar.checkbox(
    "정류장 이름 표시",
    value=False,
)

map_zoom = st.sidebar.slider(
    "지도 확대 수준",
    min_value=10,
    max_value=16,
    value=12,
)


# ============================================================
# 검색
# ============================================================

filtered = stops.copy()

if search_text.strip():

    q = search_text.strip().lower()

    filtered = filtered[
        filtered["정류소명"]
        .astype(str)
        .str.lower()
        .str.contains(
            q,
            na=False,
        )
        |
        filtered["서비스ID"]
        .astype(str)
        .str.lower()
        .str.contains(
            q,
            na=False,
        )
    ].copy()


# ============================================================
# 상단 KPI
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "전체 정류장",
    f"{len(stops):,}개",
)

c2.metric(
    "검색 결과",
    f"{len(filtered):,}개",
)

c3.metric(
    "경도 범위",
    f"{stops['좌표(X)'].min():.3f} ~ "
    f"{stops['좌표(X)'].max():.3f}",
)

c4.metric(
    "위도 범위",
    f"{stops['좌표(Y)'].min():.3f} ~ "
    f"{stops['좌표(Y)'].max():.3f}",
)


# ============================================================
# 지도
# ============================================================

st.markdown("---")
st.subheader("🗺️ 청주시 실제 버스 정류장 전체 지도")

if filtered.empty:

    st.warning(
        "검색 결과가 없습니다. "
        "다른 정류장명을 입력해 주세요."
    )

else:

    center_lat = filtered["좌표(Y)"].mean()
    center_lon = filtered["좌표(X)"].mean()

    m = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=map_zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 정류장 수가 많아도 지도 성능을 고려하여
    # CircleMarker를 사용한다.
    marker_radius = (
        5 if len(filtered) <= 500
        else 3
    )

    for _, row in filtered.iterrows():

        popup_html = f"""
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
            radius=marker_radius,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.65,
            weight=1,
            tooltip=(
                str(row["정류소명"])
                if not show_labels
                else None
            ),
            popup=folium.Popup(
                popup_html,
                max_width=320,
            ),
        ).add_to(m)

        if show_labels:

            folium.Marker(
                location=[
                    row["좌표(Y)"],
                    row["좌표(X)"],
                ],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-size:10px;
                        font-weight:600;
                        white-space:nowrap;
                        transform:translate(7px,-8px);
                    ">
                    {row['정류소명']}
                    </div>
                    """
                ),
            ).add_to(m)

    st_folium(
        m,
        height=680,
        use_container_width=True,
    )


# ============================================================
# 정류장 정보
# ============================================================

st.markdown("---")
st.subheader("📋 정류장 정보")

display_df = filtered[
    [
        "서비스ID",
        "정류소명",
        "좌표(X)",
        "좌표(Y)",
    ]
].copy()

display_df = display_df.rename(
    columns={
        "서비스ID": "서비스 ID",
        "정류소명": "정류장명",
        "좌표(X)": "경도",
        "좌표(Y)": "위도",
    }
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 정류장 선택
# ============================================================

st.markdown("---")
st.subheader("📍 정류장 상세 분석")

if not filtered.empty:

    selected_stop = st.selectbox(
        "분석할 정류장을 선택하세요",
        filtered["정류소명"].tolist(),
    )

    selected_row = filtered[
        filtered["정류소명"]
        == selected_stop
    ].iloc[0]

    a, b, c = st.columns(3)

    a.metric(
        "정류장",
        selected_row["정류소명"],
    )

    b.metric(
        "경도",
        f"{selected_row['좌표(X)']:.6f}",
    )

    c.metric(
        "위도",
        f"{selected_row['좌표(Y)']:.6f}",
    )

    st.info(
        f"""
        **서비스 ID:** {selected_row['서비스ID']}

        **위치:** 
        {selected_row['좌표(Y)']:.6f},
        {selected_row['좌표(X)']:.6f}

        이 위치를 향후 인구·수요·도시개발 데이터와 연결하여
        정류장별 대중교통 수요 및 접근성 분석에 활용할 수 있습니다.
        """
    )


# ============================================================
# 플랫폼 확장 구조
# ============================================================

st.markdown("---")
st.subheader("🔄 미래 대중교통 계획 구조")

st.markdown(
    """
    **① 실제 정류장 데이터**

    ↓

    **② 인구·도시개발 변화 입력**

    ↓

    **③ 정류장별 미래 수요 예측**

    ↓

    **④ 노선 개편 대안 생성**

    ↓

    **⑤ 통행시간·환승·접근성 비교**

    ↓

    **⑥ 시민 피드백 수집**

    ↓

    **⑦ 정책 재평가**
    """
)

with st.expander(
    "📌 현재 데이터의 한계와 다음 단계"
):

    st.write(
        """
        현재 연결된 청주시 BIS 파일은 정류장의
        서비스ID·정류소명·좌표 정보를 제공하는
        '정류장 위치 데이터'입니다.

        따라서 현재 단계에서는 실제 청주시 전체
        정류장을 정확한 위치에 표시할 수 있습니다.

        다만 실제 버스 노선의 운행 순서를 선으로
        연결하려면 별도의 '노선ID + 정류장ID +
        정류장순번' 데이터가 필요합니다.

        해당 데이터를 추가하면 실제 노선망을 지도에
        표현하고 기존 노선과 미래 노선 대안을
        비교하는 기능으로 확장할 수 있습니다.
        """
    )

st.caption(
    "청주시 버스정보시스템 정류장 위치 데이터를 "
    "기반으로 한 공모전용 프로토타입"
)
