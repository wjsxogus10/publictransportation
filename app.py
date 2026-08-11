import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

# ============================================================
# AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼
# 청주시 적용 시나리오 / 공모전용 PoC
#
# 이 버전은 sklearn을 사용하지 않습니다.
# 따라서 기존의 feature-name / model.predict 오류가 발생하지 않습니다.
# ============================================================

st.set_page_config(
    page_title="AI 미래예측형 대중교통 의사결정 지원",
    page_icon="🚌",
    layout="wide",
)

st.title("🚌 AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼")
st.caption("공모전용 PoC · 청주시 적용 시나리오")

st.info(
    "※ 현재 프로토타입은 샘플 데이터를 기반으로 작동합니다. "
    "실증 단계에서는 청주시의 실제 교통카드·BIS·도시계획 데이터를 연계하는 구조를 전제로 합니다."
)


# ============================================================
# 기본 데이터
# ============================================================

def sample_data():
    return pd.DataFrame({
        "지역명": [
            "오창읍(산업단지)",
            "오송읍(KTX·산업)",
            "가경동(터미널)",
            "복대동(상업·주거)",
            "성안동(원도심)",
        ],
        "인구수(명)": [71000, 31000, 52000, 53000, 15000],
        "일평균 승하차(건)": [14000, 7500, 16000, 17500, 9000],
        "위도": [36.7153, 36.6205, 36.6240, 36.6355, 36.6338],
        "경도": [127.4258, 127.3274, 127.3900, 127.4221, 127.4879],
        "평균속도(km/h)": [28, 25, 22, 20, 18],
    })


# ============================================================
# 거리 계산
# ============================================================

def distance_km(a, b):
    lat1, lon1 = a
    lat2, lon2 = b

    earth = 6371.0

    p1 = radians(lat1)
    p2 = radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)

    value = (
        sin(dp / 2) ** 2
        + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    )

    return 2 * earth * atan2(
        sqrt(value),
        sqrt(1 - value)
    )


def total_distance(points):
    if len(points) < 2:
        return 0.0

    return sum(
        distance_km(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )


# ============================================================
# 데이터 입력
# ============================================================

st.sidebar.header("⚙️ 시나리오 설정")

uploaded = st.sidebar.file_uploader(
    "지역별 데이터 CSV",
    type=["csv"],
)

if uploaded is None:
    df = sample_data()
    st.sidebar.info("현재 청주시 샘플 데이터로 실행 중입니다.")

else:
    try:
        df = pd.read_csv(uploaded)

        required = {
            "지역명",
            "인구수(명)",
            "일평균 승하차(건)",
            "위도",
            "경도",
        }

        missing = required - set(df.columns)

        if missing:
            st.sidebar.error(
                "필수 열이 없습니다: "
                + ", ".join(sorted(missing))
            )
            df = sample_data()

        else:
            if "평균속도(km/h)" not in df.columns:
                df["평균속도(km/h)"] = 20

            st.sidebar.success("CSV 데이터 로드 완료")

    except Exception as exc:
        st.sidebar.error("CSV를 읽지 못했습니다.")
        st.sidebar.code(str(exc))
        df = sample_data()


# 숫자형 변환
numeric_columns = [
    "인구수(명)",
    "일평균 승하차(건)",
    "위도",
    "경도",
    "평균속도(km/h)",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

df = df.dropna(
    subset=[
        "인구수(명)",
        "일평균 승하차(건)",
        "위도",
        "경도",
    ]
).reset_index(drop=True)

if len(df) == 0:
    st.error("사용할 수 있는 데이터가 없습니다.")
    st.stop()


# ============================================================
# 사용자 설정
# ============================================================

development = st.sidebar.slider(
    "미래 도시개발 영향률 (%)",
    0,
    50,
    15,
    5,
)

year = st.sidebar.selectbox(
    "예측 연도",
    [2027, 2028, 2029, 2030],
    index=1,
)

transfer_penalty = st.sidebar.slider(
    "기준 환승 추가시간 (분)",
    0,
    15,
    5,
)

selected = st.sidebar.radio(
    "지도에 표시할 대안",
    [
        "현행 유지",
        "대안 A · 거점 직결형",
        "대안 B · 간선 연장형",
        "대안 C · 환승 최적화형",
    ],
)


# ============================================================
# 1. 미래 수요 예측
#
# 기존 오류를 완전히 제거하기 위해 외부 ML 라이브러리를
# 사용하지 않고, 투명한 단순 선형 추정식을 직접 계산합니다.
#
# y = a*x + b
# a = Σ((x-x평균)(y-y평균)) / Σ((x-x평균)^2)
# b = y평균 - a*x평균
# ============================================================

x = df["인구수(명)"].astype(float)
y = df["일평균 승하차(건)"].astype(float)

x_mean = x.mean()
y_mean = y.mean()

variance = ((x - x_mean) ** 2).sum()

if variance == 0:
    slope = 0.0
else:
    slope = (
        ((x - x_mean) * (y - y_mean)).sum()
        / variance
    )

intercept = y_mean - slope * x_mean

sim = df.copy()

sim["예측인구(명)"] = (
    sim["인구수(명)"]
    * (1 + development / 100)
).round().astype(int)

sim["예측수요(건)"] = (
    slope * sim["예측인구(명)"]
    + intercept
).clip(lower=0).round().astype(int)

base_demand = int(
    sim["예측수요(건)"].sum()
)

current_demand = int(
    df["일평균 승하차(건)"].sum()
)


# ============================================================
# 2. 노선 시나리오
# ============================================================

points = list(
    zip(
        sim["위도"].tolist(),
        sim["경도"].tolist(),
    )
)

route_distance = total_distance(points)

average_speed = float(
    sim["평균속도(km/h)"].mean()
)

base_travel_time = max(
    5.0,
    route_distance / max(average_speed, 1) * 60,
)

scenario_info = {
    "현행 유지": {
        "demand_gain": 0.00,
        "time_factor": 1.00,
        "cost": 0.0,
        "transfer_factor": 1.00,
        "color": "gray",
        "description": "현재 운영체계를 유지합니다.",
    },
    "대안 A · 거점 직결형": {
        "demand_gain": 0.16,
        "time_factor": 0.78,
        "cost": 4.5,
        "transfer_factor": 0.72,
        "color": "red",
        "description": "주요 생활권·산업·환승거점을 직접 연결합니다.",
    },
    "대안 B · 간선 연장형": {
        "demand_gain": 0.08,
        "time_factor": 0.90,
        "cost": 2.8,
        "transfer_factor": 0.88,
        "color": "blue",
        "description": "기존 간선노선을 주요 개발지역까지 연장합니다.",
    },
    "대안 C · 환승 최적화형": {
        "demand_gain": 0.13,
        "time_factor": 0.84,
        "cost": 3.6,
        "transfer_factor": 0.55,
        "color": "green",
        "description": "환승 결절점 중심으로 배차와 환승을 최적화합니다.",
    },
}


routes = {
    "현행 유지": points,
    "대안 A · 거점 직결형": [
        points[0],
        points[2],
        points[3],
        points[1],
        points[4],
    ],
    "대안 B · 간선 연장형": [
        points[0],
        points[1],
        points[2],
        points[3],
        points[4],
    ],
    "대안 C · 환승 최적화형": [
        points[0],
        points[2],
        points[1],
        points[4],
        points[3],
    ],
}


results = []

for name, info in scenario_info.items():

    demand = base_demand * (
        1 + info["demand_gain"]
    )

    travel_time = (
        base_travel_time
        * info["time_factor"]
    )

    waiting_time = max(
        3.0,
        10
        * info["time_factor"]
        * (
            base_demand / max(demand, 1)
        ) ** 0.25,
    )

    transfer_time = (
        transfer_penalty
        * info["transfer_factor"]
    )

    carbon_index = max(
        0.5,
        1 - info["demand_gain"] * 0.35,
    )

    # PoC용 비교점수
    score = (
        0.40 * (
            demand / max(base_demand, 1)
        )
        + 0.25 * (
            1 / max(info["time_factor"], 0.1)
        )
        + 0.20 * (
            1 / max(
                (waiting_time + transfer_time)
                / max(10 + transfer_penalty, 1),
                0.1,
            )
        )
        + 0.15 / (
            1 + info["cost"] / 10
        )
    )

    results.append({
        "정책 대안": name,
        "예상 일일 수요(건)": int(round(demand)),
        "평균 통행시간(분)": round(travel_time, 1),
        "평균 대기시간(분)": round(waiting_time, 1),
        "환승시간(분)": round(transfer_time, 1),
        "연간 추가 운영비(억원)": round(info["cost"], 1),
        "탄소배출 지수": round(carbon_index, 2),
        "종합점수": round(score, 3),
        "설명": info["description"],
    })


scenario_df = pd.DataFrame(results)

best_policy = scenario_df.loc[
    scenario_df["종합점수"].idxmax(),
    "정책 대안",
]


# ============================================================
# 3. 미래 이동수요
# ============================================================

st.markdown("## ① 미래 이동수요 예측")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "예측 연도",
    f"{year}년",
)

c2.metric(
    "도시개발 영향",
    f"+{development}%",
)

c3.metric(
    "현재 총 승하차",
    f"{current_demand:,}건",
)

c4.metric(
    "예측 총 수요",
    f"{base_demand:,}건",
)

st.caption(
    "PoC 예측식: "
    f"예상 승하차 = {slope:.4f} × 인구수 "
    f"+ {intercept:.1f}"
)

st.dataframe(
    sim[
        [
            "지역명",
            "인구수(명)",
            "예측인구(명)",
            "일평균 승하차(건)",
            "예측수요(건)",
        ]
    ],
    use_container_width=True,
)


# ============================================================
# 4. 정책 대안 비교
# ============================================================

st.markdown("---")
st.markdown("## ② 정책 대안 비교")

display_columns = [
    "정책 대안",
    "예상 일일 수요(건)",
    "평균 통행시간(분)",
    "평균 대기시간(분)",
    "환승시간(분)",
    "연간 추가 운영비(억원)",
    "탄소배출 지수",
    "종합점수",
]

display_df = scenario_df[display_columns].copy()

st.dataframe(
    display_df,
    use_container_width=True,
)

st.success(
    f"🤖 PoC 추천안: **{best_policy}**\n\n"
    "AI는 정책 대안을 비교·추천하고, "
    "최종 정책은 행정기관이 결정하는 구조입니다."
)


# ============================================================
# 5. 지도
# ============================================================

st.markdown("---")
st.markdown("## ③ 미래수요 및 노선 대안 지도")

map_col, info_col = st.columns(
    [1.4, 1]
)

with map_col:

    center_lat = float(
        sim["위도"].mean()
    )

    center_lon = float(
        sim["경도"].mean()
    )

    map_object = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    for _, row in sim.iterrows():

        radius = max(
            5,
            min(
                28,
                row["예측수요(건)"] / 900,
            ),
        )

        folium.CircleMarker(
            location=[
                row["위도"],
                row["경도"],
            ],
            radius=radius,
            color="black",
            fill=True,
            fill_opacity=0.45,
            tooltip=(
                f"{row['지역명']} | "
                f"예측수요 {row['예측수요(건)']:,}건"
            ),
        ).add_to(map_object)

    for name, line in routes.items():

        info = scenario_info[name]

        folium.PolyLine(
            locations=line,
            color=info["color"],
            weight=(
                7 if name == selected else 3
            ),
            opacity=(
                0.9 if name == selected else 0.3
            ),
            tooltip=name,
        ).add_to(map_object)

    st_folium(
        map_object,
        width=None,
        height=540,
    )


# ============================================================
# 6. 행정 의사결정
# ============================================================

with info_col:

    selected_row = scenario_df[
        scenario_df["정책 대안"] == selected
    ].iloc[0]

    st.subheader(
        "🏛️ 행정 의사결정 지원"
    )

    st.write(
        f"**검토 대안:** {selected}"
    )

    st.write(
        selected_row["설명"]
    )

    st.metric(
        "예상 일일 수요",
        f"{selected_row['예상 일일 수요(건)']:,}건",
    )

    st.metric(
        "평균 통행시간",
        f"{selected_row['평균 통행시간(분)']:.1f}분",
    )

    st.metric(
        "평균 대기시간",
        f"{selected_row['평균 대기시간(분)']:.1f}분",
    )

    st.metric(
        "환승시간",
        f"{selected_row['환승시간(분)']:.1f}분",
    )

    decision = st.selectbox(
        "행정기관 최종 결정",
        [
            "검토 중",
            "정책 대안 채택",
            "추가 검토",
        ],
    )

    if decision == "정책 대안 채택":
        st.success(
            f"최종 선택안: **{selected}**"
        )

    elif decision == "추가 검토":
        st.warning(
            "추가 데이터와 전문가 검토 후 "
            "재평가합니다."
        )

    else:
        st.info(
            "AI 분석 결과를 참고하여 "
            "행정기관이 최종 판단합니다."
        )


# ============================================================
# 7. 피드백 순환
# ============================================================

st.markdown("---")
st.markdown("## ④ 정책 시행 후 피드백 순환")

st.markdown(
    "**정책 시행 → 운영 데이터 수집 → 시민 피드백 → "
    "AI 재학습 → 다음 정책 개선**"
)

st.info(
    "향후 실증 단계에서는 실제 운영 결과와 시민 평가를 "
    "다음 수요예측 및 정책 대안 생성에 반영하는 "
    "순환형 의사결정 구조로 고도화합니다."
)


# ============================================================
# 8. PoC 한계 및 향후 고도화
# ============================================================

with st.expander("📌 PoC 한계 및 향후 고도화"):

    st.write(
        """
현재 프로토타입은 공모전 시연을 위한 PoC입니다.

• 현재는 청주시 샘플 데이터를 사용합니다.
• 미래수요는 인구와 승하차량의 단순 선형관계를 이용해 추정합니다.
• 일부 정책 효과 지표는 시뮬레이션을 위한 가정값입니다.
• 실제 서비스에서는 교통카드 승·하차 및 환승 데이터,
  BIS 운행정보, 도로소통정보, 생활인구,
  도시·군기본계획, 공동주택·산업단지·철도·상업시설
  개발정보 등을 통합합니다.
• 향후 실제 교통수요예측 및 교통시뮬레이션 모델과
  AI를 결합하여 정책 대안의 정확도를 고도화합니다.
• AI가 최종 정책을 자동 결정하는 것이 아니라
  행정기관의 의사결정을 지원하는 구조를 유지합니다.
        """
    )
