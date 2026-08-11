import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

# ============================================================
# AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼
# 공모전용 PoC · 청주시 적용 시나리오
#
# 추가 기능
# 1) 미래 도시변화 시나리오
# 2) AI 추천 이유 설명
# 3) 시민 평가 및 정책 피드백
#
# ※ sklearn / model.predict()를 사용하지 않습니다.
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
# 함수
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
        sqrt(1 - value),
    )


def total_distance(points):
    if len(points) < 2:
        return 0.0

    return sum(
        distance_km(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )


def linear_estimate(x, y, future_x):
    """
    외부 ML 라이브러리 없이 단순 선형관계를 계산합니다.
    y = slope*x + intercept
    """
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        slope = 0.0
    else:
        slope = (
            ((x - x_mean) * (y - y_mean)).sum()
            / denominator
        )

    intercept = y_mean - slope * x_mean

    prediction = slope * pd.Series(
        future_x,
        dtype=float,
    ) + intercept

    return (
        prediction.clip(lower=0),
        float(slope),
        float(intercept),
    )


# ============================================================
# 데이터
# ============================================================

st.sidebar.header("⚙️ 미래 시나리오 설정")

uploaded = st.sidebar.file_uploader(
    "지역별 데이터 CSV",
    type=["csv"],
)

if uploaded is None:

    df = sample_data()

    st.sidebar.info(
        "현재 청주시 샘플 데이터로 실행 중입니다."
    )

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

            st.sidebar.success(
                "CSV 데이터 로드 완료"
            )

    except Exception as exc:

        st.sidebar.error(
            "CSV를 읽지 못했습니다."
        )

        st.sidebar.code(str(exc))

        df = sample_data()


for column in [
    "인구수(명)",
    "일평균 승하차(건)",
    "위도",
    "경도",
    "평균속도(km/h)",
]:

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


if len(df) < 2:

    st.error(
        "분석을 위해 최소 2개 지역의 데이터가 필요합니다."
    )

    st.stop()


# ============================================================
# ① 미래 도시변화 시나리오
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🏙️ 미래 도시변화 설정")

year = st.sidebar.selectbox(
    "예측 연도",
    [2027, 2028, 2029, 2030],
    index=1,
)

development_options = {
    "신규 공동주택 개발": 8,
    "산업단지 조성·확장": 7,
    "철도 개통·환승거점 강화": 6,
    "대규모 상업시설 개발": 5,
    "대규모 행사·관광객 증가": 4,
}

selected_developments = []

for label, impact in development_options.items():

    checked = st.sidebar.checkbox(
        label,
        value=False,
    )

    if checked:
        selected_developments.append(
            (label, impact)
        )


base_development = 5

scenario_impact = base_development + sum(
    impact for _, impact in selected_developments
)

scenario_impact = min(
    scenario_impact,
    50,
)


st.sidebar.metric(
    "AI가 반영할 미래 변화 영향",
    f"+{scenario_impact}%",
)


# ============================================================
# 기본 운영 변수
# ============================================================

transfer_penalty = st.sidebar.slider(
    "기준 환승 추가시간(분)",
    0,
    15,
    5,
)

selected = st.sidebar.radio(
    "지도에 표시할 정책 대안",
    [
        "현행 유지",
        "대안 A · 거점 직결형",
        "대안 B · 간선 연장형",
        "대안 C · 환승 최적화형",
    ],
)


# ============================================================
# 미래 수요
# ============================================================

future_population = (
    df["인구수(명)"]
    * (1 + scenario_impact / 100)
).round().astype(int)


predicted, slope, intercept = linear_estimate(
    df["인구수(명)"],
    df["일평균 승하차(건)"],
    future_population,
)


sim = df.copy()

sim["예측인구(명)"] = future_population

sim["예측수요(건)"] = (
    predicted.round().astype(int)
)


base_demand = int(
    sim["예측수요(건)"].sum()
)

current_demand = int(
    df["일평균 승하차(건)"].sum()
)


# ============================================================
# 노선 시나리오
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
    route_distance / max(
        average_speed,
        1,
    ) * 60,
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

    "현행 유지":
        points,

    "대안 A · 거점 직결형":
        [
            points[0],
            points[2],
            points[3],
            points[1],
            points[4],
        ],

    "대안 B · 간선 연장형":
        [
            points[0],
            points[1],
            points[2],
            points[3],
            points[4],
        ],

    "대안 C · 환승 최적화형":
        [
            points[0],
            points[2],
            points[1],
            points[4],
            points[3],
        ],
}


results = []


for name, info in scenario_info.items():

    demand = (
        base_demand
        * (1 + info["demand_gain"])
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
            base_demand
            / max(demand, 1)
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

    score = (

        0.40
        * (
            demand
            / max(base_demand, 1)
        )

        + 0.25
        * (
            1
            / max(
                info["time_factor"],
                0.1,
            )
        )

        + 0.20
        * (
            1
            / max(
                (
                    waiting_time
                    + transfer_time
                )
                / max(
                    10
                    + transfer_penalty,
                    1,
                ),
                0.1,
            )
        )

        + 0.15
        / (
            1
            + info["cost"] / 10
        )
    )

    results.append({

        "정책 대안": name,

        "예상 일일 수요(건)":
            int(round(demand)),

        "평균 통행시간(분)":
            round(travel_time, 1),

        "평균 대기시간(분)":
            round(waiting_time, 1),

        "환승시간(분)":
            round(transfer_time, 1),

        "연간 추가 운영비(억원)":
            round(info["cost"], 1),

        "탄소배출 지수":
            round(carbon_index, 2),

        "종합점수":
            round(score, 3),

        "설명":
            info["description"],
    })


scenario_df = pd.DataFrame(results)


best_policy = scenario_df.loc[
    scenario_df["종합점수"].idxmax(),
    "정책 대안",
]


best_row = scenario_df[
    scenario_df["정책 대안"]
    == best_policy
].iloc[0]


# ============================================================
# 화면 ①
# ============================================================

st.markdown(
    "## ① 미래 이동수요 예측"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "예측 연도",
    f"{year}년",
)

c2.metric(
    "미래 도시변화 영향",
    f"+{scenario_impact}%",
)

c3.metric(
    "현재 총 승하차",
    f"{current_demand:,}건",
)

c4.metric(
    "예측 총 수요",
    f"{base_demand:,}건",
)


if selected_developments:

    st.success(
        "반영된 미래 도시변화: "
        + ", ".join(
            label
            for label, _ in selected_developments
        )
    )

else:

    st.caption(
        "선택된 미래 개발요인이 없어 기본적인 "
        "도시 성장 영향만 반영합니다."
    )


st.caption(
    "PoC 수요예측식: "
    f"예상수요 = {slope:.4f} × 예상인구 "
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
# 화면 ②
# ============================================================

st.markdown("---")

st.markdown(
    "## ② AI 정책 대안 비교"
)


view = scenario_df[
    [
        "정책 대안",
        "예상 일일 수요(건)",
        "평균 통행시간(분)",
        "평균 대기시간(분)",
        "환승시간(분)",
        "연간 추가 운영비(억원)",
        "탄소배출 지수",
        "종합점수",
    ]
]


st.dataframe(
    view,
    use_container_width=True,
)


# ============================================================
# 추가 기능 1
# AI 추천 이유
# ============================================================

st.markdown(
    "### 🤖 AI 추천 이유"
)

reason_col1, reason_col2 = st.columns(
    [1, 1]
)

with reason_col1:

    st.success(
        f"**추천 대안: {best_policy}**"
    )

    st.write(
        best_row["설명"]
    )

    st.metric(
        "예상 수요",
        f"{best_row['예상 일일 수요(건)']:,}건",
    )

    st.metric(
        "통행시간",
        f"{best_row['평균 통행시간(분)']:.1f}분",
    )


with reason_col2:

    st.markdown(
        "**추천 근거**"
    )

    current_best = scenario_df[
        scenario_df["정책 대안"]
        == "현행 유지"
    ].iloc[0]

    demand_change = (
        (
            best_row["예상 일일 수요(건)"]
            / max(
                current_best["예상 일일 수요(건)"],
                1,
            )
        ) - 1
    ) * 100

    time_change = (
        1 -
        best_row["평균 통행시간(분)"]
        / max(
            current_best["평균 통행시간(분)"],
            0.1,
        )
    ) * 100

    transfer_change = (
        1 -
        best_row["환승시간(분)"]
        / max(
            current_best["환승시간(분)"],
            0.1,
        )
    ) * 100

    st.write(
        f"• 예상 이용수요 **{demand_change:+.1f}%**"
    )

    st.write(
        f"• 평균 통행시간 **{time_change:+.1f}%**"
    )

    st.write(
        f"• 환승시간 **{transfer_change:+.1f}%**"
    )

    st.write(
        f"• 연간 추가 운영비 "
        f"**{best_row['연간 추가 운영비(억원)']:.1f}억원**"
    )

    st.write(
        f"• 종합점수 **{best_row['종합점수']:.3f}**"
    )


st.info(
    "※ AI는 여러 지표를 종합하여 정책 대안을 추천하며, "
    "최종 정책 결정은 행정기관이 담당합니다."
)


# ============================================================
# 화면 ③ 지도
# ============================================================

st.markdown("---")

st.markdown(
    "## ③ 미래수요 및 노선 대안 지도"
)


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
                f"예측수요 "
                f"{row['예측수요(건)']:,}건"
            ),
        ).add_to(map_object)


    for name, line in routes.items():

        info = scenario_info[name]

        folium.PolyLine(
            locations=line,
            color=info["color"],
            weight=(
                7
                if name == selected
                else 3
            ),
            opacity=(
                0.9
                if name == selected
                else 0.3
            ),
            tooltip=name,
        ).add_to(map_object)


    st_folium(
        map_object,
        width=None,
        height=540,
    )


with info_col:

    selected_row = scenario_df[
        scenario_df["정책 대안"]
        == selected
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
# 추가 기능 2
# 시민 평가
# ============================================================

st.markdown("---")

st.markdown(
    "## ④ 정책 시행 후 시민 평가"
)

st.caption(
    "시범 정책이 시행되었다고 가정하고 시민의 체감 평가를 입력합니다."
)


feedback_col1, feedback_col2 = st.columns(
    [1, 1]
)


with feedback_col1:

    satisfaction = st.slider(
        "전반적인 대중교통 만족도",
        1,
        5,
        4,
    )

    waiting_feedback = st.slider(
        "대기시간 만족도",
        1,
        5,
        4,
    )

    transfer_feedback = st.slider(
        "환승 편의 만족도",
        1,
        5,
        4,
    )


with feedback_col2:

    citizen_comment = st.text_area(
        "시민 의견",
        placeholder=(
            "예: 출퇴근 시간 배차간격을 줄여주세요."
        ),
    )

    feedback_average = round(
        (
            satisfaction
            + waiting_feedback
            + transfer_feedback
        ) / 3,
        2,
    )

    st.metric(
        "시민 종합 만족도",
        f"{feedback_average:.2f} / 5.00",
    )


if st.button(
    "📨 시민 피드백 분석"
):

    if feedback_average >= 4:

        st.success(
            "시민 만족도가 높습니다. "
            "현재 정책의 유지·확대를 우선 검토할 수 있습니다."
        )

    elif feedback_average >= 3:

        st.warning(
            "시민 만족도가 보통 수준입니다. "
            "대기시간·환승체계 등 세부 운영 개선이 필요합니다."
        )

    else:

        st.error(
            "시민 만족도가 낮습니다. "
            "운영 데이터와 시민 의견을 함께 분석하여 "
            "정책 재설계를 검토해야 합니다."
        )

    if citizen_comment.strip():

        st.write(
            "**AI 피드백 분류 예시:** "
            f"{citizen_comment}"
        )


# ============================================================
# 추가 기능 3
# 정책 피드백 → 다음 정책
# ============================================================

st.markdown("---")

st.markdown(
    "## ⑤ 정책 시행 → AI 개선 순환"
)


feedback_score = (
    feedback_average / 5
)


# 시민 만족도가 낮으면 개선 필요성을 크게 표시
if feedback_score >= 0.8:

    improvement_status = (
        "현재 정책의 효과가 양호합니다."
    )

    improvement_value = (
        "유지·확대 검토"
    )

elif feedback_score >= 0.6:

    improvement_status = (
        "일부 운영요소의 개선이 필요합니다."
    )

    improvement_value = (
        "부분 조정 검토"
    )

else:

    improvement_status = (
        "정책 재설계 필요성이 높습니다."
    )

    improvement_value = (
        "노선·배차 재검토"
    )


loop1, loop2, loop3, loop4 = st.columns(4)


loop1.metric(
    "① 정책 시행",
    selected,
)

loop2.metric(
    "② 시민 만족도",
    f"{feedback_average:.2f}/5",
)

loop3.metric(
    "③ AI 학습 반영",
    "반영 예정",
)

loop4.metric(
    "④ 다음 정책",
    improvement_value,
)


st.info(
    f"**피드백 분석 결과:** {improvement_status}\n\n"
    "실증 단계에서는 실제 교통카드·BIS 운영 데이터와 "
    "시민 피드백을 축적하여 다음 수요예측과 정책 대안 생성에 반영합니다."
)


# ============================================================
# 향후 고도화
# ============================================================

with st.expander(
    "📌 PoC 한계 및 향후 고도화"
):

    st.write(
        """
현재 프로토타입은 공모전 시연을 위한 PoC입니다.

• 현재는 청주시 샘플 데이터를 사용합니다.
• 미래수요는 인구와 승하차량의 단순 선형관계를 이용해 추정합니다.
• 미래 도시변화 영향률은 공모전 시연을 위한 가정값입니다.
• 노선별 운영비·통행시간·탄소배출 지표 역시 PoC용 시뮬레이션 값입니다.
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
