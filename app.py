import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="AI 미래예측형 대중교통 의사결정 지원", page_icon="🚌", layout="wide")

st.title("🚌 AI 기반 미래예측형 대중교통 의사결정 지원 플랫폼")
st.caption("공모전용 PoC · 청주시 적용 시나리오")
st.info("※ 현재 프로토타입은 샘플 데이터를 기반으로 작동합니다. 실증 단계에서는 청주시의 실제 교통카드·BIS·도시계획 데이터를 연계하는 구조를 전제로 합니다.")

def haversine_km(a, b):
    lat1, lon1 = a; lat2, lon2 = b
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2-lat1); dl = radians(lon2-lon1)
    x = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2*R*atan2(sqrt(x), sqrt(1-x))

def route_distance(coords):
    return sum(haversine_km(coords[i], coords[i+1]) for i in range(len(coords)-1))

def default_data():
    return pd.DataFrame({
        "지역명":["오창읍(산업단지)","오송읍(KTX·산업)","가경동(터미널)","복대동(상업·주거)","성안동(원도심)"],
        "인구수(명)":[71000,31000,52000,53000,15000],
        "일평균 승하차(건)":[14000,7500,16000,17500,9000],
        "위도":[36.7153,36.6205,36.6240,36.6355,36.6338],
        "경도":[127.4258,127.3274,127.3900,127.4221,127.4879],
        "평균속도(km/h)":[28,25,22,20,18],
    })

st.sidebar.header("⚙️ 미래 시나리오")
up = st.sidebar.file_uploader("지역별 데이터 CSV", type=["csv"])
if up:
    try:
        df = pd.read_csv(up)
        required = {"지역명","인구수(명)","일평균 승하차(건)","위도","경도"}
        if not required.issubset(df.columns):
            st.sidebar.error("필수 열이 없습니다. 기본 샘플로 실행합니다.")
            df = default_data()
        else:
            if "평균속도(km/h)" not in df.columns: df["평균속도(km/h)"] = 20
            st.sidebar.success("사용자 데이터 로드 완료")
    except Exception as e:
        st.sidebar.error(f"CSV 오류: {e}")
        df = default_data()
else:
    df = default_data()
    st.sidebar.info("기본 샘플 데이터로 실행 중")

df = df.reset_index(drop=True)
development = st.sidebar.slider("미래 도시개발 영향(%)", 0, 50, 15, 5)
year = st.sidebar.selectbox("예측 연도", [2027,2028,2029,2030], index=1)
transfer_penalty = st.sidebar.slider("환승 추가시간(분)", 0, 15, 5)
selected = st.sidebar.radio("정책 대안", ["현행 유지","대안 A · 거점 직결형","대안 B · 간선 연장형","대안 C · 환승 최적화형"])

# 1. Demand model
model = LinearRegression().fit(df[["인구수(명)"]], df["일평균 승하차(건)"])
sim = df.copy()
sim["예측인구(명)"] = (sim["인구수(명)"]*(1+development/100)).round().astype(int)
sim["예측수요(건)"] = model.predict(sim[["예측인구(명)"]]).clip(0).round().astype(int)
base_demand = int(sim["예측수요(건)"].sum())

# 2. Route scenarios
coords = list(zip(sim["위도"], sim["경도"]))
distance = route_distance(coords)
speed = float(sim["평균속도(km/h)"].mean())
base_time = max(5.0, distance/speed*60)

defs = {
    "현행 유지": (0.00,1.00,0.0,1.00,"현재 운영체계를 유지"),
    "대안 A · 거점 직결형": (0.16,0.78,4.5,0.72,"주요 생활권·산업·환승거점을 직접 연결"),
    "대안 B · 간선 연장형": (0.08,0.90,2.8,0.88,"기존 간선노선을 주요 개발지역까지 연장"),
    "대안 C · 환승 최적화형": (0.13,0.84,3.6,0.55,"환승 결절점 중심으로 배차·환승을 최적화"),
}
colors = {"현행 유지":"gray","대안 A · 거점 직결형":"red","대안 B · 간선 연장형":"blue","대안 C · 환승 최적화형":"green"}
lines = {
    "현행 유지":coords,
    "대안 A · 거점 직결형":[coords[0],coords[2],coords[3],coords[1],coords[4]],
    "대안 B · 간선 연장형":[coords[0],coords[1],coords[2],coords[3],coords[4]],
    "대안 C · 환승 최적화형":[coords[0],coords[2],coords[1],coords[4],coords[3]],
}
rows=[]
for name,(gain,timef,cost,trf,desc) in defs.items():
    demand=base_demand*(1+gain)
    travel=base_time*timef
    transfer=transfer_penalty*trf
    wait=max(3,10*timef*(base_demand/max(demand,1))**0.25)
    carbon=max(0.5,1-gain*0.35)
    score=0.40*(demand/base_demand)+0.25*(1/max(timef,0.1))+0.20*(1/max((wait+transfer)/(10+transfer_penalty),0.1))+0.15/(1+cost/10)
    rows.append([name,demand,travel,wait,transfer,cost,carbon,score,desc])
sc = pd.DataFrame(rows,columns=["정책 대안","예상 일일 수요(건)","평균 통행시간(분)","평균 대기시간(분)","환승시간(분)","연간 추가 운영비(억원)","탄소배출 지수","종합점수","설명"])
best = sc.loc[sc["종합점수"].idxmax(),"정책 대안"]

# 3. UI
st.markdown("### ① 미래 이동수요 예측")
a,b,c,d=st.columns(4)
a.metric("예측 연도",f"{year}년")
b.metric("도시개발 영향",f"+{development}%")
c.metric("현재 총 승하차",f"{df['일평균 승하차(건)'].sum():,.0f}건")
d.metric("예측 총 수요",f"{base_demand:,.0f}건")
st.caption(f"PoC 예측식: 인구수와 일평균 승하차의 선형관계. 회귀계수 {model.coef_[0]:.3f}, 절편 {model.intercept_:.1f}")
st.dataframe(sim[["지역명","인구수(명)","예측인구(명)","일평균 승하차(건)","예측수요(건)"]].style.format("{:,.0f}"),use_container_width=True)

st.markdown("---")
st.markdown("### ② 정책 대안 비교")
view=sc[["정책 대안","예상 일일 수요(건)","평균 통행시간(분)","평균 대기시간(분)","환승시간(분)","연간 추가 운영비(억원)","탄소배출 지수","종합점수"]]
def hi(row):
    return ["background-color:#E8F5E9"]*len(row) if row["정책 대안"]==best else [""]*len(row)
st.dataframe(view.style.apply(hi,axis=1),use_container_width=True)
st.success(f"🤖 PoC 추천안: **{best}** · 최종 정책은 AI가 결정하지 않고 행정기관이 선택합니다.")

st.markdown("---")
st.markdown("### ③ 미래수요 및 노선 대안 지도")
l,r=st.columns([1.35,1])
with l:
    m=folium.Map(location=[sim["위도"].mean(),sim["경도"].mean()],zoom_start=11,tiles="CartoDB positron")
    for _,row in sim.iterrows():
        folium.CircleMarker([row["위도"],row["경도"]],radius=max(5,min(28,row["예측수요(건)"]/900)),color="black",fill=True,fill_opacity=.45,tooltip=f"{row['지역명']} | 예측수요 {row['예측수요(건)']:,}건").add_to(m)
    for name,line in lines.items():
        folium.PolyLine(line,color=colors[name],weight=7 if name==selected else 3,opacity=.85 if name==selected else .35,tooltip=name).add_to(m)
    st_folium(m,width=None,height=540)
with r:
    row=sc[sc["정책 대안"]==selected].iloc[0]
    st.subheader("행정 의사결정 지원")
    st.write(f"**선택 대안:** {selected}")
    st.write(row["설명"])
    st.metric("예상 일일 수요",f"{row['예상 일일 수요(건)']:,.0f}건")
    st.metric("평균 통행시간",f"{row['평균 통행시간(분)']:.1f}분")
    st.metric("평균 대기시간",f"{row['평균 대기시간(분)']:.1f}분")
    st.metric("환승시간",f"{row['환승시간(분)']:.1f}분")
    decision=st.selectbox("행정기관 최종 선택",["검토 중","정책 대안 채택","추가 검토"])
    if decision=="정책 대안 채택": st.success(f"최종 선택안: **{selected}**")
    elif decision=="추가 검토": st.warning("추가 데이터·전문가 검토 후 재평가합니다.")
    else: st.info("AI 분석 결과를 참고하여 행정기관이 최종 판단합니다.")

st.markdown("---")
st.markdown("### ④ 정책 시행 후 피드백 순환")
st.markdown("**정책 시행 → 운영 데이터 수집 → 시민 피드백 → AI 재학습 → 다음 정책 개선**")
with st.expander("📌 PoC 한계 및 향후 고도화"):
    st.write("현재 예측모델은 샘플 지역 기반 선형회귀이며 일부 정책 효과 지표는 시뮬레이션 가정값입니다. 실제 서비스에서는 정류장 단위 교통카드, BIS, 도로소통, 도시계획·개발사업 데이터를 통합하고 검증된 수요예측·교통시뮬레이션 모델로 고도화합니다.")
