import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import platform
from data_analysis import (
    funnel_analysis,
    cohort_analysis,
    cohort_weekly_analysis,
    ltv_analysis,
    aarrr_analysis,
)
from data_maker import generate_user_table, generate_event_log

# 한글 폰트 설정
if platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # macOS
    matplotlib.rc("font", family="AppleGothic")
else:  # Linux
    matplotlib.rc("font", family="NanumGothic")

if "user_table" not in st.session_state:
    st.session_state.user_table = generate_user_table(3000, "data")
if "event_log" not in st.session_state:
    st.session_state.event_log = generate_event_log(st.session_state.user_table, "data")


st.sidebar.header("분석 선택")
analysis_type = st.sidebar.selectbox(
    "분석 유형을 선택하세요", ("Main", "Funnel", "Cohort", "LTV", "AARRR")
)
if analysis_type == "Main":
    st.title("서비스 데이터 분석 대시보드")

    # 첫 화면: 데이터 생성 방법 및 결과 요약
    with st.expander("데이터 생성 방법 및 결과 요약", expanded=True):
        st.markdown(
            """
        ### 데이터 생성 방법
        - **유저 테이블**: 성별, 나이대, 가입연도, 가입일을 무작위로 생성하여 3,000명의 가상 유저 데이터를 만듭니다.
        - **이벤트 로그**: 각 유저가 보험, 카드, 대출 등 여러 캠페인에 참여하며, 각 캠페인별 여정(여러 단계)을 시간 순서대로 시뮬레이션합니다.
        - 각 단계별로 이탈 확률을 적용해 실제 서비스와 유사한 전환/이탈 패턴을 반영했습니다.
        - 최종 단계까지 도달하는 유저는 약 5%로 설정되어 있습니다.
        - LTV, Cohort, Funnel, AARRR 등 다양한 분석이 가능한 구조로 설계되었습니다.

        ### 데이터셋 요약
        - **유저 수**: {user_count}명
        - **이벤트 로그 수**: {event_count}건
        - **캠페인 종류**: 보험, 카드, 대출
        - **최종 전환 유저 비율**: 약 5%
        """.format(
                user_count=len(st.session_state.user_table),
                event_count=len(st.session_state.event_log),
            )
        )

        st.markdown("#### 유저 테이블 샘플")
        st.dataframe(st.session_state.user_table.head())

        st.markdown("#### 이벤트 로그 샘플")
        st.dataframe(st.session_state.event_log.head())
elif analysis_type == "Funnel":
    st.header("Funnel 분석")
    st.markdown(
        """
    **Funnel 분석이란?**  
    서비스의 각 단계별로 유저가 얼마나 이탈하거나 전환하는지 시각적으로 보여주는 분석입니다.  
    이를 통해 전환율이 낮은 단계(이탈 구간)를 파악하고, 개선 포인트를 찾을 수 있습니다.
    """
    )
    campaign = st.selectbox("캠페인 선택", ["insurance", "card", "loan"])
    funnel_df, fig = funnel_analysis(st.session_state.event_log, campaign)
    st.subheader(f"{campaign.capitalize()} Funnel 분석 결과")
    st.dataframe(funnel_df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
    **인사이트 예시:**  
    - 특정 단계에서 전환율이 급격히 떨어진다면, 해당 단계의 UX/UI 개선이나 추가 안내가 필요할 수 있습니다.
    - 각 캠페인별로 유저의 이탈 패턴이 다르다면, 맞춤형 전략 수립이 가능합니다.
    """
    )

elif analysis_type == "Cohort":
    st.header("Cohort 분석")
    st.markdown(
        """
    **Cohort 분석이란?**  
    유저를 가입 시점(또는 특정 기준일)별로 그룹화(cohort)하여,  
    각 그룹의 행동 패턴이나 서비스 이용률을 비교하는 분석입니다.  
    이를 통해 시기별 유저의 리텐션, 전환, 이탈 등 행동 변화를 파악할 수 있습니다.
    """
    )
    base_date = st.date_input("기준일을 선택하세요", value=pd.to_datetime("2024-06-15"))
    cohort_counts = cohort_analysis(
        st.session_state.event_log, st.session_state.user_table, base_date
    )

    st.subheader("Cohort별 이벤트 참여자 수")
    # 완전히 비어있는 경우 예외 처리

    st.dataframe(cohort_counts)
    st.bar_chart(cohort_counts)

    cohort_rate = cohort_weekly_analysis(
        st.session_state.event_log, st.session_state.user_table, base_date
    )
    cohort_rate = cohort_rate.round(1)
    st.write("### 주차별 Cohort 월별 서비스 이용률 (%)")
    st.write("#### Heatmap (월별 이용률)")
    st.write("가로: 가입 후 월, 세로: Cohort 주차")
    st.dataframe(
        cohort_rate.style.format("{:.1f}").background_gradient(cmap="Blues", axis=None)
    )
    st.markdown(
        """
    **인사이트 예시:**  
    - 특정 시기(주차)에 가입한 유저의 리텐션이 높거나 낮은 패턴을 확인할 수 있습니다.
    - 신규 유저와 기존 유저의 서비스 적응 속도, 이탈률 차이를 파악해 마케팅/온보딩 전략에 활용할 수 있습니다.
    """
    )

elif analysis_type == "LTV":
    st.header("LTV 분석")
    st.markdown(
        """
    **LTV(Lifetime Value) 분석이란?**  
    유저가 서비스에 머무는 동안 발생시키는 총 가치를 추정하는 분석입니다.  
    LTV가 높을수록 충성도 높은 유저가 많다는 의미이며, 마케팅 투자 효율성 판단에 활용됩니다.
    
    본 분석에서의 매출 정의는 유저가 서비스 이용 중 발생한 이벤트(예: 보험 가입, 카드 발급 등)로부터 최종 단계까지 도달한 유저마다 1만원의 매출이 발생한다고 가정합니다.
    """
    )
    ltv_df, ltv_stats, top10 = ltv_analysis(st.session_state.event_log)
    st.subheader("LTV 분석 결과")
    st.dataframe(ltv_stats)
    st.subheader("\n[LTV 상위 10명]")
    st.dataframe(top10)

    all_users = pd.DataFrame({"user_id": st.session_state.user_table["user_id"]})
    ltv_full = all_users.merge(ltv_df, on="user_id", how="left").fillna(0)
    counts, bins, patches = plt.hist(
        ltv_full["LTV"], bins=[0, 10000, 20000, 30000], edgecolor="black"
    )
    plt.yscale("log")
    plt.title("유저별 LTV 분포")
    plt.xlabel("LTV (KRW)")
    plt.ylabel("User Count")
    plt.xticks([0, 10000, 20000])
    for count, bin_left, patch in zip(counts, bins, patches):
        if count > 0:
            plt.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="blue",
            )
    st.subheader("LTV 분포 히스토그램")
    st.write(
        "LTV 분포를 시각화한 히스토그램입니다., 구간별 유저 수의 편차가 커 log 스케일로 표시했습니다."
    )
    st.pyplot(plt)
    st.markdown(
        """
    **인사이트 예시:**  
    - LTV가 높은 유저의 특성을 분석해, 유사 유저 타겟팅 및 리텐션 전략을 세울 수 있습니다.
    - LTV 분포가 한쪽으로 치우쳐 있다면, 서비스 구조나 과금 정책 개선이 필요할 수 있습니다.
    """
    )

elif analysis_type == "AARRR":
    st.header("AARRR 분석")
    st.markdown(
        """
    **AARRR 분석이란?**  
    스타트업/서비스 성장의 5단계(획득, 활성화, 유지, 수익, 추천)를 지표로  
    유저의 전체 여정을 분석하는 프레임워크입니다.
    """
    )
    st.info(
        """ 
            Acquisition
            
            유입 채널별(Organic, Ad, Referral 등) 유저 수
            캠페인별 첫 유입 유저 수
            신규/재방문 유저 비율

            Activation
            
            첫 이벤트 후 2~3단계까지 도달한 유저 비율
            Activation까지 걸린 평균 시간
            Activation 단계별 이탈률

            Retention
            
            가입 후 1/7/30일 리텐션(잔존율)
            주차별/월별 리텐션 곡선
            리텐션 상위/하위 그룹 특성

            Revenue
            
            LTV 분포(상위 10%, 하위 10% 등)
            캠페인/채널별 평균 LTV
            LTV 상위 유저의 행동 패턴

            Referral
            
            추천 이벤트 참여 유저 수
            추천을 통한 신규 유저 유입 비율
            추천 유저의 LTV/리텐션 비교"""
    )
    aarrr = aarrr_analysis(st.session_state.event_log, st.session_state.user_table)
    st.subheader("AARRR 심층 분석 결과")
    items = list(aarrr.items())
    for row in range(3):
        cols = st.columns(3)
        for col_idx in range(3):
            idx = row * 3 + col_idx
            if idx < len(items):
                k, v = items[idx]
                cols[col_idx].markdown(f"###### {k}")
                if isinstance(v, float):
                    v_str = f"{v:,.2f}"
                else:
                    v_str = str(v)
                cols[col_idx].markdown(
                    f"<span style='font-size:1.5em; color:#2E86C1;'><b>{v_str}</b></span>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
    **인사이트 예시:**  
    - 유입 채널별로 LTV/리텐션 차이가 크다면, 효율적인 채널에 마케팅 예산을 집중하세요.
    - Activation까지 평균 소요 시간이 길면, 온보딩 과정을 개선해보세요.
    - Retention이 특정 시점에 급락한다면, 해당 시점의 UX나 혜택을 점검하세요.
    - Referral 유저의 LTV/리텐션이 높다면, 추천 프로그램을 강화할 가치가 있습니다.
    """
    )
