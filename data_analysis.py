from data_maker import generate_user_table, generate_event_log
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import random


# Funnel 분석
def funnel_analysis(event_log, campaign_type):
    steps = event_log[event_log["campaign_type"] == campaign_type][
        "event_name"
    ].unique()
    steps = list(steps)
    funnel = []
    total_users = event_log[event_log["campaign_type"] == campaign_type][
        "user_id"
    ].nunique()
    for step in steps:
        users = event_log[
            (event_log["campaign_type"] == campaign_type)
            & (event_log["event_name"] == step)
        ]["user_id"].nunique()
        rate = users / total_users if total_users else 0
        funnel.append({"step": step, "users": users, "rate": rate})
    funnel_df = pd.DataFrame(funnel)
    funnel_df["step"] = pd.Categorical(
        funnel_df["step"], categories=steps, ordered=True
    )
    # Plotly Funnel Chart
    fig = go.Figure(
        go.Funnel(
            y=funnel_df["step"], x=funnel_df["users"], textinfo="value+percent initial"
        )
    )
    fig.update_layout(title=f"{campaign_type.capitalize()} Funnel (Funnel Chart)")

    return funnel_df, fig


# Cohort 분석 (특정 기준일 기준 ±10일 cohort)
def cohort_analysis(event_log, user_table, base_date):
    print(event_log.head())
    print(user_table.head())
    base_date = pd.to_datetime(base_date)
    user_table["join_date"] = pd.to_datetime(user_table["join_date"])
    user_table["cohort"] = user_table["join_date"].apply(
        lambda x: (
            "before"
            if x < base_date - pd.Timedelta(days=10)
            else "after" if x > base_date + pd.Timedelta(days=10) else "target"
        )
    )
    merged = event_log.merge(
        user_table[["user_id", "cohort"]], on="user_id", how="left"
    )
    cohort_counts = (
        merged.groupby(["cohort", "event_name"])["user_id"]
        .nunique()
        .unstack()
        .fillna(0)
    )
    cohort_counts.T.plot(kind="bar")
    plt.title("Cohort별 이벤트 참여자 수")
    plt.ylabel("Unique Users")
    print(type(cohort_counts))
    return cohort_counts


def cohort_weekly_analysis(event_log, user_table, base_date):
    """
    기준일로부터 6주간, 주차별 가입 cohort를 만들고
    각 cohort의 가입 후 0~5개월(월별) 서비스 이용률(재방문률)을 계산합니다.
    """
    base_date = pd.to_datetime(base_date)
    user_table["join_date"] = pd.to_datetime(user_table["join_date"])
    event_log["event_time"] = pd.to_datetime(event_log["event_time"])

    # 1. 가입일 기준 6개 주차 cohort 생성
    bins = [base_date + pd.Timedelta(days=7 * i) for i in range(7)]
    labels = [
        f"{i+1}주차({bins[i].strftime('%m/%d')}~{(bins[i+1]-pd.Timedelta(days=1)).strftime('%m/%d')})"
        for i in range(6)
    ]
    user_table["cohort"] = pd.cut(
        user_table["join_date"], bins=bins, labels=labels, right=False
    )

    # 2. cohort별 유저 추출
    cohort_users = user_table.dropna(subset=["cohort"])[
        ["user_id", "join_date", "cohort"]
    ]

    # 3. 각 유저의 가입 후 월차 계산
    event_log = event_log.merge(cohort_users, on="user_id", how="inner")
    event_log["month_from_join"] = (
        (
            event_log["event_time"].dt.to_period("M")
            - event_log["join_date"].dt.to_period("M")
        ).apply(lambda x: x.n)
    ).clip(lower=0, upper=5)

    # 4. cohort별, 가입 후 월별 이용 유저 수 집계
    cohort_pivot = (
        event_log.groupby(["cohort", "month_from_join"], observed=True)["user_id"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )

    # 5. cohort별 전체 유저 수로 나눠 이용률(%) 계산
    cohort_sizes = cohort_users.groupby("cohort")["user_id"].nunique()
    cohort_rate = cohort_pivot.divide(cohort_sizes, axis=0).round(3) * 100

    # NaN, inf 값 처리
    cohort_rate = cohort_rate.replace([np.inf, -np.inf], 0).fillna(0)

    # 컬럼, 인덱스 문자열화
    cohort_rate.index = cohort_rate.index.astype(str)
    cohort_rate.columns = [str(col) for col in cohort_rate.columns]
    print(type(cohort_rate))
    # 완전히 비어있는 경우 빈 DataFrame 반환
    if cohort_rate.empty:
        return pd.DataFrame()

    return cohort_rate


# LTV 분석 (유저별 캠페인 참여 횟수 기반 가상 매출)
def ltv_analysis(event_log):
    # 결제 이벤트가 있다고 가정 (event_name == 'payment'), 금액 컬럼 'amount'
    if "amount" in event_log.columns and "payment" in event_log["event_name"].unique():
        payment_log = event_log[event_log["event_name"] == "payment"]
        ltv_df = (
            payment_log.groupby("user_id")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "LTV"})
        )
    else:
        # 기존 방식(최종 전환 1건당 1만원)
        last_steps = {
            "insurance": "click_consult",
            "card": "join_event",
            "loan": "agree_terms",
        }
        event_log["is_conversion"] = event_log.apply(
            lambda x: x["event_name"] == last_steps[x["campaign_type"]], axis=1
        )
        ltv = event_log[event_log["is_conversion"]].groupby("user_id").size() * 10000
        ltv_df = ltv.reset_index().rename(columns={0: "LTV"})

    # LTV 통계 및 상위 유저 분석
    ltv_df["LTV"] = ltv_df["LTV"].fillna(0)
    ltv_stats = ltv_df["LTV"].describe()
    top10 = ltv_df.sort_values("LTV", ascending=False).head(10)

    return ltv_df, ltv_stats, top10


# AARRR 분석 (Acquisition, Activation, Retention, Revenue, Referral)
def aarrr_analysis(event_log, user_table):
    result = {}

    # Acquisition: 유입 채널별 유저 수
    if "join_channel" in user_table.columns:
        acq_by_channel = user_table.groupby("join_channel")["user_id"].nunique()
        result["Acquisition_by_channel"] = acq_by_channel

    # Activation: 첫 이벤트 후 2단계 이상 도달 유저 비율, 평균 소요 시간
    activation_users = event_log.groupby("user_id").filter(lambda x: len(x) > 1)
    activation_rate = (
        activation_users["user_id"].nunique() / event_log["user_id"].nunique()
    )
    # Activation까지 걸린 평균 시간
    activation_time = activation_users.groupby("user_id").apply(
        lambda x: (
            x["event_time"].sort_values().iloc[1]
            - x["event_time"].sort_values().iloc[0]
        ).total_seconds()
    )
    result["Activation_rate"] = activation_rate
    result["Activation_time_mean(sec)"] = activation_time.mean()

    # Retention: 가입 후 1/7/30일 리텐션
    event_log["event_date"] = pd.to_datetime(event_log["event_time"]).dt.date
    user_table["join_date"] = pd.to_datetime(user_table["join_date"]).dt.date
    merged = event_log.merge(
        user_table[["user_id", "join_date"]], on="user_id", how="left"
    )
    merged["days_from_join"] = (merged["event_date"] - merged["join_date"]).apply(
        lambda x: x.days
    )
    retention_1d = (
        merged[merged["days_from_join"] == 1]["user_id"].nunique()
        / user_table["user_id"].nunique()
    )
    retention_7d = (
        merged[merged["days_from_join"] == 7]["user_id"].nunique()
        / user_table["user_id"].nunique()
    )
    retention_30d = (
        merged[merged["days_from_join"] == 30]["user_id"].nunique()
        / user_table["user_id"].nunique()
    )
    result["Retention_1d"] = retention_1d
    result["Retention_7d"] = retention_7d
    result["Retention_30d"] = retention_30d

    # Revenue: LTV 분포, 상위/하위 그룹
    ltv_df, ltv_stats, _ = ltv_analysis(event_log)
    result["LTV_mean"] = ltv_stats["mean"]
    result["LTV_median"] = ltv_stats["50%"]
    result["LTV_top10_mean"] = (
        ltv_df.sort_values("LTV", ascending=False).head(10)["LTV"].mean()
    )
    result["LTV_bottom10_mean"] = ltv_df.sort_values("LTV").head(10)["LTV"].mean()

    # Referral: 추천 이벤트 참여 유저 수, 추천 유저의 LTV/리텐션
    if (
        "event_name" in event_log.columns
        and "referral" in event_log["event_name"].unique()
    ):
        referral_users = event_log[event_log["event_name"] == "referral"][
            "user_id"
        ].unique()
        referral_ltv = ltv_df[ltv_df["user_id"].isin(referral_users)]["LTV"].mean()
        referral_retention = merged[
            merged["user_id"].isin(referral_users) & (merged["days_from_join"] >= 7)
        ]["user_id"].nunique() / len(referral_users)
        result["Referral_user_count"] = len(referral_users)
        result["Referral_LTV_mean"] = referral_ltv
        result["Referral_7d_retention"] = referral_retention

    return result


# 아래 코드로 분석 실행 예시
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    n_users = 3000  # 예시: 전체 유저 수
    save_dir = "data"  # 데이터 저장 디렉토리

    user_table = generate_user_table(n_users, save_dir)
    event_log = generate_event_log(user_table, save_dir)
    # ...기존 코드...
    # Funnel
    for campaign in ["insurance", "card", "loan"]:
        funnel_analysis(event_log, campaign)
    # Cohort (예시 기준일: 2024-06-15)
    cohort_analysis(event_log, user_table, "2024-06-15")
    # LTV
    ltv_analysis(event_log)
    # AARRR
    aarrr_analysis(event_log)
