import pandas as pd
import numpy as np
from datetime import timedelta
import random
import matplotlib.pyplot as plt
import os
import sys


def generate_user_table(n_users, save_dir):
    """
    유저 테이블을 생성하는 함수
    :param n_users: 생성할 유저 수
    :return: 유저 정보가 담긴 DataFrame
    """
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if not os.path.exists(f"{save_dir}/user_table.csv"):
        # 성별
        genders = np.random.choice(["M", "F"], size=n_users, p=[0.5, 0.5])

        # 나이대
        age_bins = ["10", "20", "30", "40", "50", "60+"]
        age_probs = [0.1, 0.2, 0.3, 0.2, 0.15, 0.05]  # 예시 분포
        ages = np.random.choice(age_bins, size=n_users, p=age_probs)

        # 가입연도 생성
        join_years = np.random.choice(range(2024, 2025), size=n_users)

        # 각 연도별로 무작위 월, 일, 시, 분, 초 생성
        join_months = np.random.choice(range(1, 13), size=n_users)
        join_days = np.random.choice(
            range(1, 29), size=n_users
        )  # 28일까지로 제한(윤년/월 고려)
        join_hours = np.random.choice(range(0, 24), size=n_users)
        join_minutes = np.random.choice(range(0, 60), size=n_users)
        join_seconds = np.random.choice(range(0, 60), size=n_users)

        # join_date 컬럼 생성
        join_dates = [
            pd.Timestamp(year, month, day, hour, minute, second)
            for year, month, day, hour, minute, second in zip(
                join_years,
                join_months,
                join_days,
                join_hours,
                join_minutes,
                join_seconds,
            )
        ]

        # 지역
        regions = ["Seoul", "Busan", "Incheon", "Gyeonggi", "Daegu", "Other"]
        region_probs = [0.3, 0.1, 0.1, 0.3, 0.05, 0.15]
        user_regions = np.random.choice(regions, size=n_users, p=region_probs)

        # 유저 테이블 생성
        user_table = pd.DataFrame(
            {
                "user_id": np.NAN,
                "gender": genders,
                "age_group": ages,
                "join_date": join_dates,
                "region": user_regions,
            }
        )

        # join_date 기준으로 정렬 후 user_id 재할당
        user_table = user_table.sort_values("join_date").reset_index(drop=True)
        user_table["user_id"] = [f"user_{i}" for i in range(len(user_table))]

        # 저장
        user_table.to_csv(f"{save_dir}/user_table.csv", index=False)
        return user_table
    else:
        user_table = pd.read_csv(f"{save_dir}/user_table.csv")
        user_table["join_date"] = pd.to_datetime(user_table["join_date"])
        return user_table


def generate_event_log(user_table, save_dir):
    if os.path.exists(f"{save_dir}/event_log.csv"):
        event_log = pd.read_csv(f"{save_dir}/event_log.csv")
        event_log["event_time"] = pd.to_datetime(event_log["event_time"])
        return event_log
    else:
        # 캠페인 및 여정 단계 정의
        campaign_steps = {
            "insurance": [
                "event_participation",
                "click_diagnosis",
                "identity_verification",
                "click_consult",
            ],
            "card": ["card_select", "identity_verification", "join_event"],
            "loan": ["click_compare", "check_limit", "agree_terms"],
        }
        campaign_types = list(campaign_steps.keys())

        # 각 유저가 여러 서비스(캠페인)를 이용할 수 있도록 설정
        user_campaigns = []
        for user_id in user_table["user_id"]:
            # 1~3개 캠페인 무작위 선택
            n = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            selected = np.random.choice(campaign_types, size=n, replace=False)
            for campaign in selected:
                user_campaigns.append((user_id, campaign))

        # 이벤트 로그 생성
        event_logs = []
        final_conversion_rate = 0.05  # 최종 여정까지 가는 비율

        for user_id, campaign in user_campaigns:
            steps = campaign_steps[campaign]
            # 유저의 가입일 이후 무작위 시작 시각
            user_join = user_table.loc[
                user_table["user_id"] == user_id, "join_date"
            ].values[0]
            if isinstance(user_join, np.datetime64):
                user_join = pd.Timestamp(user_join).to_pydatetime()
            start_time = user_join + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            current_time = start_time
            reached_final = np.random.rand() < final_conversion_rate
            for idx, step in enumerate(steps):
                # 각 단계별로 1~10초 이내의 시간 증가
                delay = random.randint(1, 10)
                current_time += timedelta(seconds=delay)
                # 마지막 단계까지 도달할 확률 적용
                if idx == len(steps) - 1 and not reached_final:
                    break
                event_logs.append(
                    {
                        "user_id": user_id,
                        "campaign_type": campaign,
                        "event_name": step,
                        "event_time": current_time,
                    }
                )
                # 중간 단계에서 랜덤하게 이탈(최종 단계 제외)
                if idx < len(steps) - 1 and not reached_final:
                    if np.random.rand() > 0.7:  # 약 30% 확률로 이탈
                        break

        event_log = pd.DataFrame(event_logs)
        event_log = event_log.sort_values(["event_time"]).reset_index(drop=True)
        event_log.to_csv(f"{save_dir}/event_log.csv", index=False)
    return event_log


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    n_users = 3000  # 예시: 전체 유저 수
    save_dir = "data"  # 데이터 저장 디렉토리

    user_table = generate_user_table(n_users, save_dir)
    event_log = generate_event_log(user_table, save_dir)
    print(event_log.head())
