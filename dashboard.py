import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Constants
SHAREPOINT_URL = "https://impelsysinc-my.sharepoint.com/:x:/g/personal/anandu_m_medlern_com/EXxi7DECTpxDgA-Hx44P-G8B-PgU74kHUVKlz3VfbTNX5w?download=1"
COMPLETED_STATUSES = ["ACCEPTED IN QA", "CLOSED"]
DEVELOPERS = [
    "Anandu Mohan",
    "Ravi Kumar",
    "shree.vidya",
    "Brijesh Kanchan",
    "Hari Prasad H S",
    "Fahad P K",
    "Venukumar DL",
    "Kishore C",
    "padmaja"
]

@st.cache_data(show_spinner=False)
def load_excel():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(SHAREPOINT_URL, headers=headers)
        r.raise_for_status()

        if b"<html" in r.content[:100].lower():
            raise ValueError("Downloaded file is not a valid Excel document.")

        return pd.read_excel(BytesIO(r.content), engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to load data from SharePoint: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    df = df.copy()
    df["Due Date"] = pd.to_datetime(df["Due Date"], errors="coerce")
    df["Story Points"] = pd.to_numeric(df["Story Points"], errors="coerce").fillna(0)
    df["Status"] = df["Status"].fillna("")
    df["Week"] = df["Due Date"].dt.strftime("%Y-%W")
    df["Week Start"] = df["Due Date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["Is Completed"] = df["Status"].str.upper().isin(COMPLETED_STATUSES)
    return df

def get_week_options(df):
    week_map = df.dropna(subset=["Week", "Week Start"]).drop_duplicates(subset="Week")[["Week", "Week Start"]]
    return week_map.sort_values("Week Start").reset_index(drop=True)

def main():
    st.set_page_config("Productivity Dashboard", layout="wide")
    st.title("\U0001F4CA Weekly Productivity Dashboard")

    with st.status("Fetching and processing data from SharePoint..."):
        df = load_excel()
        if df.empty:
            st.stop()
        df = preprocess_data(df)

    week_options_df = get_week_options(df)
    week_label_map = {row["Week"]: f"{row['Week']} ({row['Week Start'].date()} to {(row['Week Start'] + timedelta(days=4)).date()})" for _, row in week_options_df.iterrows()}
    selected_week = st.selectbox("Select week to view:", options=list(week_label_map.keys()), format_func=lambda x: week_label_map[x])

    filtered_df = df[df["Week"] == selected_week]

    st.subheader("Developer Productivity Summary")
    completed_df = filtered_df[filtered_df["Is Completed"]]
    developer_points = completed_df.groupby("Developer")["Story Points"].sum().to_dict()

    all_developers = set(DEVELOPERS).union(set(developer_points.keys()))
    data = []
    for dev in sorted(all_developers):
        points = developer_points.get(dev, 0)
        productivity_percent = round(points / 5 * 100, 1)
        productivity_display = f"{productivity_percent}%" if productivity_percent <= 100 else "100.0%+"
        data.append({"Developer": dev, "Completed Points": points, "Productivity % (out of 5 SP)": productivity_display})

    summary_df = pd.DataFrame(data)
    st.dataframe(summary_df)

    # Team Productivity Summary
    total_possible = len(all_developers) * 5
    total_completed = summary_df["Completed Points"].sum()
    team_productivity = round((total_completed / total_possible) * 100, 1) if total_possible else 0.0

    st.subheader("Developer-wise Completed Points")
    fig, ax = plt.subplots(figsize=(8, 4))
    summary_df.set_index("Developer")["Completed Points"].plot(kind="line", marker='o', ax=ax)
    ax.set_ylabel("Story Points")
    ax.set_title("Completed Story Points by Developer")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Team Overview")
    team_summary = pd.DataFrame({
        "Week": [selected_week],
        "Story Points": [total_completed],
        "Team Productivity %": [f"{team_productivity}%"]
    })
    st.dataframe(team_summary)

    st.subheader("Detailed Task Table")
    st.dataframe(filtered_df[["Key", "Summary", "Developer", "Status", "Due Date", "Story Points", "Week"]])

    st.download_button("Download Summary CSV", data=team_summary.to_csv(index=False), file_name="team_productivity.csv")

if __name__ == "__main__":
    main()
