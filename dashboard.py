import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import altair as alt

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

def render_summary_tab(df, selected_week):
    st.subheader("Developer Productivity Summary")
    filtered_df = df[df["Week"] == selected_week]
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

    total_possible = len(all_developers) * 5
    total_completed = summary_df["Completed Points"].sum()
    team_productivity = round((total_completed / total_possible) * 100, 1) if total_possible else 0.0

    top_3 = summary_df.sort_values("Completed Points", ascending=False).head(3)
    zero_productivity = summary_df[summary_df["Completed Points"] == 0]

    most_improved = None
    largest_drop = None
    consistent_performer = None

    df_dev = df[df["Developer"].isin(all_developers) & df["Is Completed"]]
    recent_weeks = sorted(df_dev["Week"].unique())[-4:]

    if recent_weeks:
        trend_data = df_dev[df_dev["Week"].isin(recent_weeks)].groupby(["Developer", "Week"])["Story Points"].sum().unstack(fill_value=0)
        trend_data["Delta"] = trend_data[recent_weeks[-1]] - trend_data[recent_weeks[-2]]
        if not trend_data.empty:
            most_improved = trend_data["Delta"].idxmax(), trend_data["Delta"].max()
            largest_drop = trend_data["Delta"].idxmin(), trend_data["Delta"].min()
            consistent_mask = (trend_data[recent_weeks] > 3).all(axis=1)
            consistent = trend_data[consistent_mask].index.tolist()
            consistent_performer = consistent[0] if consistent else None

    st.markdown("### Insights")
    st.markdown("**Top 3 Developers**")
    st.dataframe(top_3)
    st.markdown("**Developers with 0 Productivity**")
    st.write("None" if zero_productivity.empty else zero_productivity)

    st.markdown("**Most Improved Developer**")
    if most_improved:
        st.write(f"⬆️ {most_improved[0]} (+{most_improved[1]} SP)")
    else:
        st.write("N/A")

    st.markdown("**Largest Drop in Productivity**")
    if largest_drop:
        st.write(f"⬇️ {largest_drop[0]} ({largest_drop[1]} SP)")
    else:
        st.write("N/A")

    st.markdown("**Consistent Performer (4 Weeks > 3 SP)**")
    if consistent_performer:
        st.write(f"🏆 {consistent_performer}")
    else:
        st.write("N/A")

    st.subheader("Team Overview")
    team_summary = pd.DataFrame({
        "Week": [selected_week],
        "Story Points": [total_completed],
        "Team Productivity %": [f"{team_productivity}%"]
    })
    st.dataframe(team_summary)

    return summary_df, team_summary

def render_trend_tab(df):
    st.markdown("### Developer Productivity Trend (Last 4 Weeks)")
    all_weeks_df = df.dropna(subset=["Week", "Week Start"]).drop_duplicates(subset="Week")[["Week", "Week Start"]]
    all_weeks_df = all_weeks_df.sort_values("Week Start").reset_index(drop=True)
    all_weeks = all_weeks_df["Week"].tolist()

    dev_option = st.selectbox("Select Developer:", options=sorted(set(DEVELOPERS)))
    df_dev = df[df["Developer"] == dev_option]
    df_dev_completed = df_dev[df_dev["Is Completed"]]
    weekly_dev = df_dev_completed.groupby("Week")["Story Points"].sum().reindex(all_weeks, fill_value=0).reset_index()
    weekly_dev.columns = ["Week", "Story Points"]

    if not weekly_dev.empty:
        weekly_dev["Delta"] = weekly_dev["Story Points"].diff().fillna(0)
        weekly_dev["Trend"] = weekly_dev["Delta"].apply(lambda x: "⬆️" if x > 0 else ("⬇️" if x < 0 else "➖"))
        weekly_dev["Color"] = weekly_dev["Delta"].apply(lambda x: "green" if x > 0 else ("red" if x < 0 else "gray"))

        chart = alt.Chart(weekly_dev).mark_line(point=True).encode(
            x=alt.X("Week:N", title="Week"),
            y=alt.Y("Story Points:Q", title="Story Points"),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Week", "Story Points", "Trend"]
        ).properties(title=f"{dev_option} Productivity (Last 4 Weeks)", height=250)
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Tabular View")
        st.dataframe(weekly_dev[["Week", "Story Points", "Trend"]])
    else:
        st.info("No completed tasks found for selected developer.")

def render_tasks_tab(df, selected_week):
    st.subheader("Detailed Task Table")
    filtered_df = df[df["Week"] == selected_week]
    st.dataframe(filtered_df[["Key", "Summary", "Developer", "Status", "Due Date", "Story Points", "Week"]])

def render_export_tab(team_summary):
    st.subheader("Export Summary")
    st.download_button("Download Summary CSV", data=team_summary.to_csv(index=False), file_name="team_productivity.csv")

def main():
    st.set_page_config("Productivity Dashboard", layout="wide")
    st.title("📊 Weekly Productivity Dashboard")

    with st.spinner("Fetching and processing data from SharePoint..."):
        df = load_excel()
        if df.empty:
            st.stop()
        df = preprocess_data(df)

    week_options_df = get_week_options(df)
    week_label_map = {row["Week"]: f"{row['Week']} ({row['Week Start'].date()} to {(row['Week Start'] + timedelta(days=4)).date()})" for _, row in week_options_df.iterrows()}
    selected_week = st.selectbox("Select week to view:", options=list(week_label_map.keys()), format_func=lambda x: week_label_map[x])

    tabs = st.tabs(["Summary", "Trends", "Tasks", "Export"])

    with tabs[0]:
        summary_df, team_summary = render_summary_tab(df, selected_week)

    with tabs[1]:
        render_trend_tab(df)

    with tabs[2]:
        render_tasks_tab(df, selected_week)

    with tabs[3]:
        render_export_tab(team_summary)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.caption(f"Last data refresh: {datetime.now().astimezone().strftime('%Y-%m-%d %I:%M %p %Z')}")

if __name__ == "__main__":
    main()
