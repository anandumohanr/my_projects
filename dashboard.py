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
        color = "red" if productivity_percent < 80 else "black"
        data.append({"Developer": dev, "Completed Points": points, "Productivity % (out of 5 SP)": productivity_display, "_style": f"color: {color};"})

    summary_df = pd.DataFrame(data)
    styled_summary = summary_df.style.apply(
        lambda row: ["color: red" if row["Productivity % (out of 5 SP)"] != "100.0%+" and float(row["Productivity % (out of 5 SP)"].replace("%", "")) < 80 else ""]
        * len(row),
        axis=1
    )
    st.dataframe(styled_summary.drop(columns=["_style"]))

    total_possible = len(all_developers) * 5
    total_completed = summary_df["Completed Points"].sum()
    team_productivity = round((total_completed / total_possible) * 100, 1) if total_possible else 0.0

    top_3 = summary_df.sort_values("Completed Points", ascending=False).head(3)
    zero_productivity = summary_df[summary_df["Completed Points"] == 0]

    st.markdown("### Insights")
    st.markdown("**Top 3 Developers**")
    st.dataframe(top_3)
    st.markdown("**Developers with 0 Productivity**")
    if zero_productivity.empty:
        st.write("None")
    else:
        st.dataframe(zero_productivity)

    st.subheader("Team Overview")
    team_summary = pd.DataFrame({
        "Week": [selected_week],
        "Story Points": [total_completed],
        "Team Productivity %": [f"{team_productivity}%"]
    })
    st.dataframe(team_summary)

    st.markdown("### Team Productivity Chart")
    chart_data = pd.DataFrame({"Week": [selected_week], "Story Points": [total_completed]})
    chart = alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X("Week:N", title="Week"),
        y=alt.Y("Story Points:Q", title="Total Story Points"),
        tooltip=["Week", "Story Points"]
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)

    return summary_df, team_summary

def render_trend_tab(df):
    st.markdown("### Developer Productivity Trend (Last 4 Weeks)")
    all_weeks_df = df.dropna(subset=["Week", "Week Start"]).drop_duplicates(subset="Week")[["Week", "Week Start"]]
    all_weeks_df = all_weeks_df.sort_values("Week Start").reset_index(drop=True)
    all_weeks = all_weeks_df["Week"].tolist()
    recent_weeks = all_weeks[-4:]

    dev_option = st.selectbox("Select Developer:", options=sorted(set(DEVELOPERS)))
    df_dev = df[df["Developer"] == dev_option]
    df_dev_completed = df_dev[df_dev["Is Completed"]]
    weekly_dev = df_dev_completed.groupby("Week")["Story Points"].sum().reindex(recent_weeks, fill_value=0).reset_index()
    weekly_dev.columns = ["Week", "Story Points"]

    if not weekly_dev.empty:
        weekly_dev["Delta"] = weekly_dev["Story Points"].diff().fillna(0)
        weekly_dev["Color"] = weekly_dev["Delta"].apply(lambda x: "green" if x > 0 else ("red" if x < 0 else "gray"))

        dev_chart = alt.Chart(weekly_dev).mark_line(point=True).encode(
            x=alt.X("Week:N", title="Week"),
            y=alt.Y("Story Points:Q", title="Story Points"),
            color=alt.Color("Color:N", scale=None),
            tooltip=["Week", "Story Points", "Delta"]
        ).properties(title=f"{dev_option} Productivity (Last 4 Weeks)", height=250)
        st.altair_chart(dev_chart, use_container_width=True)

        st.markdown("#### Tabular View")
        st.dataframe(weekly_dev[["Week", "Story Points", "Delta"]])
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
