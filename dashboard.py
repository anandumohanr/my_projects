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
    "Anandu Mohan", "Ravi Kumar", "shree.vidya", "Brijesh Kanchan",
    "Hari Prasad H S", "Fahad P K", "Venukumar DL", "Kishore C", "padmaja"
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

    df_dev = df[df["Developer"].isin(all_developers) & df["Is Completed"]]
    recent_weeks = sorted(df_dev["Week"].dropna().unique())[-4:]

    trend_data = pd.DataFrame()
    most_improved = None
    largest_drop = None
    consistent_performer = None

    if len(recent_weeks) >= 2:
        trend_data = df_dev[df_dev["Week"].isin(recent_weeks)].groupby(["Developer", "Week"])["Story Points"].sum().unstack(fill_value=0)
        trend_data = trend_data.reindex(columns=recent_weeks, fill_value=0)
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
    st.subheader("Developer-wise 4-Week Trend")
    recent_weeks = sorted(df["Week"].dropna().unique())[-4:]
    df_trend = df[df["Week"].isin(recent_weeks) & df["Is Completed"]]
    dev_selection = st.selectbox("Select developer to view trend:", sorted(set(DEVELOPERS).union(df_trend["Developer"].dropna().astype(str))))
    df_dev = df_trend[df_trend["Developer"] == dev_selection]
    weekly_points = df_dev.groupby("Week")["Story Points"].sum().reindex(recent_weeks, fill_value=0).reset_index()

    base = alt.Chart(weekly_points).mark_line(point=True).encode(
        x=alt.X("Week", sort=recent_weeks),
        y="Story Points"
    ).properties(width=600, height=300)
    st.altair_chart(base, use_container_width=False)

    weekly_points["Trend"] = weekly_points["Story Points"].diff().fillna(0)
    def trend_marker(val):
        if val > 0:
            return f"⬆️ {val}"
        elif val < 0:
            return f"⬇️ {abs(val)}"
        return "➖"
    weekly_points["Trend"] = weekly_points["Trend"].apply(trend_marker)
    st.dataframe(weekly_points.set_index("Week"))

def render_tasks_tab(df):
    st.subheader("Filtered Task Table")
    weeks = sorted(df["Week"].unique())[::-1]
    selected_weeks = st.multiselect("Select weeks", options=weeks, default=weeks[:1])
    developers = sorted(set(df["Developer"]).union(DEVELOPERS))
    selected_devs = st.multiselect("Select developers", options=developers, default=developers)

    filtered_df = df[df["Week"].isin(selected_weeks) & df["Developer"].isin(selected_devs)]
    filtered_df = filtered_df[["Developer", "Title", "Status", "Story Points", "Due Date", "Week"]]
    st.dataframe(filtered_df)

def render_export_tab(df):
    st.subheader("Export Data")
    st.download_button("Download Full Task List", data=df.to_csv(index=False), file_name="full_tasks.csv", mime="text/csv")

def main():
    st.set_page_config(page_title="Team Productivity Dashboard", layout="wide")
    st.title("📊 Development Team Productivity Dashboard")

    df = load_excel()
    if df.empty:
        st.warning("No data available to display.")
        return

    df = preprocess_data(df)
    week_options = get_week_options(df)

    if week_options.empty:
        st.warning("No valid week data found.")
        return

    week_labels = [f"{row['Week']} ({row['Week Start'].strftime('%b %d')})" for _, row in week_options.iterrows()]
    selected = st.selectbox("Select week to view:", options=week_labels)

    if selected:
        selected_week = selected.split()[0]
        tabs = st.tabs(["Summary", "Trends", "Tasks", "Export"])

        with tabs[0]:
            render_summary_tab(df, selected_week)
        with tabs[1]:
            render_trend_tab(df)
        with tabs[2]:
            render_tasks_tab(df)
        with tabs[3]:
            render_export_tab(df)

if __name__ == "__main__":
    main()
