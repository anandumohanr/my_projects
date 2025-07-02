import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import altair as alt
import pytz
import time

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

@st.cache_data(ttl=14400, show_spinner=False)
def load_jira_data():
    jira_domain = st.secrets["JIRA_DOMAIN"]
    url = f"https://{jira_domain}/rest/api/3/search"
    auth = HTTPBasicAuth(st.secrets["JIRA_EMAIL"], st.secrets["JIRA_API_TOKEN"])
    headers = {"Accept": "application/json"}
    jql = f"filter={st.secrets['JIRA_FILTER_ID']}"
    params = {"jql": jql, "fields": "key,summary,status,duedate,customfield_10016,assignee", "maxResults": 1000}

    try:
        response = requests.get(url, headers=headers, auth=auth, params=params)
        response.raise_for_status()
        issues = response.json()["issues"]

        data = []
        for issue in issues:
            fields = issue["fields"]
            data.append({
                "Key": issue["key"],
                "Summary": fields.get("summary", ""),
                "Status": fields.get("status", {}).get("name", ""),
                "Due Date": fields.get("duedate"),
                "Story Points": fields.get("customfield_10016", 0),
                "Developer": fields.get("assignee", {}).get("displayName", "")
            })

        df = pd.DataFrame(data)
        df["Due Date"] = pd.to_datetime(df["Due Date"])
        df["Story Points"] = pd.to_numeric(df["Story Points"]).fillna(0)
        df["Status"] = df["Status"].fillna("")
        df["Week"] = df["Due Date"].dt.strftime("%Y-%W")
        df["Week Start"] = df["Due Date"].dt.to_period("W").dt.start_time
        df["Is Completed"] = df["Status"].str.upper().isin(COMPLETED_STATUSES)
        return df

    except Exception as e:
        st.error(f"Failed to fetch JIRA data: {e}")
        return pd.DataFrame()
    
def preprocess_data(df):
    df = df.copy()
    df["Due Date"] = pd.to_datetime(df["Due Date"], errors="coerce")
    df["Story Points"] = pd.to_numeric(df["Story Points"], errors="coerce").fillna(0)
    df["Status"] = df["Status"].fillna("")
    df["Week"] = df["Due Date"].dt.strftime("%Y-%W")
    df["Week Start"] = pd.NaT
    valid_dates = df["Due Date"].notna()
    df.loc[valid_dates, "Week Start"] = df.loc[valid_dates, "Due Date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["Is Completed"] = df["Status"].str.upper().isin(COMPLETED_STATUSES)
    return df

def get_week_options(df):
    today = datetime.today()
    week_map = df.dropna(subset=["Week", "Week Start"]).drop_duplicates(subset="Week")[["Week", "Week Start"]]
    week_map = week_map[week_map["Week Start"] <= today]
    return week_map.sort_values("Week Start", ascending=False).reset_index(drop=True)

def render_summary_tab(df, selected_week):
    st.subheader("Developer Productivity Summary")
    
    # Only completed tasks for selected week
    filtered_df = df[df["Week"] == selected_week]
    completed_df = filtered_df[filtered_df["Is Completed"]]

    # All in-progress tasks regardless of week
    in_progress_df_all = df[~df["Is Completed"]]

    developer_completed = completed_df.groupby("Developer")["Story Points"].sum().astype(int).to_dict()
    developer_inprogress = in_progress_df_all.groupby("Developer")["Story Points"].sum().astype(int).to_dict()

    all_developers = set(DEVELOPERS).union(developer_completed).union(developer_inprogress)
    data = []
    for dev in sorted(all_developers):
        completed = developer_completed.get(dev, 0)
        inprogress = developer_inprogress.get(dev, 0)
        productivity_percent = round(completed / 5 * 100, 1)
        productivity_display = f"{productivity_percent}%" if productivity_percent <= 100 else "100.0%+"
        data.append({
            "Developer": dev,
            "Completed Points": completed,
            "In Progress Points": inprogress,
            "Productivity % (out of 5 SP)": productivity_display
        })

    # Sort by productivity %
    summary_df = pd.DataFrame(data)
    summary_df["SortKey"] = summary_df["Productivity % (out of 5 SP)"].str.replace('%', '').str.replace('+', '').astype(float)
    summary_df = summary_df.sort_values("SortKey", ascending=False).drop(columns="SortKey")

    def highlight_low_productivity(val):
        try:
            pct = float(str(val).strip('%').replace('+', ''))
            return 'color: red' if pct < 80 else ''
        except:
            return ''

    styled_summary = summary_df.style.applymap(highlight_low_productivity, subset=["Productivity % (out of 5 SP)"])
    st.dataframe(styled_summary)

    total_possible = len(all_developers) * 5
    total_completed = summary_df["Completed Points"].sum()
    team_productivity = round((total_completed / total_possible) * 100, 1) if total_possible else 0.0

    st.markdown("### Active Developers with In-Progress Tasks")
    active_in_progress = summary_df[summary_df["In Progress Points"] > 0]
    
    if active_in_progress.empty:
        st.write("None")
    else:

        if active_in_progress.empty:
            st.write("None")
        else:
            if "expanded_dev" not in st.session_state:
                st.session_state.expanded_dev = None
        
            for _, row in active_in_progress.iterrows():
                dev = row["Developer"]
                dev_tasks = in_progress_df_all[in_progress_df_all["Developer"] == dev]
                due_dates = dev_tasks["Due Date"].dropna().dt.strftime("%d-%b-%Y").unique()
                due_str = ", ".join(sorted(due_dates))
        
                # Control expansion state
                expand_key = f"expander_{dev}"
                is_expanded = st.session_state.expanded_dev == dev
        
                with st.expander(f"🚧 {dev} - {row['In Progress Points']} SP | Due on: {due_str}", expanded=is_expanded):
                    if st.session_state.expanded_dev != dev:
                        st.session_state.expanded_dev = dev
        
                    if not dev_tasks.empty:
                        display_df = dev_tasks[["Key", "Summary", "Status", "Due Date", "Story Points"]].copy()
                        display_df["Key"] = display_df["Key"].apply(
                            lambda x: f"[{x}](https://impelsys.atlassian.net/browse/{x})" if isinstance(x, str) and not x.startswith("http") else f"[{x}]({x})"
                        )
                        display_df["Due Date"] = display_df["Due Date"].dt.strftime("%d-%b-%Y")
                        st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)
                    else:
                        st.write("No task details available.")

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
    today = datetime.today()
    all_weeks_df = df.dropna(subset=["Week", "Week Start"]).drop_duplicates(subset="Week")[["Week", "Week Start"]]
    all_weeks_df = all_weeks_df[all_weeks_df["Week Start"] <= today]
    all_weeks_df = all_weeks_df.sort_values("Week Start", ascending=False).reset_index(drop=True)
    all_weeks = all_weeks_df["Week"].tolist()
    recent_weeks = all_weeks[:4][::-1]

    dev_option = st.selectbox("Select Developer:", options=sorted(set(DEVELOPERS)))
    df_dev = df[df["Developer"] == dev_option]
    df_dev_completed = df_dev[df_dev["Is Completed"]]
    weekly_dev = df_dev_completed.groupby("Week")["Story Points"].sum().astype(int).reindex(recent_weeks, fill_value=0).reset_index()
    weekly_dev.columns = ["Week", "Story Points"]

    if not weekly_dev.empty:
        weekly_dev["Delta"] = weekly_dev["Story Points"].diff().fillna(0)
        weekly_dev["Color"] = weekly_dev["Delta"].apply(lambda x: "green" if x > 0 else ("red" if x < 0 else "gray"))
        weekly_dev["Change"] = weekly_dev["Delta"].apply(lambda x: "⬆️" if x > 0 else ("⬇️" if x < 0 else "➖"))

        dev_chart = alt.Chart(weekly_dev).mark_line(point=True).encode(
            x=alt.X("Week:N", title="Week"),
            y=alt.Y("Story Points:Q", title="Story Points"),
            color=alt.Color("Color:N", scale=None),
            tooltip=["Week", "Story Points", "Change"]
        ).properties(title=f"{dev_option} Productivity (Last 4 Weeks)", height=250)
        st.altair_chart(dev_chart, use_container_width=True)

        st.markdown("#### Tabular View")
        st.dataframe(weekly_dev[["Week", "Story Points", "Change"]])
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

    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist).strftime('%Y-%m-%d %I:%M %p %Z')
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.success("✅ JIRA data refreshed successfully")
            time.sleep(1)
            st.rerun()
    with col2:
        st.caption(f"Last data refresh: {now_ist}")

    with st.spinner("Fetching and processing data from SharePoint..."):
        df = load_jira_data()
        if df.empty:
            st.stop()
        df = preprocess_data(df)

    week_options_df = get_week_options(df)
    week_label_map = {
        row["Week"]: f"{row['Week']} ({row['Week Start'].strftime('%d-%B-%Y').upper()} to {(row['Week Start'] + timedelta(days=4)).strftime('%d-%B-%Y').upper()})"
        for _, row in week_options_df.iterrows()
    }
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

if __name__ == "__main__":
    main()
