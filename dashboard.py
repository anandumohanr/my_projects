import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import altair as alt
import pytz
import time
from requests.auth import HTTPBasicAuth
import openai
import re
import numpy as np
from dateutil.relativedelta import relativedelta

# Constants
COMPLETED_STATUSES = ["ACCEPTED IN QA", "CLOSED"]

@st.cache_data(ttl=14400, show_spinner=False)
def load_jira_data():
    jira_domain = st.secrets["JIRA_DOMAIN"]
    url = f"https://{jira_domain}/rest/api/3/search"
    auth = HTTPBasicAuth(st.secrets["JIRA_EMAIL"], st.secrets["JIRA_API_TOKEN"])
    headers = {"Accept": "application/json"}
    jql = f"filter={st.secrets['JIRA_FILTER_ID']}"
    params = {"jql": jql, "fields": "key,summary,status,customfield_11020,customfield_10010,customfield_11012", "maxResults": 1000}

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
                "Due Date": fields.get("customfield_11020"),
                "Story Points": fields.get("customfield_10010", 0),
                "Developer": fields.get("customfield_11012", {}).get("displayName", "")
            })

        df = pd.DataFrame(data)
        df["Due Date"] = pd.to_datetime(df["Due Date"])
        df["Story Points"] = pd.to_numeric(df["Story Points"]).fillna(0)
        df["Status"] = df["Status"].fillna("")
        df["Week"] = df["Due Date"].dt.strftime("%Y-%W")
        df["Week Start"] = df["Due Date"].dt.to_period("W").dt.start_time
        df["Month"] = df["Due Date"].dt.to_period("M").dt.to_timestamp()
        df["Quarter"] = df["Due Date"].dt.to_period("Q").dt.to_timestamp()
        df["Year"] = df["Due Date"].dt.year
        df["Is Completed"] = df["Status"].str.upper().isin(COMPLETED_STATUSES)
        return df
    except Exception as e:
        st.error(f"Failed to fetch JIRA data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=14400, show_spinner=False)
def load_bug_data():
    jira_domain = st.secrets["JIRA_DOMAIN"]
    url = f"https://{jira_domain}/rest/api/3/search"
    auth = HTTPBasicAuth(st.secrets["JIRA_EMAIL"], st.secrets["JIRA_API_TOKEN"])
    headers = {"Accept": "application/json"}
    jql = f"filter=18484"
    params = {"jql": jql, "fields": "key,summary,created,customfield_11012", "maxResults": 1000}

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
                "Created": fields.get("created"),
                "Developer": fields.get("customfield_11012", {}).get("displayName", "")
            })
        df = pd.DataFrame(data)
        df["Created"] = pd.to_datetime(df["Created"])
        df["Week"] = df["Created"].dt.strftime("%Y-%W")
        df["Week Start"] = df["Created"].dt.to_period("W").dt.start_time
        df["Month"] = df["Created"].dt.to_period("M").dt.to_timestamp()
        df["Quarter"] = df["Created"].dt.to_period("Q").dt.to_timestamp()
        df["Year"] = df["Created"].dt.year
        return df
    except Exception as e:
        st.error(f"Failed to fetch Bug data: {e}")
        return pd.DataFrame()

# 🆕 Utility function to format period display
@st.cache_data(show_spinner=False)
def format_period_display(value, period_type):
    if period_type == "Month":
        return value.strftime("%b-%Y").upper()  # Example: JUL-2025
    if period_type == "Quarter":
        quarter = (value.month - 1) // 3 + 1
        return f"Q{quarter}-{value.year}"
    if period_type == "Year":
        return str(value)
    return value

# 🆕 Format period scope string
@st.cache_data(show_spinner=False)
def format_period_scope(row, period):
    if period == "Week":
        start = row.to_period("W").start_time.date()
        end = row.to_period("W").end_time.date()
    elif period == "Month":
        start = row.to_period("M").start_time.date()
        end = row.to_period("M").end_time.date()
    elif period == "Quarter":
        start = row.to_period("Q").start_time.date()
        end = row.to_period("Q").end_time.date()
    elif period == "Year":
        start = datetime(row, 1, 1).date()
        end = datetime(row, 12, 31).date()
    else:
        return ""
    return f"{format_period_display(row, period)} ({start.strftime('%d-%b-%Y').upper()} to {end.strftime('%d-%b-%Y').upper()})"

def get_week_label(week_str):
    year, week = map(int, week_str.split('-'))
    week_start = datetime.strptime(f'{year}-W{int(week)}-1', '%Y-W%W-%w').date()
    week_end = week_start + timedelta(days=6)
    return f"{week_str} ({week_start.strftime('%d-%b-%Y').upper()} to {week_end.strftime('%d-%b-%Y').upper()})"

def prepare_trend_data_fixed(df, period, developer=None):
    df = df[df["Is Completed"]].copy()
    devs = sorted(df["Developer"].dropna().unique()) if developer is None else [developer]

    if period == "Week":
        group_col = "Week"
        date_col = "Week Start"
    elif period == "Month":
        group_col = "Month"
        date_col = "Month"
    elif period == "Quarter":
        group_col = "Quarter"
        date_col = "Quarter"
    else:
        group_col = "Year"
        date_col = "Year"

    # Ensure all (period, developer) pairs exist
    all_periods = sorted(df[group_col].dropna().unique())
    index = pd.MultiIndex.from_product([all_periods, devs], names=[group_col, "Developer"])
    trend_df = df.groupby([group_col, "Developer"])["Story Points"].sum().reindex(index, fill_value=0).reset_index()

    # Add full calendar-based range labels
    if period == "Week":
        trend_df["Label"] = trend_df[group_col]
        trend_df["Scope"] = trend_df[group_col].apply(get_week_label)
    elif period == "Month":
        trend_df["Label"] = trend_df[group_col].dt.strftime('%b-%Y').str.upper()
        trend_df["Scope"] = trend_df[group_col].apply(lambda d: f"{d.strftime('%b-%Y').upper()} ({d.strftime('%d-%b-%Y').upper()} to {(d + pd.offsets.MonthEnd(0)).strftime('%d-%b-%Y').upper()})")
    elif period == "Quarter":
        trend_df["Label"] = trend_df[group_col].apply(lambda d: f"Q{((d.month-1)//3)+1}-{d.year}")
        trend_df["Scope"] = trend_df[group_col].apply(lambda d: f"Q{((d.month-1)//3)+1}-{d.year} ({d.strftime('%d-%b-%Y').upper()} to {(d + pd.DateOffset(months=3) - pd.Timedelta(days=1)).strftime('%d-%b-%Y').upper()})")
    else:
        trend_df["Label"] = trend_df[group_col].astype(str)
        trend_df["Scope"] = trend_df[group_col].apply(lambda y: f"{y} (01-JAN-{y} to 31-DEC-{y})")

    return trend_df

def render_summary_tab(df, selected_week):
    st.subheader("Developer Productivity Summary")
    filtered_df = df[df["Week"] == selected_week]
    completed_df = filtered_df[filtered_df["Is Completed"]]
    in_progress_df_all = df[~df["Is Completed"]]

    developer_completed = completed_df.groupby("Developer")["Story Points"].sum().astype(int).to_dict()
    developer_inprogress = in_progress_df_all.groupby("Developer")["Story Points"].sum().astype(int).to_dict()
    all_developers = sorted(df["Developer"].dropna().unique())
    data = []
    for dev in sorted(all_developers):
        completed = developer_completed.get(dev, 0)
        inprogress = developer_inprogress.get(dev, 0)
        productivity_percent = round(completed / 5 * 100, 1)
        data.append({
            "Developer": dev,
            "Completed Points": completed,
            "In Progress Points": inprogress,
            "Productivity % (out of 5 SP)": f"{productivity_percent}%"
        })

    summary_df = pd.DataFrame(data)
    summary_df["SortKey"] = summary_df["Productivity % (out of 5 SP)"].str.replace('%', '').astype(float)
    summary_df = summary_df.sort_values("SortKey", ascending=False).drop(columns="SortKey")

    def highlight_low_productivity(val):
        try:
            pct = float(str(val).strip('%'))
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
        if "expanded_dev" not in st.session_state:
            st.session_state.expanded_dev = None

        for _, row in active_in_progress.iterrows():
            dev = row["Developer"]
            dev_tasks = in_progress_df_all[in_progress_df_all["Developer"] == dev]
            due_dates = dev_tasks["Due Date"].dropna().dt.strftime("%d-%b-%Y").unique()
            due_str = ", ".join(sorted(due_dates))
            is_expanded = st.session_state.expanded_dev == dev
            with st.expander(f"🚧 {dev} - {row['In Progress Points']} SP | Due on: {due_str}", expanded=is_expanded):
                if st.session_state.expanded_dev != dev:
                    st.session_state.expanded_dev = dev
                if not dev_tasks.empty:
                    display_df = dev_tasks[["Key", "Summary", "Status", "Due Date", "Story Points"]].copy()
                    display_df["Key"] = display_df["Key"].apply(
                        lambda x: f"[{x}](https://impelsys.atlassian.net/browse/{x})" if isinstance(x, str) else x)
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
    st.subheader("📈 Developer Productivity Trends")
    period = st.selectbox("View By", ["Week", "Month", "Quarter", "Year"], key="trend_period")
    developer = st.selectbox("Developer", sorted(df["Developer"].dropna().unique()), key="trend_dev")
    df_dev = df[(df["Developer"] == developer) & (df["Is Completed"])]

    if period == "Week":
        group_col = "Week"
    elif period == "Month":
        group_col = "Month"
    elif period == "Quarter":
        group_col = "Quarter"
    else:
        group_col = "Year"

    trend_data = prepare_trend_data_fixed(df, period, developer)

    if trend_data.empty:
        st.info("No data available.")
        return

    def format_scope(row):
        if period == "Week":
            start = df_dev[df_dev[group_col] == row]["Due Date"].min().date()
            end = df_dev[df_dev[group_col] == row]["Due Date"].max().date()
            return f"{row} ({start.strftime('%d-%b-%Y').upper()} to {end.strftime('%d-%b-%Y').upper()})"
        elif period == "Month":
            start = row.to_period("M").start_time.date()
            end = row.to_period("M").end_time.date()
            return f"{row.strftime('%b-%Y').upper()} ({start.strftime('%d-%b-%Y').upper()} to {end.strftime('%d-%b-%Y').upper()})"
        elif period == "Quarter":
            start = row.to_period("Q").start_time.date()
            end = row.to_period("Q").end_time.date()
            quarter = f"Q{((row.month - 1) // 3) + 1}-{row.year}"
            return f"{quarter} ({start.strftime('%d-%b-%Y').upper()} to {end.strftime('%d-%b-%Y').upper()})"
        elif period == "Year":
            start = datetime(row, 1, 1).date()
            end = datetime(row, 12, 31).date()
            return f"{row} ({start.strftime('%d-%b-%Y').upper()} to {end.strftime('%d-%b-%Y').upper()})"

    trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
        x=alt.X("Label:N", title=period),
        y=alt.Y("Story Points", title="Completed Story Points"),
        tooltip=["Scope", "Story Points"]
    ).properties(height=300)

    st.altair_chart(trend_chart, use_container_width=True)
    st.dataframe(trend_data[["Scope", "Story Points"]].rename(columns={"Scope": period}))

def render_team_trend(df):
    st.subheader("👥 Team Trend")
    period = st.selectbox("Team View By", ["Week", "Month", "Quarter", "Year"], key="team_period")
    df_team = df[df["Is Completed"]]
    if df_team.empty:
        st.info("No story point data available.")
        return

    if period == "Week":
        group_col = "Week"
        formatter = lambda x: (
            f"{x} ({df_team[df_team['Week'] == x]['Due Date'].min().strftime('%d-%b-%Y').upper()} to "
            f"{df_team[df_team['Week'] == x]['Due Date'].max().strftime('%d-%b-%Y').upper()})"
        )
        labeler = lambda x: str(x)
    elif period == "Month":
        group_col = "Month"
        formatter = lambda x: (
            f"{x.strftime('%b-%Y').upper()} ({x.to_period('M').start_time.strftime('%d-%b-%Y').upper()} to {x.to_period('M').end_time.strftime('%d-%b-%Y').upper()})"
        )
        labeler = lambda x: x.strftime('%b-%Y').upper()
    elif period == "Quarter":
        group_col = "Quarter"
        formatter = lambda x: (
            f"Q{((x.month - 1) // 3) + 1}-{x.year} ({x.to_period('Q').start_time.strftime('%d-%b-%Y').upper()} to {x.to_period('Q').end_time.strftime('%d-%b-%Y').upper()})"
        )
        labeler = lambda x: f"Q{((x.month - 1) // 3) + 1}-{x.year}"
    else:
        group_col = "Year"
        formatter = lambda x: f"{x} (01-JAN-{x} to 31-DEC-{x})"
        labeler = lambda x: str(x)

    grouped_chart = prepare_trend_data_fixed(df, period)
    grouped_chart["Period"] = grouped_chart["Label"]

    grouped_table = grouped_chart.groupby("Scope")["Story Points"].sum().reset_index().rename(columns={"Scope": "Period"})

    chart = alt.Chart(grouped_chart).mark_bar().encode(
        x=alt.X("Period:N", title=period),
        y=alt.Y("Story Points:Q", title="Team Story Points"),
        color="Developer:N"
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(grouped_table[["Period", "Story Points"]].rename(columns={"Period": period}))

def render_quality_tab(bugs_df):
    st.subheader("🐞 Bug and Quality Metrics")
    period = st.selectbox("View Bugs By", ["Week", "Month", "Quarter", "Year"], key="bug_period")
    bugs = bugs_df.copy()
    bugs["Bugs"] = 1

    # Create all combinations to avoid missing data
    unique_periods = sorted(bugs[period].dropna().unique())
    unique_devs = sorted(bugs["Developer"].dropna().unique())
    full_index = pd.MultiIndex.from_product([unique_periods, unique_devs], names=[period, "Developer"])
    grouped = bugs.groupby([period, "Developer"])["Bugs"].sum().reindex(full_index, fill_value=0).reset_index()

    # Sort developers by total bugs (descending)
    dev_order = grouped.groupby("Developer")["Bugs"].sum().sort_values(ascending=False).index.tolist()
    grouped["Developer"] = pd.Categorical(grouped["Developer"], categories=dev_order, ordered=True)

    # Format period for display
    def format_period(x):
        if period == "Week":
            return str(x)
        elif period == "Month":
            return x.strftime('%b-%Y').upper()
        elif period == "Quarter":
            return f"Q{((x.month - 1) // 3) + 1}-{x.year}"
        elif period == "Year":
            return str(x)

    grouped["Formatted Period"] = grouped[period].apply(format_period)
    grouped = grouped.drop(columns=[period])  # Avoid column duplication

    # Chart
    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X("Formatted Period:N", title=period),
        y=alt.Y("Bugs", title="Bug Count"),
        color=alt.Color("Developer:N", sort=dev_order)
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

    # Option A: Pivot Table Developer vs Period
    pivot_df = grouped.pivot_table(index="Developer", columns="Formatted Period", values="Bugs", fill_value=0)
    pivot_df["Total Bugs"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values("Total Bugs", ascending=False)
    st.markdown("#### 📊 Developer-wise Bug Summary")
    st.dataframe(pivot_df)

    # Option B: Sorted Developer Summary
    buggy_period_df = grouped.sort_values(["Developer", "Bugs"], ascending=[True, False]).drop_duplicates("Developer")

    summary_df = grouped.groupby("Developer").agg(
        Total_Bugs=("Bugs", "sum"),
        Max_Bugs_in_One_Period=("Bugs", "max")
    ).reset_index()

    summary_df = summary_df.merge(
        buggy_period_df[["Developer", "Formatted Period"]],
        on="Developer",
        how="left"
    ).rename(columns={"Formatted Period": "Most_Buggy_Period"})

    summary_df = summary_df.sort_values("Total_Bugs", ascending=False)

    st.markdown("#### 🧾 Developer Bug Summary")
    st.dataframe(summary_df)

def render_insights_tab(df, bugs_df):
    import streamlit as st
    import altair as alt
    from datetime import datetime, timedelta

    st.header("📊 Developer Insights (Productivity & Quality)")

    def count_working_days(start_date, end_date):
        return sum(1 for d in pd.date_range(start=start_date, end=end_date) if d.weekday() < 5)

    # --- Date Filter ---
    today = datetime.today().date()
    default_start = today - timedelta(weeks=4)
    min_date = df["Week Start"].min().date()
    max_date = df["Week Start"].max().date()

    filter_option = st.selectbox(
        "Select Insight Duration",
        ["Last 4 Weeks", "Last 3 Months", "Current Quarter", "Previous Quarter", "Custom Range"],
        index=0
    )

    if filter_option == "Last 4 Weeks":
        start_date = today - timedelta(weeks=4)
        end_date = today
    elif filter_option == "Last 3 Months":
        start_date = today - timedelta(days=90)
        end_date = today
    elif filter_option == "Current Quarter":
        current_month = today.month
        quarter_start = 3 * ((current_month - 1) // 3) + 1
        start_date = datetime(today.year, quarter_start, 1).date()
        end_date = today
    elif filter_option == "Previous Quarter":
        current_month = today.month
        quarter_start = 3 * ((current_month - 1) // 3) + 1
        prev_q_end = datetime(today.year, quarter_start, 1) - timedelta(days=1)
        quarter_start = 3 * ((prev_q_end.month - 1) // 3) + 1
        start_date = datetime(prev_q_end.year, quarter_start, 1).date()
        end_date = prev_q_end.date()
    else:
        start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=today, min_value=start_date, max_value=max_date)

    # --- Filter Data ---
    df_filtered = df[(df["Week Start"].dt.date >= start_date) & (df["Week Start"].dt.date <= end_date) & (df["Is Completed"])]
    bugs_filtered = bugs_df[(bugs_df["Week Start"].dt.date >= start_date) & (bugs_df["Week Start"].dt.date <= end_date)]

    # --- SP Summary ---
    sp_summary = df_filtered.groupby("Developer")["Story Points"].sum().reset_index().rename(columns={"Story Points": "Completed SP"})
    sp_summary["Expected SP"] = count_working_days(start_date, end_date)
    sp_summary["Productivity %"] = (sp_summary["Completed SP"] / sp_summary["Expected SP"] * 100).round(2)

    # --- Bug Summary ---
    bug_summary = bugs_filtered.groupby("Developer").size().reset_index(name="Total Bugs")
    merged = pd.merge(sp_summary, bug_summary, on="Developer", how="outer").fillna({"Completed SP": 0, "Expected SP": count_working_days(start_date, end_date), "Total Bugs": 0})

    # --- Bug Density ---
    merged["Bug Density"] = np.where(merged["Completed SP"] == 0, np.nan, merged["Total Bugs"] / merged["Completed SP"])

    # --- Normalize for Quality % ---
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) if series.max() > series.min() else series

    merged["Quality %"] = 100 - normalize(merged["Bug Density"].fillna(merged["Bug Density"].max())) * 100
    merged["Quality %"] = merged["Quality %"].where(merged["Completed SP"] > 0, np.nan).round(2)

    # --- Show Tables ---
    st.subheader("✅ Productivity Summary")
    st.dataframe(merged[["Developer", "Completed SP", "Expected SP", "Productivity %"]].sort_values("Productivity %", ascending=False))

    st.subheader("🧪 Quality Summary")
    quality_df = merged[["Developer", "Total Bugs", "Bug Density", "Quality %"]].copy()
    quality_df = quality_df.sort_values("Quality %", ascending=False)
    quality_df["Bug Density"] = quality_df["Bug Density"].round(2)
    quality_df["Quality %"] = quality_df["Quality %"].round(2)
    quality_df["Quality %"] = quality_df["Quality %"].fillna("N/A")
    st.dataframe(quality_df)

# Chat history session init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Flag to indicate code was successfully verified
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_attempted" not in st.session_state:
    st.session_state.auth_attempted = False
if "auth_failed" not in st.session_state:
    st.session_state.auth_failed = False

def render_message(sender, message):
    if sender == "user":
        st.markdown(f"""
        <div style="background-color:#DCF8C6; padding:10px 15px; border-radius:10px; margin:5px 0; text-align:right">
            <b>🧑 You:</b> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#F1F0F0; padding:10px 15px; border-radius:10px; margin:5px 0; text-align:left">
            <b>🤖 AI:</b> {message}
        </div>
        """, unsafe_allow_html=True)

def render_ai_assistant_tab(df, bugs_df):
    import re
    from dateutil.relativedelta import relativedelta

    st.subheader("🤖 AI Assistant (via OpenAI)")

    if not st.session_state.authenticated:
        with st.form("auth_form"):
            access_code_input = st.text_input("🔐 Enter access code to use AI assistant:", type="password")
            submitted_auth = st.form_submit_button("Submit")

            if submitted_auth:
                st.session_state.auth_attempted = True
                if access_code_input == st.secrets.get("ACCESS_CODE"):
                    st.session_state.authenticated = True
                    st.session_state.auth_failed = False
                    st.rerun()
                else:
                    st.session_state.auth_failed = True

        if st.session_state.auth_failed:
            st.error("Incorrect access code. Please try again.")
        return

    if st.button("🔁 Clear Context & Chat History"):
        st.session_state.pop("chat_context", None)
        st.session_state.chat_history.clear()
        st.rerun()

    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)

    # Extract dynamic timeframe from last question if exists
    def extract_date_range_from_prompt(prompt):
        prompt = prompt.lower()
        today = now_ist.date()
        if "current week" in prompt or "this week" in prompt:
            start = (now_ist - timedelta(days=now_ist.weekday())).date()
            return (start, today)
        if "last few weeks" in prompt:
            return ((now_ist - timedelta(weeks=4)).date(), today)
        if m := re.search(r"last (\d+) days", prompt):
            return ((now_ist - timedelta(days=int(m.group(1)))).date(), today)
        if m := re.search(r"last (\d+) weeks", prompt):
            return ((now_ist - timedelta(weeks=int(m.group(1)))).date(), today)
        if "last 5 days" in prompt:
            return ((now_ist - timedelta(days=5)).date(), today)
        return ((now_ist - timedelta(weeks=4)).date(), today)  # default 4 weeks

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    last_prompt = st.session_state.chat_history[0][0] if st.session_state.chat_history else ""
    start_date, end_date = extract_date_range_from_prompt(last_prompt)

    # Filter data by date range
    df["Due Date"] = pd.to_datetime(df["Due Date"])
    bugs_df["Created"] = pd.to_datetime(bugs_df["Created"])
    df_recent = df[(df["Due Date"].dt.date >= start_date) & (df["Due Date"].dt.date <= end_date)].copy()
    bugs_recent = bugs_df[(bugs_df["Created"].dt.date >= start_date) & (bugs_df["Created"].dt.date <= end_date)].copy()

    context_lines = [f"## Developer Activity Summary ({start_date} to {end_date})\n"]

    all_devs = set(df_recent["Developer"].dropna().unique()).union(bugs_recent["Developer"].dropna().unique())

    for dev in sorted(all_devs):
        lines = [f"### 👨‍💼 {dev}"]

        completed = df_recent[(df_recent["Developer"] == dev) & (df_recent["Is Completed"])]
        if not completed.empty:
            lines.append("**Completed Tasks:**")
            lines.append("| Date | Key | Story Points |")
            lines.append("|------|-----|---------------|")
            for _, row in completed.sort_values("Due Date", ascending=False).iterrows():
                lines.append(f"| {row['Due Date'].date()} | {row['Key']} | {int(row['Story Points'])} |")
        else:
            lines.append("**Completed Tasks:** None")

        inprogress = df_recent[(df_recent["Developer"] == dev) & (~df_recent["Is Completed"])]
        if not inprogress.empty:
            lines.append("**In-Progress Tasks:**")
            lines.append("| Key | Story Points | Due Date |")
            lines.append("|-----|---------------|-----------|")
            for _, row in inprogress.sort_values("Due Date").iterrows():
                lines.append(f"| {row['Key']} | {int(row['Story Points'])} | {row['Due Date'].date()} |")
        else:
            lines.append("**In-Progress Tasks:** None")

        dev_bugs = bugs_recent[bugs_recent["Developer"] == dev]
        if not dev_bugs.empty:
            lines.append("**Bugs Created:**")
            lines.append("| Date | Bug ID |")
            lines.append("|------|--------|")
            for _, row in dev_bugs.sort_values("Created", ascending=False).iterrows():
                lines.append(f"| {row['Created'].date()} | {row['Key']} |")
        else:
            lines.append("**Bugs Created:** None")

        lines.append("")
        context_lines.extend(lines)

    st.session_state.chat_context = "\n".join(context_lines)

    st.markdown("---")
    st.markdown("Ask me anything about developer productivity or bugs 👇")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here...", key="user_input_text", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")

        if submitted and user_input:
            with st.spinner("Thinking..."):
                try:
                    client = openai.OpenAI(api_key=st.secrets["OPENAI"]["API_KEY"])
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                            "role": "system",
                            "content": (
                                "You are an analytical assistant. Use only the markdown tables provided in the context to answer questions about developer performance. "
                                "Count rows and values exactly. Do not assume or estimate. "
                                "Treat completed story points as a measure of productivity. "
                                "Treat the number of bugs created as a negative indicator of code quality. "
                                "Do not confuse bugs created with bug fixing. "
                                "If developers tie in counts, mention all of them. Be factual, grounded, and concise in your response."
                            )
                            },
                            {"role": "user", "content": f"Context:\n{st.session_state.chat_context}"},
                            {"role": "user", "content": user_input}
                        ]
                    )
                    reply = response.choices[0].message.content
                    st.session_state.chat_history.insert(0, (user_input, reply))
                except Exception as e:
                    st.error(f"OpenAI error: {e}")
                    return

    for q, a in st.session_state.chat_history:
        render_message("user", q)
        render_message("assistant", a)

def main():
    st.set_page_config("📊 Team Productivity Dashboard", layout="wide")
    st.title("📊 Weekly Productivity Dashboard")

    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist).strftime('%Y-%m-%d %I:%M %p %Z')
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
    with col2:
        st.caption(f"Last data refresh: {now_ist}")

    with st.spinner("Fetching and processing data from JIRA..."):
        df = load_jira_data()
        bugs_df = load_bug_data()
        if df.empty:
            st.stop()

    week_options_df = df.dropna(subset=["Week", "Week Start"])
    week_options_df = week_options_df[week_options_df["Week Start"] <= datetime.today()]
    week_options_df = week_options_df.sort_values("Week Start", ascending=False).drop_duplicates("Week Start")[["Week", "Week Start"]]
    week_label_map = {
        row["Week"]: f"{row['Week']} ({row['Week Start'].strftime('%d-%B-%Y').upper()} to {(row['Week Start'] + timedelta(days=4)).strftime('%d-%B-%Y').upper()})"
        for _, row in week_options_df.iterrows()
    }

    tabs = st.tabs(["Summary", "Trends", "Team View", "Tasks", "Quality", "Insights", "AI Assistant"])
    with tabs[0]:
        selected_week = st.selectbox("Select week to view:", options=list(week_label_map.keys()), format_func=lambda x: week_label_map[x])
        render_summary_tab(df, selected_week)
    with tabs[1]:
        render_trend_tab(df)
    with tabs[2]:
        render_team_trend(df)
    with tabs[3]:
        selected_week = st.selectbox("Select week to view:", options=list(week_label_map.keys()), format_func=lambda x: week_label_map[x], key="task_week")
        filtered_df = df[df["Week"] == selected_week]
        st.dataframe(filtered_df[["Key", "Summary", "Developer", "Status", "Due Date", "Story Points", "Week"]])
    with tabs[4]:
        render_quality_tab(bugs_df)
    with tabs[5]:
        render_insights_tab(df, bugs_df)
    with tabs[6]:
        render_ai_assistant_tab(df, bugs_df)

if __name__ == "__main__":
    main()
