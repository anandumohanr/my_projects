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

    start_at = 0
    max_results = 100   # page size
    all_issues = []
    try:
        while True:
            params = {
                "jql": jql,
                "fields": "key,summary,status,customfield_11020,customfield_10010,customfield_11012,created",
                "startAt": start_at,
                "maxResults": max_results
            }
            response = requests.get(url, headers=headers, auth=auth, params=params)
            response.raise_for_status()
            payload = response.json()

            issues = payload.get("issues", []) or []
            page_keys = [iss.get("key") for iss in issues]

            all_issues.extend(issues)

            # increment start_at by number of issues returned (safer than payload.maxResults)
            if len(issues) == 0:
                break
            start_at += len(issues)

            # stop when we've reached the total
            total = payload.get("total", 0)
            if start_at >= total:
                break

        # After paging, show collected summary
        collected_keys = [iss.get("key") for iss in all_issues]

        # OPTIONAL: quickly check for specific keys you know are missing
        expected_missing = ["MDLRN-25338", "MDLRN-25360"]  # replace with your missing keys
        still_missing = [k for k in expected_missing if k not in collected_keys]
        if still_missing:
            st.error(f"Expected keys still not in collected pages: {still_missing}")
        else:
            st.success("All expected keys found in the collected pages.")

        # build DataFrame rows
        data = []
        for issue in all_issues:
            fields = issue.get("fields", {}) or {}
            dev_field = fields.get("customfield_11012")
            # robust developer extraction
            developer = ""
            if isinstance(dev_field, dict):
                developer = dev_field.get("displayName") or dev_field.get("name") or dev_field.get("accountId") or ""
            elif isinstance(dev_field, list) and len(dev_field) > 0:
                developer = ", ".join([d.get("displayName","") if isinstance(d, dict) else str(d) for d in dev_field])
            elif dev_field is not None:
                developer = str(dev_field)

            data.append({
                "Key": issue.get("key"),
                "Summary": fields.get("summary", "") or "",
                "Status": (fields.get("status") or {}).get("name", "") or "",
                "Due Date": fields.get("customfield_11020"),
                "Story Points": fields.get("customfield_10010", 0),
                "Developer": developer
            })

        df = pd.DataFrame(data)

        # normalize strings
        import unicodedata, re
        def normalize_str(v):
            if pd.isna(v):
                return ""
            s = str(v)
            s = unicodedata.normalize("NFKC", s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        for col in ["Key", "Summary", "Status", "Developer"]:
            if col in df.columns:
                df[col] = df[col].apply(normalize_str)

        # coerce types
        df["Due Date"] = pd.to_datetime(df["Due Date"], errors="coerce")
        df["Story Points"] = pd.to_numeric(df["Story Points"], errors="coerce").fillna(0)

        iso = df["Due Date"].dt.isocalendar()
        # handle possible NaT gracefully
        df["Week"] = iso["year"].astype('Int64').astype(str) + "-" + iso["week"].astype('Int64').apply(lambda x: f"{int(x):02d}" if pd.notna(x) else None)
        df["Week Start"] = df["Due Date"].dt.to_period("W").dt.start_time
        df["Month"] = df["Due Date"].dt.to_period("M").dt.to_timestamp()
        df["Quarter"] = df["Due Date"].dt.to_period("Q").dt.to_timestamp()
        df["Year"] = df["Due Date"].dt.year
        df["Is Completed"] = df["Status"].str.upper().isin([s.upper() for s in COMPLETED_STATUSES])
        df["Developer"] = df["Developer"].replace("", "(Unassigned)")
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

    st.header("📊 Developer Insights (Productivity & Quality)")

    def count_working_days(start_date, end_date):
        return sum(1 for d in pd.date_range(start=start_date, end=end_date) if d.weekday() < 5)

    today = datetime.today().date()
    default_start = today - timedelta(weeks=4)
    min_date = df["Due Date"].min().date()
    max_date = df["Due Date"].max().date()

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
        quarter_start = 3 * ((today.month - 1) // 3) + 1
        start_date = datetime(today.year, quarter_start, 1).date()
        end_date = today
    elif filter_option == "Previous Quarter":
        quarter_start = 3 * ((today.month - 1) // 3) + 1
        prev_q_end = datetime(today.year, quarter_start, 1) - timedelta(days=1)
        prev_q_start = 3 * ((prev_q_end.month - 1) // 3) + 1
        start_date = datetime(prev_q_end.year, prev_q_start, 1).date()
        end_date = prev_q_end.date()
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
        with col2:
            # Ensure valid range for end date input
            adjusted_today = min(today, max_date)
            adjusted_start = min(start_date, max_date)
            end_date = st.date_input("End Date", value=adjusted_today, min_value=adjusted_start, max_value=max_date)

    with st.spinner("🔄 Generating insights... Please wait."):
        df_filtered = df[
            (df["Due Date"].dt.date >= start_date) &
            (df["Due Date"].dt.date <= end_date) &
            (df["Is Completed"])
        ]
        bugs_filtered = bugs_df[
            (bugs_df["Created"].dt.date >= start_date) &
            (bugs_df["Created"].dt.date <= end_date)
        ]

        sp_summary = df_filtered.groupby("Developer")["Story Points"].sum().reset_index().rename(columns={"Story Points": "Completed SP"})
        all_devs = pd.DataFrame(df_filtered["Developer"].unique(), columns=["Developer"])
        sp_summary = pd.merge(all_devs, sp_summary, on="Developer", how="left").fillna(0)
        sp_summary["Expected SP"] = count_working_days(start_date, end_date)
        sp_summary["Productivity Numeric"] = (sp_summary["Completed SP"] / sp_summary["Expected SP"] * 100).round().fillna(0)

        bug_summary = bugs_filtered.groupby("Developer").size().reset_index(name="Total Bugs")
        merged = pd.merge(sp_summary, bug_summary, on="Developer", how="outer").fillna({
            "Completed SP": 0,
            "Expected SP": count_working_days(start_date, end_date),
            "Total Bugs": 0
        })

        merged["Productivity Numeric"] = merged["Productivity Numeric"].fillna(0)
        merged["Productivity %"] = merged["Productivity Numeric"].astype(int).astype(str) + " %"
        merged["Bug Density"] = np.where(merged["Completed SP"] == 0, np.nan, merged["Total Bugs"] / merged["Completed SP"]).round(3)
        merged["Quality Numeric"] = 100 - (merged["Bug Density"] * 200)
        merged["Quality Numeric"] = merged["Quality Numeric"].where(merged["Completed SP"] > 0, 0).clip(lower=0).round()
        merged["Quality %"] = merged["Quality Numeric"].astype(int).astype(str) + " %"

        st.subheader("✅ Productivity Summary")
        st.dataframe(
            merged[["Developer", "Completed SP", "Expected SP", "Productivity %", "Productivity Numeric"]]
            .sort_values("Productivity Numeric", ascending=False)
            .drop(columns=["Productivity Numeric"])
        )

        st.subheader("🧪 Quality Summary")
        quality_df = merged[["Developer", "Completed SP", "Total Bugs", "Bug Density", "Quality %", "Quality Numeric"]].copy()
        quality_df["Bug Density"] = quality_df["Bug Density"].round(2)
        st.dataframe(
            quality_df.sort_values("Quality Numeric", ascending=False)
            .drop(columns=["Quality Numeric"])
        )

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

def render_tasks_tab(df, bugs_df):
    import streamlit as st
    from datetime import datetime, timedelta
    import pandas as pd

    st.header("🗂️ Task & Bug Listings")

    # --- Date Range Filter ---
    today = datetime.today().date()
    min_due = df["Due Date"].min().date()
    max_due = df["Due Date"].max().date()
    min_created = bugs_df["Created"].min().date()
    max_created = bugs_df["Created"].max().date()

    global_min = min(min_due, min_created)
    global_max = max(max_due, max_created)
    default_start = global_max - timedelta(weeks=4)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start, min_value=global_min, max_value=global_max, key="tasks_start")
    with col2:
        # Ensure valid range for end date input
        adjusted_today = min(today, global_max)
        adjusted_start = min(start_date, global_max)
        end_date = st.date_input("End Date", value=adjusted_today, min_value=adjusted_start, max_value=global_max, key="tasks_end")

    # --- Developer Filter ---
    all_devs = sorted(set(df["Developer"].dropna().unique()) | set(bugs_df["Developer"].dropna().unique()))
    selected_devs = st.multiselect("Filter by Developer (optional)", options=all_devs)

    # --- Filter Tasks ---
    filtered_tasks = df[
        (df["Due Date"].dt.date >= start_date) &
        (df["Due Date"].dt.date <= end_date)
    ]
    if selected_devs:
        filtered_tasks = filtered_tasks[filtered_tasks["Developer"].isin(selected_devs)]

    # --- Filter Bugs ---
    filtered_bugs = bugs_df[
        (bugs_df["Created"].dt.date >= start_date) &
        (bugs_df["Created"].dt.date <= end_date)
    ]
    if selected_devs:
        filtered_bugs = filtered_bugs[filtered_bugs["Developer"].isin(selected_devs)]

    # --- Format Dates ---
    formatted_tasks = filtered_tasks.copy()
    formatted_tasks["Due Date"] = formatted_tasks["Due Date"].dt.strftime("%d-%B-%Y").str.upper()

    formatted_bugs = filtered_bugs.copy()
    formatted_bugs["Created"] = pd.to_datetime(formatted_bugs["Created"]).dt.strftime("%d-%B-%Y").str.upper()

    # --- Show Tables ---
    st.subheader("✅ Tasks")
    st.dataframe(
        formatted_tasks[["Key", "Summary", "Developer", "Status", "Due Date", "Story Points"]],
        use_container_width=True
    )

    st.subheader("🐞 Bugs")
    st.dataframe(
        formatted_bugs[["Key", "Summary", "Developer", "Created"]],
        use_container_width=True
    )

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

    tabs = st.tabs(["Summary", "Trends", "Team View", "Quality", "Insights", "Tasks & Bug List", "AI Assistant"])
    with tabs[0]:
        selected_week = st.selectbox("Select week to view:", options=list(week_label_map.keys()), format_func=lambda x: week_label_map[x])
        render_summary_tab(df, selected_week)
    with tabs[1]:
        render_trend_tab(df)
    with tabs[2]:
        render_team_trend(df)
    with tabs[3]:
        render_quality_tab(bugs_df)
    with tabs[4]:
        render_insights_tab(df, bugs_df)
    with tabs[5]:
        render_tasks_tab(df, bugs_df)
    with tabs[6]:
        render_ai_assistant_tab(df, bugs_df)

if __name__ == "__main__":
    main()
