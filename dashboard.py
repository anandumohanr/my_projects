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
DEVELOPERS = [f
    "Anandu Mohan", "Ravi Kumar", "shree.vidya", "Brijesh Kanchan", "Hari Prasad H S",
    "Fahad P K", "Venukumar DL", "Kishore C", "padmaja"
]

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

def render_summary_tab(df, selected_week):
    st.subheader("Developer Productivity Summary")
    filtered_df = df[df["Week"] == selected_week]
    completed_df = filtered_df[filtered_df["Is Completed"]]
    in_progress_df_all = df[~df["Is Completed"]]

    developer_completed = completed_df.groupby("Developer")["Story Points"].sum().astype(int).to_dict()
    developer_inprogress = in_progress_df_all.groupby("Developer")["Story Points"].sum().astype(int).to_dict()
    all_developers = set(DEVELOPERS).union(developer_completed).union(developer_inprogress)
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
    trend_data = df_dev.groupby(group_col)["Story Points"].sum().reset_index()

    if trend_data.empty:
        st.info("No data available.")
        return

    trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
        x=alt.X(f"{group_col}:N", title=period),
        y=alt.Y("Story Points", title="Completed Story Points"),
        tooltip=[group_col, "Story Points"]
    ).properties(height=300)

    st.altair_chart(trend_chart, use_container_width=True)
    st.dataframe(trend_data)

def render_team_trend(df):
    st.subheader("👥 Team Trend")
    period = st.selectbox("Team View By", ["Week", "Month", "Quarter", "Year"], key="team_period")
    df_team = df[df["Is Completed"]]
    if period == "Week":
        group_col = "Week"
    elif period == "Month":
        group_col = "Month"
    elif period == "Quarter":
        group_col = "Quarter"
    else:
        group_col = "Year"
    grouped = df_team.groupby(group_col)["Story Points"].sum().reset_index()
    grouped.columns = [group_col, "Total SP"]

    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X(f"{group_col}:N", title=period),
        y=alt.Y("Total SP", title="Team Story Points")
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(grouped)

def render_quality_tab(bugs_df):
    st.subheader("🐞 Bug and Quality Metrics")
    period = st.selectbox("View Bugs By", ["Week", "Month", "Quarter", "Year"], key="bug_period")
    bugs = bugs_df.copy()
    bugs["Bugs"] = 1

    full_index = pd.MultiIndex.from_product([
        sorted(bugs[period].dropna().unique()),
        sorted(bugs["Developer"].dropna().unique())
    ], names=[period, "Developer"])

    grouped = bugs.groupby([period, "Developer"])["Bugs"].sum().reindex(full_index, fill_value=0).reset_index()

    st.altair_chart(
        alt.Chart(grouped).mark_bar().encode(
            x=alt.X(f"{period}:N", title=period),
            y=alt.Y("Bugs", title="Bug Count"),
            color="Developer"
        ).properties(height=300),
        use_container_width=True
    )
    st.dataframe(grouped)

def render_ai_assistant_tab(df, bugs_df):
    st.subheader("🤖 AI Assistant (via OpenAI)")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        with st.form("auth_form"):
            access_code_input = st.text_input("🔐 Enter access code to use AI assistant:", type="password")
            submitted_auth = st.form_submit_button("Submit")
            if submitted_auth and access_code_input == st.secrets.get("ACCESS_CODE"):
                st.session_state.authenticated = True
                st.rerun()
        return

    if st.button("🔁 Clear Context & Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    prompt = st.text_input("Ask about productivity or bugs", key="user_input_text")
    if prompt:
        start = now_ist - timedelta(weeks=4)
        df_recent = df[(df["Due Date"] >= start) & (df["Due Date"] <= now_ist)]
        bugs_recent = bugs_df[(bugs_df["Created"] >= start) & (bugs_df["Created"] <= now_ist)]
        context = []
        for dev in sorted(set(df_recent["Developer"]).union(bugs_recent["Developer"])):
            completed = df_recent[(df_recent["Developer"] == dev) & df_recent["Is Completed"]]
            inprogress = df_recent[(df_recent["Developer"] == dev) & (~df_recent["Is Completed"])]
            bugs = bugs_recent[bugs_recent["Developer"] == dev]
            context.append(f"### {dev}")
            if not completed.empty:
                context.append("**Completed Tasks:**")
                for _, r in completed.iterrows():
                    context.append(f"- {r['Key']} ({r['Story Points']} SP)")
            if not inprogress.empty:
                context.append("**In-Progress Tasks:**")
                for _, r in inprogress.iterrows():
                    context.append(f"- {r['Key']} ({r['Story Points']} SP)")
            if not bugs.empty:
                context.append("**Bugs Created:**")
                for _, r in bugs.iterrows():
                    context.append(f"- {r['Key']}")
        full_context = "".join(context)
        with st.spinner("Thinking..."):
            try:
                client = openai.OpenAI(api_key=st.secrets["OPENAI"]["API_KEY"])
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a factual assistant. Use only the context to respond."},
                        {"role": "user", "content": f"Context:
{full_context}"},
                        {"role": "user", "content": prompt}
                    ]
                )
                reply = response.choices[0].message.content
                st.session_state.chat_history.insert(0, (prompt, reply))
            except Exception as e:
                st.error(f"OpenAI error: {e}")

    for q, a in st.session_state.get("chat_history", []):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 AI:** {a}")

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

    week_options_df = df.dropna(subset=["Week", "Week Start"]).drop_duplicates("Week")[["Week", "Week Start"]]
    week_options_df = week_options_df[week_options_df["Week Start"] <= datetime.today()]
    week_label_map = {
        row["Week"]: f"{row['Week']} ({row['Week Start'].strftime('%d-%B-%Y').upper()} to {(row['Week Start'] + timedelta(days=4)).strftime('%d-%B-%Y').upper()})"
        for _, row in week_options_df.iterrows()
    }
    selected_week = st.selectbox("Select week to view:", options=list(week_label_map.keys()), format_func=lambda x: week_label_map[x])

    tabs = st.tabs(["Summary", "Trends", "Team View", "Tasks", "Quality", "AI Assistant"])
    with tabs[0]:
        render_summary_tab(df, selected_week)
    with tabs[1]:
        render_trend_tab(df)
    with tabs[2]:
        render_team_trend(df)
    with tabs[3]:
        filtered_df = df[df["Week"] == selected_week]
        st.dataframe(filtered_df[["Key", "Summary", "Developer", "Status", "Due Date", "Story Points", "Week"]])
    with tabs[4]:
        render_quality_tab(bugs_df)
    with tabs[5]:
        render_ai_assistant_tab(df, bugs_df)

if __name__ == "__main__":
    main()
