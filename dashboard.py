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
        df["Week"] = df["Created"].dt.to_period("W").astype(str)
        return df

    except Exception as e:
        st.error(f"Failed to fetch Bug data: {e}")
        return pd.DataFrame()

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

        line = alt.Chart(weekly_dev).mark_line(color="steelblue").encode(
        x=alt.X("Week:N", title="Week"),
        y=alt.Y("Story Points:Q", title="Story Points"),
        tooltip=["Week", "Story Points"]
        )
        
        points = alt.Chart(weekly_dev).mark_point(filled=True).encode(
            x="Week:N",
            y="Story Points:Q",
            color=alt.Color("Color:N", scale=None),
            tooltip=["Week", "Story Points", "Change"]
        )
        
        st.altair_chart((line + points).properties(title=f"{dev_option} Productivity (Last 4 Weeks)", height=250), use_container_width=True)

        st.markdown("#### Tabular View")
        st.dataframe(weekly_dev[["Week", "Story Points", "Change"]])
    else:
        st.info("No completed tasks found for selected developer.")

def render_tasks_tab(df, selected_week):
    st.subheader("Detailed Task Table")
    filtered_df = df[df["Week"] == selected_week]
    st.dataframe(filtered_df[["Key", "Summary", "Developer", "Status", "Due Date", "Story Points", "Week"]])

def render_quality_tab(bugs_df):
    st.subheader("Bug and Quality Metrics")

    if bugs_df.empty:
        st.warning("No bug data available.")
        return

    # Ensure 'Week' column exists in correct format
    bugs_df["Created"] = pd.to_datetime(bugs_df["Created"])
    bugs_df["Week"] = bugs_df["Created"].dt.strftime("%Y-%W")
    bugs_df["Week Start"] = bugs_df["Created"].dt.to_period("W").dt.start_time
    bugs_df["Week End"] = bugs_df["Week Start"] + timedelta(days=4)

    # Build full label for table use
    bugs_df["Week Label"] = bugs_df["Week"] + " (" + bugs_df["Week Start"].dt.strftime("%d-%B-%Y").str.upper() + " to " + bugs_df["Week End"].dt.strftime("%d-%B-%Y").str.upper() + ")"

    # Recent 6 weeks filter
    today = datetime.today()
    recent_weeks = pd.date_range(end=today, periods=6, freq='W-MON')
    recent_weeks_str = [dt.strftime("%Y-%W") for dt in recent_weeks]

    st.markdown("### 📈 Bug Trends by Week")
    weekly_bugs = bugs_df[bugs_df["Week"].isin(recent_weeks_str)]
    weekly_summary = weekly_bugs.groupby(["Week", "Week Label"]).size().reset_index(name="Bug Count").sort_values("Week")

    # Line chart using just 'Week'
    chart = alt.Chart(weekly_summary).mark_line(point=True).encode(
        x=alt.X("Week:N", title="Week"),
        y=alt.Y("Bug Count", title="Bug Count")
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)

    # Table with detailed label
    st.dataframe(weekly_summary[["Week Label", "Bug Count"]].rename(columns={"Week Label": "Week"}))

    st.markdown("### 👩‍💻 Developer Bug Breakdown")
    dev_option = st.selectbox("Select Developer:", options=sorted(bugs_df["Developer"].dropna().unique()))
    df_dev = bugs_df[(bugs_df["Developer"] == dev_option) & (bugs_df["Week"].isin(recent_weeks_str))]

    dev_summary = df_dev.groupby(["Week", "Week Label"]).size().reset_index(name="Bug Count").sort_values("Week")

    dev_chart = alt.Chart(dev_summary).mark_line(point=True).encode(
        x=alt.X("Week:N", title="Week"),
        y=alt.Y("Bug Count", title="Bugs Reported")
    ).properties(height=250)
    st.altair_chart(dev_chart, use_container_width=True)

    st.dataframe(dev_summary[["Week Label", "Bug Count"]].rename(columns={"Week Label": "Week"}))

    st.markdown("### 💬 Insights")
    top_buggers = bugs_df.groupby("Developer").size().reset_index(name="Bug Count").sort_values("Bug Count", ascending=False)
    st.write("**Top Bug Reporters:**")
    st.dataframe(top_buggers.head(5))

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
    st.set_page_config("Productivity Dashboard", layout="wide")
    st.title("📊 Weekly Productivity Dashboard")

    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist).strftime('%Y-%m-%d %I:%M %p %Z')
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
            st.success("✅ JIRA data refreshed successfully")
    with col2:
        st.caption(f"Last data refresh: {now_ist}")

    with st.spinner("Fetching and processing data from JIRA..."):
        df = load_jira_data()
        bugs_df = load_bug_data()
                
        if df.empty:
            st.stop()

    week_options_df = get_week_options(df)
    week_label_map = {
        row["Week"]: f"{row['Week']} ({row['Week Start'].strftime('%d-%B-%Y').upper()} to {(row['Week Start'] + timedelta(days=4)).strftime('%d-%B-%Y').upper()})"
        for _, row in week_options_df.iterrows()
    }
    selected_week = st.selectbox("Select week to view:", options=list(week_label_map.keys()), format_func=lambda x: week_label_map[x])

    tabs = st.tabs(["Summary", "Trends", "Tasks", "Quality", "AI Assistant"])

    with tabs[0]:
        summary_df, team_summary = render_summary_tab(df, selected_week)

    with tabs[1]:
        render_trend_tab(df)

    with tabs[2]:
        render_tasks_tab(df, selected_week)

    with tabs[3]:
        render_quality_tab(bugs_df)

    with tabs[4]:
        render_ai_assistant_tab(df, bugs_df)

if __name__ == "__main__":
    main()
