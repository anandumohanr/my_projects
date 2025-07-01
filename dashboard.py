import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt

# Constants
SHAREPOINT_URL = "https://impelsysinc-my.sharepoint.com/:x:/g/personal/anandu_m_medlern_com/EXxi7DECTpxDgA-Hx44P-G8B-PgU74kHUVKlz3VfbTNX5w?download=1"
COMPLETED_STATUSES = ["ACCEPTED IN QA", "CLOSED"]

@st.cache_data(show_spinner=False)
def load_excel():
    try:
        st.info("Downloading Excel from SharePoint...")
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
    df["Is Completed"] = df["Status"].str.upper().isin(COMPLETED_STATUSES)
    return df

def get_week_options(df):
    return sorted(df["Week"].dropna().unique())

def plot_productivity_bar(df, title, by="Developer"):
    summary = df.groupby(by)["Story Points"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    summary.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Story Points")
    st.pyplot(fig)

def main():
    st.set_page_config("Productivity Dashboard", layout="wide")
    st.title("\U0001F4CA Weekly Productivity Dashboard")

    df = load_excel()
    if df.empty:
        st.stop()

    df = preprocess_data(df)

    week_options = get_week_options(df)
    selected_weeks = st.multiselect("Select week(s) to view:", week_options, default=week_options[-1:])
    filtered_df = df[df["Week"].isin(selected_weeks)]

    st.subheader("Developer-wise Productivity")
    completed_df = filtered_df[filtered_df["Is Completed"]]
    plot_productivity_bar(completed_df, "Completed Story Points per Developer")

    st.subheader("Team Overview")
    team_summary = completed_df.groupby("Week")["Story Points"].sum().reset_index()
    st.dataframe(team_summary)

    st.subheader("Detailed Task Table")
    st.dataframe(filtered_df[["Key", "Summary", "Developer", "Status", "Due Date", "Story Points", "Week"]])

    st.download_button("Download Summary CSV", data=team_summary.to_csv(index=False), file_name="team_productivity.csv")

if __name__ == "__main__":
    main()
