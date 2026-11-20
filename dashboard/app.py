"""GameOps AI dashboard.  Run:  streamlit run dashboard/app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="GameOps AI", page_icon=":material/sports_esports:", layout="wide")

V = Path(__file__).resolve().parent / "views"
pages = {
    "Insights": [
        st.Page(str(V / "overview.py"), title="Executive Overview", icon=":material/dashboard:", default=True),
        st.Page(str(V / "sentiment.py"), title="Player Sentiment", icon=":material/sentiment_satisfied:"),
        st.Page(str(V / "complaints.py"), title="Complaint Intelligence", icon=":material/report:"),
        st.Page(str(V / "updates.py"), title="Game Update Impact", icon=":material/system_update:"),
    ],
    "AI": [
        st.Page(str(V / "analyst.py"), title="AI Analyst", icon=":material/smart_toy:"),
        st.Page(str(V / "evaluation.py"), title="AI Evaluation & Monitoring", icon=":material/monitoring:"),
        st.Page(str(V / "playground.py"), title="Model Playground", icon=":material/science:"),
    ],
}
st.sidebar.markdown("### 🎮 GameOps AI")
st.sidebar.caption("Agentic player-feedback intelligence")
st.navigation(pages).run()
