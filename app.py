import streamlit as st
from research_agent import run_research

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Research Agent")
st.write(
    "Enter a research topic and the AI agent will search the web "
    "and create a structured research report."
)

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Secrets.")
    st.stop()

with st.sidebar:
    st.header("Research Settings")
    number_of_sources = st.slider(
        "Number of sources",
        min_value=3,
        max_value=10,
        value=5,
    )
    st.caption("Model: openai/gpt-oss-120b via Groq")

topic = st.text_area(
    "Research Topic",
    placeholder="Example: Impact of artificial intelligence on education",
    height=120,
)

if st.button("🚀 Start Research", type="primary", use_container_width=True):
    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    with st.spinner("🔎 Research agent is working..."):
        try:
            report = run_research(
                topic=topic.strip(),
                number_of_sources=number_of_sources,
                api_key=st.secrets["GROQ_API_KEY"],
            )

            st.success("Research completed!")
            st.markdown("## 📄 Research Report")
            st.markdown(str(report))

            st.download_button(
                "⬇️ Download Report",
                data=str(report),
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception as exc:
            st.error("The research agent encountered an error.")
            st.exception(exc)
