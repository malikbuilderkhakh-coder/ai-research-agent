---
title: AI Research Agent
emoji: 🔎
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# 🔎 AI Research Agent

A beginner-friendly single-agent research application built with:

- Streamlit
- CrewAI
- Groq
- OpenAI GPT-OSS 120B
- DuckDuckGo Search

## How it works

1. Enter a research topic.
2. The CrewAI research agent searches the web.
3. GPT-OSS 120B analyzes the collected information.
4. The app generates a structured research report.
5. The report can be downloaded as Markdown.

## API key

The application requires a `GROQ_API_KEY`.

For Streamlit Community Cloud:

1. Open your deployed app.
2. Go to Settings.
3. Open Secrets.
4. Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Never commit your real API key to GitHub.

## Local development

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```
