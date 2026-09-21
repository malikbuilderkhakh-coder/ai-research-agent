from typing import Type

from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(
        ...,
        description="A focused web search query.",
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Search the public web with DuckDuckGo. "
        "Returns titles, URLs, and snippets."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return "No search results were found."

            output = []
            for i, result in enumerate(results, start=1):
                output.append(
                    f"SOURCE {i}\n"
                    f"Title: {result.get('title', 'No title')}\n"
                    f"URL: {result.get('href', 'No URL')}\n"
                    f"Snippet: {result.get('body', 'No snippet')}\n"
                )

            return "\n".join(output)

        except Exception as exc:
            return f"Search failed: {exc}"


def run_research(topic: str, number_of_sources: int, api_key: str):
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
        reasoning_effort="medium",
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            "Research the user's topic using web sources and create "
            "an accurate, structured and understandable report."
        ),
        backstory=(
            "You are a careful research analyst. You search for "
            "relevant information, compare sources, avoid unsupported "
            "claims, and clearly separate facts from uncertain claims."
        ),
        tools=[DuckDuckGoSearchTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Research this topic:

{topic}

Use the DuckDuckGo Web Search tool to perform real web research.

Requirements:
1. Use multiple searches when useful.
2. Use approximately {number_of_sources} useful sources where possible.
3. Prefer authoritative and reputable sources.
4. Do not invent facts, statistics, quotations, or sources.
5. Clearly distinguish established facts from uncertain claims.
6. Include source titles and URLs in the final report.
7. Focus on useful and understandable information.

Write the report using this structure:

# Research Report

## 1. Executive Summary
## 2. Introduction
## 3. Key Findings
## 4. Detailed Analysis
## 5. Benefits / Opportunities
## 6. Challenges / Limitations
## 7. Current Developments
## 8. Conclusion
## 9. Sources

Make the report clear enough for a beginner to understand.
""",
        expected_output=(
            "A complete, well-structured research report with source "
            "titles and URLs."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True,
    )

    return crew.kickoff()
