from typing import Type

# Workaround for CrewAI/Groq cache_breakpoint compatibility issue
import crewai.llms.cache as crewai_cache

crewai_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


# ============================================================
# 1. SEARCH INPUT
# ============================================================

class SearchInput(BaseModel):
    query: str = Field(
        ...,
        description="A focused web search query."
    )


# ============================================================
# 2. DUCKDUCKGO SEARCH TOOL
# ============================================================

class DuckDuckGoSearchTool(BaseTool):

    name: str = "DuckDuckGo Web Search"

    description: str = (
        "Search the public web using DuckDuckGo. "
        "Returns titles, URLs, and snippets from search results."
    )

    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:

        try:

            results = DDGS().text(
                query,
                max_results=5
            )

            if not results:
                return "No search results were found."

            output = []

            for i, result in enumerate(results, start=1):

                title = result.get(
                    "title",
                    "No title"
                )

                url = result.get(
                    "href",
                    "No URL"
                )

                snippet = result.get(
                    "body",
                    "No description"
                )

                output.append(
                    f"""
SOURCE {i}

Title:
{title}

URL:
{url}

Description:
{snippet}
"""
                )

            return "\n".join(output)

        except Exception as exc:

            return f"Search failed: {exc}"


# ============================================================
# 3. RUN RESEARCH
# ============================================================

def run_research(
    topic: str,
    number_of_sources: int,
    api_key: str
):

    # --------------------------------------------------------
    # GROQ LLM
    # --------------------------------------------------------

    llm = LLM(

        model="groq/openai/gpt-oss-120b",

        api_key=api_key,

        temperature=0.2,

        reasoning_effort="low"
    )


    # --------------------------------------------------------
    # RESEARCH AGENT
    # --------------------------------------------------------

    researcher = Agent(

        role="Senior Research Analyst",

        goal=(
            "Research the user's topic using reliable web sources "
            "and create an accurate, structured and understandable "
            "research report."
        ),

        backstory=(
            "You are a careful research analyst. "
            "You search for relevant information, compare sources, "
            "avoid unsupported claims, and clearly separate facts "
            "from uncertain claims."
        ),

        tools=[
            DuckDuckGoSearchTool()
        ],

        llm=llm,

        verbose=True,

        allow_delegation=False
    )


    # --------------------------------------------------------
    # RESEARCH TASK
    # --------------------------------------------------------

    task = Task(

        description=f"""

Research the following topic:

{topic}


Use the DuckDuckGo Web Search tool to perform real web research.


Research requirements:

1. Perform multiple searches when useful.

2. Use approximately {number_of_sources}
   useful sources where possible.

3. Prefer authoritative and reputable sources.

4. Do not invent facts, statistics,
   quotations or sources.

5. Clearly distinguish established facts
   from uncertain claims.

6. Include source titles and URLs
   in the final report.

7. Focus on useful and understandable information.


Write the report using this structure:


# Research Report


## 1. Executive Summary

Give a short summary of the research.


## 2. Introduction

Explain the topic and why it matters.


## 3. Key Findings

Present the most important findings.


## 4. Detailed Analysis

Explain the topic in detail.


## 5. Benefits / Opportunities

Discuss important benefits or opportunities.


## 6. Challenges / Limitations

Discuss important challenges and limitations.


## 7. Current Developments

Discuss recent developments when reliable
information is available.


## 8. Conclusion

Summarize the major findings.


## 9. Sources

List the sources used with their titles
and URLs.


Make the report clear enough for a beginner
to understand.

""",

        expected_output=(
            "A complete, well-structured research report "
            "containing an executive summary, introduction, "
            "key findings, detailed analysis, benefits, "
            "challenges, current developments, conclusion, "
            "and source URLs."
        ),

        agent=researcher
    )


    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

    crew = Crew(

        agents=[
            researcher
        ],

        tasks=[
            task
        ],

        verbose=True
    )


    # --------------------------------------------------------
    # START RESEARCH
    # --------------------------------------------------------

    result = crew.kickoff()

    return result
