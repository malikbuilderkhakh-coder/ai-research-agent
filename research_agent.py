import crewai.llms.cache as crewai_cache

# Workaround for the CrewAI/Groq cache_breakpoint issue
crewai_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Agent, Crew, Task, LLM
from ddgs import DDGS


def search_web(query: str, max_results: int = 3):
    """
    Search the web using DuckDuckGo.
    This does not use the Groq API.
    """
    try:
        results = DDGS().text(
            query,
            max_results=max_results
        )

        if not results:
            return []

        return results

    except Exception as e:
        return [
            {
                "title": "Search Error",
                "href": "",
                "body": str(e)
            }
        ]


def collect_research(topic: str, number_of_sources: int):
    """
    Collect web information before calling the LLM.
    We use a small number of searches to reduce API usage.
    """

    queries = [
        topic,
        f"{topic} latest information",
    ]

    all_results = []

    for query in queries:
        results = search_web(
            query,
            max_results=number_of_sources
        )

        all_results.extend(results)

    # Remove duplicate URLs
    unique_results = []
    seen_urls = set()

    for result in all_results:
        url = result.get("href", "")

        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    # Keep the requested number of sources
    return unique_results[:number_of_sources]


def run_research(
    topic: str,
    number_of_sources: int,
    api_key: str
):
    """
    Main research function.

    DuckDuckGo collects the sources first.
    CrewAI then uses ONE LLM call to write the report.
    """

    # --------------------------------------------------
    # STEP 1: Search the web
    # --------------------------------------------------

    sources = collect_research(
        topic,
        number_of_sources
    )

    if not sources:
        raise Exception(
            "DuckDuckGo did not return any search results. "
            "Please try another research topic."
        )

    # --------------------------------------------------
    # STEP 2: Prepare research information
    # --------------------------------------------------

    research_text = ""

    for i, source in enumerate(sources, start=1):

        title = source.get("title", "No title")
        url = source.get("href", "No URL")
        snippet = source.get("body", "No description")

        research_text += f"""
SOURCE {i}

Title:
{title}

URL:
{url}

Information:
{snippet}

-----------------------------------
"""

    # --------------------------------------------------
    # STEP 3: Create Groq LLM
    # --------------------------------------------------

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
        reasoning_effort="low",
    )

    # --------------------------------------------------
    # STEP 4: Create ONE CrewAI agent
    # --------------------------------------------------

    researcher = Agent(
        role="Senior Research Analyst",

        goal=(
            "Create an accurate, well-structured and easy-to-understand "
            "research report using the research information provided."
        ),

        backstory=(
            "You are a professional research analyst. "
            "You carefully analyze research sources, "
            "avoid inventing information, and clearly explain "
            "important findings."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=1,
    )

    # --------------------------------------------------
    # STEP 5: Create research task
    # --------------------------------------------------

    task = Task(

        description=f"""
You need to write a research report about:

{topic}

The following information was collected from DuckDuckGo:

{research_text}

Use ONLY the information provided above as the research
evidence.

IMPORTANT RULES:

1. Do not invent facts.
2. Do not invent statistics.
3. Do not invent quotations.
4. Do not create fake sources.
5. Clearly explain information in simple language.
6. Mention limitations when the available information is limited.
7. Include the actual URLs provided in the sources.
8. Do not claim that you personally visited or verified a website.
9. Use the source information carefully.

Write the report using this structure:

# Research Report

## 1. Executive Summary

Give a short summary of the research.

## 2. Introduction

Explain the topic and why it is important.

## 3. Key Findings

List the most important findings.

## 4. Detailed Analysis

Explain the research information in detail.

## 5. Benefits / Opportunities

Explain important benefits or opportunities if applicable.

## 6. Challenges / Limitations

Explain important problems, risks, or limitations.

## 7. Current Developments

Discuss recent developments only when supported by
the provided research.

## 8. Conclusion

Give a short conclusion based on the evidence.

## 9. Sources

List every source with:

- Source title
- URL

Make the report professional but easy for a beginner
to understand.
""",

        expected_output=(
            "A complete research report with an executive summary, "
            "introduction, key findings, detailed analysis, "
            "benefits, challenges, current developments, "
            "conclusion, and source URLs."
        ),

        agent=researcher,
    )

    # --------------------------------------------------
    # STEP 6: Run CrewAI
    # --------------------------------------------------

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()

    return result
