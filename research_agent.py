# ============================================================
# research_agent.py
# Beginner-friendly CrewAI + DDGS Research Agent
# ============================================================

import os

# ------------------------------------------------------------
# CrewAI cache workaround
# ------------------------------------------------------------

try:
    import crewai.llms.cache as crewai_cache

    if not hasattr(crewai_cache, "mark_cache_breakpoint"):
        crewai_cache.mark_cache_breakpoint = lambda msg: msg

except Exception:
    # If this module/version does not need the workaround,
    # continue normally.
    pass


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from crewai import Agent, Crew, Task, LLM
from ddgs import DDGS


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(query: str, max_results: int = 5):
    """
    Search the web using DDGS.

    Returns:
        list: Search result dictionaries.
    """

    if not query or not query.strip():
        return []

    max_results = max(1, min(max_results, 10))

    try:

        print(f"Searching web for: {query}")

        # Create a new DDGS object for every search.
        # This is more reliable for repeated searches.
        ddgs = DDGS(timeout=20)

        results = ddgs.text(
            query=query.strip(),
            region="us-en",
            safesearch="moderate",
            max_results=max_results,
            backend="auto",
        )

        if results is None:
            return []

        # Convert to list in case the installed DDGS version
        # returns an iterable.
        results = list(results)

        print(f"Search returned {len(results)} results.")

        return results

    except Exception as e:

        print("DDGS SEARCH ERROR:")
        print(type(e).__name__)
        print(str(e))

        return []


# ============================================================
# COLLECT RESEARCH
# ============================================================

def collect_research(topic: str, number_of_sources: int = 5):
    """
    Collect web sources for the research topic.

    We use ONE search instead of multiple searches.
    This reduces the possibility of DuckDuckGo rate limits.
    """

    if not topic or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    try:
        number_of_sources = int(number_of_sources)
    except (ValueError, TypeError):
        number_of_sources = 5

    number_of_sources = max(1, min(number_of_sources, 10))

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = search_web(
        topic.strip(),
        max_results=number_of_sources
    )

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "Web search returned no results.\n\n"
            "Possible reasons:\n"
            "1. DDGS/DuckDuckGo is temporarily unavailable.\n"
            "2. Your network or hosting platform blocked the request.\n"
            "3. You reached a search rate limit.\n"
            "4. The DDGS package needs to be updated.\n\n"
            "Try again in a few seconds or test another topic."
        )

    # --------------------------------------------------------
    # Remove duplicate URLs
    # --------------------------------------------------------

    unique_results = []
    seen_urls = set()

    for result in results:

        if not isinstance(result, dict):
            continue

        title = result.get("title", "").strip()
        url = result.get("href", "").strip()
        body = result.get("body", "").strip()

        # Need at least some useful information.
        if not title and not body:
            continue

        # Avoid duplicate URLs.
        if url:

            if url in seen_urls:
                continue

            seen_urls.add(url)

        unique_results.append(
            {
                "title": title or "Untitled source",
                "href": url,
                "body": body or "No description available."
            }
        )

        if len(unique_results) >= number_of_sources:
            break

    if not unique_results:

        raise RuntimeError(
            "DDGS returned data, but no usable research sources "
            "could be extracted."
        )

    return unique_results


# ============================================================
# BUILD RESEARCH TEXT
# ============================================================

def build_research_text(sources):
    """
    Convert search results into text that can be given to CrewAI.
    """

    research_parts = []

    for i, source in enumerate(sources, start=1):

        title = source.get(
            "title",
            "No title"
        )

        url = source.get(
            "href",
            "No URL"
        )

        snippet = source.get(
            "body",
            "No description available."
        )

        research_parts.append(
            f"""
SOURCE {i}

Title:
{title}

URL:
{url}

Information:
{snippet}

-----------------------------------
"""
        )

    return "\n".join(research_parts)


# ============================================================
# MAIN RESEARCH FUNCTION
# ============================================================

def run_research(
    topic: str,
    number_of_sources: int,
    api_key: str
):
    """
    Run the complete research pipeline.

    1. Search web
    2. Collect sources
    3. Create Groq LLM
    4. Create one CrewAI agent
    5. Generate research report
    """

    # --------------------------------------------------------
    # Validate topic
    # --------------------------------------------------------

    if not topic or not topic.strip():

        raise ValueError(
            "Please enter a research topic."
        )

    # --------------------------------------------------------
    # Validate API key
    # --------------------------------------------------------

    if not api_key or not api_key.strip():

        raise ValueError(
            "Groq API key is missing."
        )

    # --------------------------------------------------------
    # STEP 1
    # Search web
    # --------------------------------------------------------

    sources = collect_research(
        topic=topic,
        number_of_sources=number_of_sources
    )

    # --------------------------------------------------------
    # STEP 2
    # Prepare research
    # --------------------------------------------------------

    research_text = build_research_text(
        sources
    )

    # --------------------------------------------------------
    # STEP 3
    # Create LLM
    # --------------------------------------------------------

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key.strip(),
        temperature=0.2,
        reasoning_effort="low",
    )

    # --------------------------------------------------------
    # STEP 4
    # Create agent
    # --------------------------------------------------------

    researcher = Agent(

        role="Senior Research Analyst",

        goal=(
            "Create an accurate, well-structured and "
            "easy-to-understand research report using "
            "the research information provided."
        ),

        backstory=(
            "You are a professional research analyst. "
            "You carefully analyze the provided sources, "
            "avoid inventing information, and clearly "
            "explain important findings."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=1,
    )

    # --------------------------------------------------------
    # STEP 5
    # Create task
    # --------------------------------------------------------

    task = Task(

        description=f"""
Research topic:

{topic}

============================================================
RESEARCH SOURCES
============================================================

{research_text}

============================================================
IMPORTANT INSTRUCTIONS
============================================================

Use ONLY the research information provided above.

Do NOT invent facts.

Do NOT invent statistics.

Do NOT invent quotations.

Do NOT create fake sources.

Do NOT pretend that you personally visited websites.

If information is missing, clearly say that the
available research information is limited.

Use simple language that a beginner can understand.

Include the actual URLs from the research sources.

============================================================
REPORT STRUCTURE
============================================================

# Research Report

## 1. Executive Summary

Provide a short summary of the research.

## 2. Introduction

Explain the topic and why it is important.

## 3. Key Findings

List the most important findings.

## 4. Detailed Analysis

Explain the available research information in detail.

## 5. Benefits / Opportunities

Explain important benefits or opportunities
when applicable.

## 6. Challenges / Limitations

Explain important challenges, risks, or limitations.

## 7. Current Developments

Discuss recent developments only when supported
by the provided sources.

## 8. Conclusion

Provide a short evidence-based conclusion.

## 9. Sources

List all sources.

For every source include:

- Source title
- URL

============================================================

Make the report professional, factual,
well organized, and easy for a beginner to understand.
""",

        expected_output=(
            "A complete research report containing an "
            "executive summary, introduction, key findings, "
            "detailed analysis, benefits, challenges, "
            "current developments, conclusion, and source URLs."
        ),

        agent=researcher,
    )

    # --------------------------------------------------------
    # STEP 6
    # Create Crew
    # --------------------------------------------------------

    crew = Crew(

        agents=[
            researcher
        ],

        tasks=[
            task
        ],

        verbose=False,
    )

    # --------------------------------------------------------
    # STEP 7
    # Run CrewAI
    # --------------------------------------------------------

    try:

        result = crew.kickoff()

        return result

    except Exception as e:

        raise RuntimeError(
            "CrewAI failed while generating the report.\n\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {str(e)}"
        ) from e
