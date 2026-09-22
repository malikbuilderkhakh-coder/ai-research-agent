# ============================================================
# research_agent.py
# CrewAI + Groq + Web Research
# ============================================================

import crewai.llms.cache as crewai_cache

# CrewAI compatibility workaround
if not hasattr(crewai_cache, "mark_cache_breakpoint"):
    crewai_cache.mark_cache_breakpoint = lambda msg: msg


from crewai import Agent, Crew, Task, LLM
from ddgs import DDGS


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(query: str, max_results: int = 5):

    if not query or not query.strip():
        return []

    max_results = max(1, min(int(max_results), 10))

    # Try several search engines.
    # If one fails, the next one is attempted.
    backends = [
        "bing",
        "brave",
        "duckduckgo",
        "yahoo",
    ]

    for backend in backends:

        try:

            print(f"Trying search backend: {backend}")

            searcher = DDGS(timeout=20)

            results = searcher.text(
                query=query.strip(),
                region="us-en",
                safesearch="moderate",
                max_results=max_results,
                backend=backend,
            )

            if results:

                results = list(results)

                if len(results) > 0:

                    print(
                        f"{backend} returned "
                        f"{len(results)} results."
                    )

                    return results

        except Exception as e:

            print(
                f"{backend} search failed: "
                f"{type(e).__name__}: {e}"
            )

            continue

    return []


# ============================================================
# COLLECT RESEARCH
# ============================================================

def collect_research(
    topic: str,
    number_of_sources: int
):

    if not topic or not topic.strip():

        raise ValueError(
            "Research topic cannot be empty."
        )

    number_of_sources = max(
        1,
        min(int(number_of_sources), 10)
    )

    # One search only.
    # This reduces rate-limit problems.
    results = search_web(
        topic,
        number_of_sources
    )

    if not results:

        raise RuntimeError(
            "Unable to retrieve web research at this time. "
            "All configured search backends failed. "
            "Please try again later."
        )

    # --------------------------------------------------------
    # Remove duplicate URLs
    # --------------------------------------------------------

    unique_results = []

    seen_urls = set()

    for result in results:

        if not isinstance(result, dict):
            continue

        title = result.get(
            "title",
            "No title"
        )

        url = result.get(
            "href",
            ""
        )

        body = result.get(
            "body",
            "No description available."
        )

        # Skip completely empty results
        if not title and not body:
            continue

        # Remove duplicates
        if url:

            if url in seen_urls:
                continue

            seen_urls.add(url)

        unique_results.append(
            {
                "title": title,
                "href": url,
                "body": body,
            }
        )

        if len(unique_results) >= number_of_sources:
            break

    return unique_results


# ============================================================
# BUILD RESEARCH TEXT
# ============================================================

def build_research_text(sources):

    research_text = ""

    for i, source in enumerate(
        sources,
        start=1
    ):

        research_text += f"""
SOURCE {i}

Title:
{source.get("title", "No title")}

URL:
{source.get("href", "No URL")}

Information:
{source.get("body", "No information")}

--------------------------------------------------
"""

    return research_text


# ============================================================
# MAIN RESEARCH FUNCTION
# ============================================================

def run_research(
    topic: str,
    number_of_sources: int,
    api_key: str
):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not topic or not topic.strip():

        raise ValueError(
            "Please enter a research topic."
        )

    if not api_key or not api_key.strip():

        raise ValueError(
            "Groq API key is missing."
        )

    # --------------------------------------------------------
    # STEP 1: Web research
    # --------------------------------------------------------

    sources = collect_research(
        topic=topic,
        number_of_sources=number_of_sources
    )

    # --------------------------------------------------------
    # STEP 2: Prepare research
    # --------------------------------------------------------

    research_text = build_research_text(
        sources
    )

    # --------------------------------------------------------
    # STEP 3: Groq LLM
    # --------------------------------------------------------

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key.strip(),
        temperature=0.2,
        reasoning_effort="low",
    )

    # --------------------------------------------------------
    # STEP 4: Agent
    # --------------------------------------------------------

    researcher = Agent(

        role="Senior Research Analyst",

        goal=(
            "Create an accurate and well-structured "
            "research report using the supplied "
            "web research."
        ),

        backstory=(
            "You are a professional research analyst. "
            "You analyze the provided sources carefully "
            "and never invent unsupported facts."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=1,
    )

    # --------------------------------------------------------
    # STEP 5: Task
    # --------------------------------------------------------

    task = Task(

        description=f"""

Research topic:

{topic}


The following information was collected from
web search:

==================================================
{research_text}
==================================================


IMPORTANT RULES:

1. Use only the information supplied above.
2. Do not invent facts.
3. Do not invent statistics.
4. Do not invent quotations.
5. Do not create fake sources.
6. Do not claim that you personally visited websites.
7. Clearly mention limitations.
8. Include the source URLs.


Write the report using this structure:


# Research Report

## 1. Executive Summary

Give a short summary.


## 2. Introduction

Explain the topic.


## 3. Key Findings

List the major findings.


## 4. Detailed Analysis

Explain the information in detail.


## 5. Benefits / Opportunities

Discuss benefits when applicable.


## 6. Challenges / Limitations

Discuss challenges and limitations.


## 7. Current Developments

Discuss recent developments only when
supported by the supplied research.


## 8. Conclusion

Give a short evidence-based conclusion.


## 9. Sources

List every source with:

- Title
- URL


Make the report professional,
factual and easy for a beginner to understand.

""",

        expected_output=(
            "A complete research report with "
            "summary, introduction, findings, "
            "analysis, benefits, challenges, "
            "current developments, conclusion "
            "and source URLs."
        ),

        agent=researcher,
    )

    # --------------------------------------------------------
    # STEP 6: Crew
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
    # STEP 7: Generate report
    # --------------------------------------------------------

    try:

        result = crew.kickoff()

        return result

    except Exception as e:

        raise RuntimeError(
            "CrewAI/Groq failed while generating "
            "the research report.\n\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {str(e)}"
        ) from e
