# ============================================================
# research_agent.py
# AI Research Agent
# CrewAI + Groq + DDGS
# ============================================================


# ============================================================
# STEP 1: LITELLM COMPATIBILITY FIX
# ============================================================

import litellm

# IMPORTANT:
# CrewAI can add provider-specific parameters such as
# cache_breakpoint.
#
# Groq may reject unsupported parameters.
# This tells LiteLLM to remove unsupported parameters
# before sending the request to Groq.

litellm.drop_params = True


# ============================================================
# STEP 2: CREWAI CACHE COMPATIBILITY
# ============================================================

try:

    import crewai.llms.cache as crewai_cache

    # Some CrewAI versions expect this function.
    # If it does not exist, create a safe fallback.

    if not hasattr(
        crewai_cache,
        "mark_cache_breakpoint"
    ):

        crewai_cache.mark_cache_breakpoint = (
            lambda msg: msg
        )

except Exception:

    # If this particular CrewAI cache module is not
    # available in the installed version, continue.
    pass


# ============================================================
# STEP 3: IMPORTS
# ============================================================

from crewai import Agent, Crew, Task, LLM
from ddgs import DDGS


# ============================================================
# STEP 4: WEB SEARCH FUNCTION
# ============================================================

def search_web(
    query: str,
    max_results: int = 5
):
    """
    Search the internet using DDGS.

    Several backends are attempted so that one failed
    search provider does not immediately stop the application.
    """

    # Check query
    if not query or not query.strip():
        return []

    # Keep source count reasonable
    max_results = max(
        1,
        min(int(max_results), 10)
    )

    # Search backends
    backends = [
        "bing",
        "brave",
        "duckduckgo",
        "yahoo",
    ]

    # Try each backend
    for backend in backends:

        try:

            print(
                f"Trying search backend: {backend}"
            )

            searcher = DDGS(
                timeout=20
            )

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
                f"{backend} failed: "
                f"{type(e).__name__}: {e}"
            )

            # Try the next backend
            continue

    # Nothing worked
    return []


# ============================================================
# STEP 5: COLLECT RESEARCH
# ============================================================

def collect_research(
    topic: str,
    number_of_sources: int
):
    """
    Search the web and collect unique sources.
    """

    # Validate topic
    if not topic or not topic.strip():

        raise ValueError(
            "Research topic cannot be empty."
        )

    # Validate number of sources
    try:

        number_of_sources = int(
            number_of_sources
        )

    except (
        ValueError,
        TypeError
    ):

        number_of_sources = 5

    # Keep source number between 1 and 10
    number_of_sources = max(
        1,
        min(number_of_sources, 10)
    )

    # --------------------------------------------------------
    # Perform web search
    # --------------------------------------------------------

    results = search_web(
        query=topic,
        max_results=number_of_sources
    )

    # --------------------------------------------------------
    # Check search result
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "Unable to retrieve web research.\n\n"
            "The configured web search providers did not "
            "return any results. Please try again."
        )

    # --------------------------------------------------------
    # Remove duplicate URLs
    # --------------------------------------------------------

    unique_results = []

    seen_urls = set()

    for result in results:

        # Make sure result is a dictionary
        if not isinstance(
            result,
            dict
        ):
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

        # Clean values
        title = str(title).strip()
        url = str(url).strip()
        body = str(body).strip()

        # Skip completely empty results
        if not title and not body:
            continue

        # Remove duplicate URLs
        if url:

            if url in seen_urls:
                continue

            seen_urls.add(url)

        # Store clean result
        unique_results.append(
            {
                "title": title or "Untitled source",
                "href": url,
                "body": body or "No description available.",
            }
        )

        # Stop when enough sources are collected
        if len(unique_results) >= number_of_sources:
            break

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not unique_results:

        raise RuntimeError(
            "Search completed, but no usable sources "
            "were found."
        )

    return unique_results


# ============================================================
# STEP 6: BUILD RESEARCH TEXT
# ============================================================

def build_research_text(
    sources
):
    """
    Convert web search results into text for the LLM.
    """

    research_text = ""

    for i, source in enumerate(
        sources,
        start=1
    ):

        title = source.get(
            "title",
            "No title"
        )

        url = source.get(
            "href",
            "No URL"
        )

        information = source.get(
            "body",
            "No information available."
        )

        research_text += f"""

============================================================
SOURCE {i}
============================================================

Title:
{title}

URL:
{url}

Information:
{information}

------------------------------------------------------------
"""

    return research_text


# ============================================================
# STEP 7: MAIN RESEARCH FUNCTION
# ============================================================

def run_research(
    topic: str,
    number_of_sources: int,
    api_key: str
):
    """
    Complete research pipeline.

    Flow:

    User topic
        ↓
    Web search
        ↓
    Research sources
        ↓
    CrewAI Agent
        ↓
    Groq LLM
        ↓
    Research report
    """

    # ========================================================
    # VALIDATE TOPIC
    # ========================================================

    if not topic or not topic.strip():

        raise ValueError(
            "Please enter a research topic."
        )

    # ========================================================
    # VALIDATE API KEY
    # ========================================================

    if not api_key or not api_key.strip():

        raise ValueError(
            "Groq API key is missing."
        )

    # ========================================================
    # STEP 1: SEARCH WEB
    # ========================================================

    sources = collect_research(
        topic=topic,
        number_of_sources=number_of_sources
    )

    # ========================================================
    # STEP 2: PREPARE RESEARCH
    # ========================================================

    research_text = build_research_text(
        sources
    )

    # ========================================================
    # STEP 3: CREATE GROQ LLM
    # ========================================================

    llm = LLM(

        # Current Groq model
        model="groq/openai/gpt-oss-120b",

        # API key from Streamlit Secrets
        api_key=api_key.strip(),

        # Low temperature gives more factual output
        temperature=0.2,
    )

    # ========================================================
    # STEP 4: CREATE CREWAI AGENT
    # ========================================================

    researcher = Agent(

        role="Senior Research Analyst",

        goal=(
            "Create an accurate, well-structured and "
            "easy-to-understand research report using "
            "the provided web research."
        ),

        backstory=(
            "You are a professional research analyst. "
            "You carefully analyze research sources, "
            "avoid inventing facts, and explain "
            "important findings clearly."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=1,
    )

    # ========================================================
    # STEP 5: CREATE TASK
    # ========================================================

    task = Task(

        description=f"""

You are writing a research report.

RESEARCH TOPIC:

{topic}


============================================================
WEB RESEARCH
============================================================

The following information was collected from web search:

{research_text}


============================================================
IMPORTANT RULES
============================================================

Use the supplied research as your evidence.

1. Do NOT invent facts.

2. Do NOT invent statistics.

3. Do NOT invent quotations.

4. Do NOT create fake sources.

5. Do NOT create fake URLs.

6. Do NOT claim that you personally visited a website.

7. If the available information is limited,
   clearly mention the limitation.

8. Keep the report factual.

9. Use simple language.

10. Include the actual URLs provided by the search.


============================================================
REPORT STRUCTURE
============================================================

# Research Report


## 1. Executive Summary

Give a short summary of the research.


## 2. Introduction

Explain the topic and why it is important.


## 3. Key Findings

List the most important findings.


## 4. Detailed Analysis

Explain the available information in detail.


## 5. Benefits / Opportunities

Explain important benefits or opportunities,
when applicable.


## 6. Challenges / Limitations

Explain important challenges, risks,
and limitations.


## 7. Current Developments

Discuss recent developments only when they are
supported by the supplied research.


## 8. Conclusion

Give a short conclusion based on the evidence.


## 9. Sources

List every source.

For every source include:

- Source title
- URL


============================================================

Make the report professional, factual,
well organized, and easy for a beginner
to understand.

""",

        expected_output=(
            "A complete research report containing "
            "an executive summary, introduction, "
            "key findings, detailed analysis, "
            "benefits, challenges, current "
            "developments, conclusion, and "
            "source URLs."
        ),

        agent=researcher,
    )

    # ========================================================
    # STEP 6: CREATE CREW
    # ========================================================

    crew = Crew(

        agents=[
            researcher
        ],

        tasks=[
            task
        ],

        verbose=False,
    )

    # ========================================================
    # STEP 7: RUN CREWAI
    # ========================================================

    try:

        result = crew.kickoff()

        return result

    except Exception as e:

        # Print complete error in server logs
        print(
            "=================================================="
        )

        print(
            "CREWAI / GROQ ERROR"
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        print(
            "=================================================="
        )

        # Give Streamlit a readable error
        raise RuntimeError(
            "CrewAI/Groq failed while generating "
            "the research report.\n\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {str(e)}"
        ) from e
