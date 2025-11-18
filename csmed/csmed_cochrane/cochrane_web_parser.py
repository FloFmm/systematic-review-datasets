import contextlib
import logging
import re
import time
from typing import Optional, Union
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PlaywrightResponse:
    """Minimal response object to mimic requests.Response"""
    def __init__(self, status_code: int, content: bytes):
        self.status_code = status_code
        self.content = content
    
    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

def get_request(url: str) -> PlaywrightResponse:
    """
    Drop-in replacement for requests.get(url) using Playwright.
    Returns an object with .status_code and .content like requests.Response.
    Ignores headers — uses a real browser to bypass Cloudflare.
    """
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)  # waits for network to settle

        html = page.content()
        browser.close()
        # Always return status_code 200 if page loaded
        return PlaywrightResponse(status_code=200, content=html.encode("utf-8"))

def get_safe_text(element) -> str:
    return element.text if element else ""


def get_safe_soup(url: str, headers: dict[str, str]) -> Optional[BeautifulSoup]:
    """Returns a BeautifulSoup object or None. Tries three times and waits 60 seconds between each try."""
    wait_time = 60
    for _ in range(3):
        r = get_request(url)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
        else:
            wait_time *= 2
            logger.error(
                f"Request failed with status code {r.status_code}. Waiting {wait_time} seconds before trying again."
            )
            time.sleep(wait_time)


# def get_safe_soup(url: str, headers: dict[str, str]) -> Optional[BeautifulSoup]:
#     """
#     Playwright-based HTML fetcher with retries.
#     Supports full custom headers.
#     """
#     wait_time = 60

#     for attempt in range(3):
#         try:
#             with sync_playwright() as p:
#                 print(f"scraping url {url}")
#                 browser = p.firefox.launch(headless=True)
#                 context = browser.new_context()
#                 # context = browser.new_context(extra_http_headers=headers)

#                 # Playwright requires UA to be set separately
#                 # if "User-Agent" in headers:
#                 #     context.set_extra_http_headers({
#                 #         k: v for k, v in headers.items() if k.lower() != "user-agent"
#                 #     })
#                 #     context = browser.new_context(
#                 #         user_agent=headers["User-Agent"],
#                 #         extra_http_headers={
#                 #             k: v for k, v in headers.items() if k.lower() != "user-agent"
#                 #         }
#                 #     )
    
#                 page = context.new_page()
#                 page.goto(url)

#                 html = page.content()
#                 browser.close()
#                 return BeautifulSoup(html, "html.parser")

#         except Exception as e:
#             logger.error(
#                 f"Attempt {attempt+1} failed: {e}. Waiting {wait_time} seconds before retrying."
#             )
#             time.sleep(wait_time)
#             wait_time *= 2

#     return None

def _find_doi(urls: list[str]) -> Optional[str]:
    """searches for DOI in urls"""
    doi = None
    for url in urls:
        if x := re.search(r"https://doi.org/(.*)", url):
            doi = f"https://doi.org/{str(x[1])}"
    return doi


def _find_pubmed_id(citation_text: str, urls: list[str]) -> Optional[str]:
    """Find PubMed ID in citation text. Example: [PUBMED: 123456].
    Alternatively there might be a link."""
    _pubmed_id = (
        str(x[1]) if (x := re.search(r"\[PUBMED: (\d+)\]", citation_text)) else None
    )
    if not _pubmed_id:
        for url in urls:
            if x := re.search(r"https://www.ncbi.nlm.nih.gov/pubmed/(\d+)", url):
                _pubmed_id = str(x[1])
    return _pubmed_id


def get_citation_data(citation) -> dict[str, Union[Optional[str], list[str]]]:
    _reference_id = citation.attrs.get("id")

    citation_class = [x.attrs.get("class") for x in citation.find_all()][0]
    citation_text = citation.text

    title = get_safe_text(citation.find("span", {"class": "citation-title"}))
    journal = get_safe_text(citation.find("span", {"class": "citation"}))
    year = get_safe_text(citation.find("span", {"class": "pubYear"}))
    volume = get_safe_text(citation.find("span", {"class": "volume"}))

    citation_div = citation.find("div", class_=citation_class)
    first_child = citation_div.find_all("span")[0]
    last_child = citation_div.find_all("span")[-1]

    authors = citation_div.text[: citation_div.text.index(first_child.text)]
    # fixme get text after last child
    citation_footer = citation_div.text[
        citation_div.text.rindex(last_child.text) + len(last_child.text) :
    ]

    urls = [
        citation_link.attrs.get("href")
        for citation_link in citation.find_all("a", {"class": "citation-link"})
    ]
    pubmed_id = _find_pubmed_id(citation_text, urls=urls)
    doi = _find_doi(urls)
    return {
        "reference_id": _reference_id,
        "authors": authors,
        "title": title,
        "journal": journal,
        "year": year,
        "volume": volume,
        "citation_footer": citation_footer,
        "citation_type": citation_class,
        "citation": citation_text,
        "urls": urls,
        "pubmed_id": pubmed_id,
        "doi": doi,
    }


def parse_cochrane_review_reference_section(
    references_section, reference_class: str, reference_category: str
) -> pd.DataFrame:
    _studies = []
    for _study in references_section.find_all(
        "div", {"class": f"bibliographies {reference_class}"}
    ):
        header = _study.find("div", {"class": "reference-title-banner"}).text
        _study_id = _study.attrs.get("id")
        for citation in _study.find_all("div", {"class": "bibliography-section"}):
            try:
                citation_data = get_citation_data(citation)
                citation_data["header"] = header
                citation_data["reference_type"] = reference_category
                citation_data["study_id"] = _study_id
                _studies.append(citation_data)
            except IndexError as e:
                logger.error(f"Error parsing citation: {e}")
                continue

    return pd.DataFrame(_studies)


def get_exclusion_reasons(exclusion_section, df: pd.DataFrame) -> pd.DataFrame:
    df["exclusion_reason"] = ""
    logging.debug(
        f"Found {len(list(exclusion_section.find_all('tr')))} documents with exclusion reasons"
    )
    for _excluded_paper in exclusion_section.find("tbody").find_all("tr"):
        study_id = (
            list(_excluded_paper.find_all("td"))[0]
            .find("a")
            .attrs.get("href")
            .split("#")[-1]
        )
        exclusion_reason = list(_excluded_paper.find_all("td"))[1].text

        df.loc[df["study_id"] == study_id, "exclusion_reason"] = exclusion_reason

    return df


def safe_convert_to_float(number):
    if not number:
        return None
    try:
        return float(number)
    except ValueError:
        number = number.replace("‐", "-").strip("'")
        try:
            return float(number)
        except ValueError:
            return None


def get_effect_size(
    text: str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not text:
        return None, None, None
    if len(text.split("[")) == 1:
        return None, None, None
    effect_size = text.split("[")[0].strip()
    try:
        lower_ci = text.split("[")[1].split(",")[0].strip()
    except IndexError:
        lower_ci = None

    try:
        upper_ci = text.split("[")[1].split(",")[1].split("]")[0].strip()
    except IndexError:
        upper_ci = None

    return (
        safe_convert_to_float(effect_size),
        safe_convert_to_float(lower_ci),
        safe_convert_to_float(upper_ci),
    )


def postprocess_comparison_table(df):
    current_outcome_name = ""
    for index, row in df.iterrows():
        _id = row["Outcome or subgroup title"].split(" ")[0]
        if "." in _id:
            outcome_id = _id.split(".")[0]
            subgroup_id = _id.split(".")[1]
        else:
            outcome_id = _id
            subgroup_id = None
        df.loc[index, "outcome_id"] = outcome_id
        df.loc[index, "subgroup_id"] = subgroup_id

        title = " ".join(row["Outcome or subgroup title"].split(" ")[1:])
        if not subgroup_id:
            current_outcome_name = title[:-16].strip()
            df.loc[index, "outcome_name"] = current_outcome_name
            df.loc[index, "subgroup_name"] = None
        else:
            df.loc[index, "outcome_name"] = current_outcome_name
            df.loc[index, "subgroup_name"] = title

    return df


def parse_data_and_analyses_section(
    soup: Optional[BeautifulSoup], review_id: str
) -> pd.DataFrame:
    if not soup:
        return pd.DataFrame()

    data_section = soup.find("section", {"id": "dataAndAnalyses"})

    out_df = pd.DataFrame()
    for comparison_id, comparison in enumerate(
        data_section.find_all("div", {"class": "table"})
    ):
        comparison_name = (
            comparison.find("div", {"class": "table-heading"})
            .find("span", {"class": "table-title"})
            .text.strip()
        )
        try:
            df = pd.read_html(
                str(comparison.find("table", {"class": "comparison"})),
                flavor="html5lib",
            )[0]

            if df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(list(range(1, df.columns.nlevels)))
        except ValueError:
            logger.error(f"Error reading table {comparison_id + 1} for {review_id}")
            continue

        df = df.dropna(axis=0, how="all").reset_index(drop=True)
        df["comparison_name"] = comparison_name
        df["comparison_id"] = comparison_id + 1

        has_effect_size = True
        with contextlib.suppress(AttributeError):
            if "Effect size" not in df.columns:
                has_effect_size = False
                continue
            effects = df["Effect size"].apply(get_effect_size).apply(pd.Series)
            df["effect_size"] = effects[0]
            df["lower_ci"] = effects[1]
            df["upper_ci"] = effects[2]

        if has_effect_size:
            df = df[~df[df.columns[0]].str.startswith("Open in figure viewer")]
            df = postprocess_comparison_table(df)
        out_df = pd.concat([out_df, df])

    return out_df.reset_index(drop=True)


def parse_cochrane_references(soup: Optional[BeautifulSoup]) -> pd.DataFrame:
    if not soup:
        return pd.DataFrame()

    references_section = soup.find("section", {"class": "references bibliographies"})

    reference_types = {
        "included": "references_includedStudies",
        "excluded": "references_excludedStudies",
        "additional": "references_additionalReferences",
        "awaiting_assessment": "references_awaitingAssessmentStudies",
        "other_versions": "references_otherVersions",
    }
    out_df = pd.DataFrame()
    for reference_category, reference_class in reference_types.items():
        df = parse_cochrane_review_reference_section(
            references_section=references_section,
            reference_class=reference_class,
            reference_category=reference_category,
        )

        if reference_category == "excluded" and soup.find(
            "section", {"class": "characteristicsOfExcludedStudies"}
        ):
            df = get_exclusion_reasons(
                soup.find("section", {"class": "characteristicsOfExcludedStudies"}),
                df,
            )

        out_df = pd.concat([out_df, df])

    return out_df.reset_index(drop=True)


def parse_search_strategy(soup: BeautifulSoup) -> dict[str, str]:
    """Parse appendix page containing search queries used in the systematic review.
    It returns the search query from the EMBASE databse."""
    appendices = soup.find("section", {"class": "appendices"})

    if not appendices:
        return {}
    appendices_dict = {}
    for section in appendices.find_all("section"):
        # find the highest level heading and use it as key
        try:
            key = section.find(re.compile("^h[1-6]$")).text.strip()
            value = section.text.replace(key, "").strip()
            appendices_dict[key] = value
        except AttributeError:
            continue

    return appendices_dict


def parse_eligibility_criteria(soup: BeautifulSoup) -> dict[str, str]:
    """Parse methods section containing eligibility criteria used in the systematic review."""
    methods = soup.find("section", {"class": "methods"})
    if not methods:
        return {}
    criteria = methods.find(
        re.compile("^h[1-6]$"),
        text=re.compile(
            "Criteria for considering (studies|reviews) for (this review|inclusion|this synthesis)",
            re.IGNORECASE,
        ),
    ).parent

    criteria_dict = {}
    for section in criteria.find_all("section"):
        criteria_text = ""
        category = section.find("h4")
        if category:
            category = category.text.strip()
            criteria_text = section.find_all("p")
            criteria_text = " ".join([p.text.strip() for p in criteria_text])
        else:
            category = section.find("h5")
            if category:
                category = category.text.strip()
                criteria_text = section.text.replace(category, "").strip()
        criteria_dict[category] = criteria_text

    return criteria_dict
