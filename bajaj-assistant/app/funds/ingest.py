"""PDF -> chunked text (Stage 3).

DESIGN NOTE — section-boundary detection deviates from a naive "scan every
page for a fund name heading" approach: this factsheet (like most AMC
factsheets) has an actual table of contents on its second page mapping
every section title to its exact start page (e.g. "Bajaj Finserv Large Cap
Fund 16"). We parse that TOC instead of pattern-matching headings page by
page, because naive scanning has a real, confirmed failure mode here: the
4-page "Fund Snapshot" summary table (pages 12-15) lists rows like "Bajaj
Finserv Nifty 50 Index Fund Category Index Fund" — text that would
false-positive as that fund's dedicated section start under a heading-scan
rule, when it's actually a general cross-fund summary table. The TOC gives
exact, unambiguous boundaries and degrades to a loud warning (not a silent
mis-segmentation) if a differently-formatted factsheet doesn't have one in
the expected place.

Everything that isn't a canonical fund's own section (MD/CIO letters,
macro outlook, "how to read a factsheet", the Fund Snapshot table,
cross-fund Performance/SIP/PRC/Risk-o-meter reference sections at the tail)
is tagged fund_name="general" — these answer fund-agnostic questions
("what does expense ratio mean") and shouldn't be tied to one fund.
"""

import re

from pypdf import PdfReader

# The 24 funds in the factsheet, grouped by category for readability.
# Hardcoded per Stage 3 spec rather than waiting on PDF extraction to
# discover it — extraction is instead diff-checked against this list as a
# sanity check (see _print_fund_diff below).
CANONICAL_FUNDS = [
    # Equity
    "Bajaj Finserv Large Cap Fund",
    "Bajaj Finserv Large and Mid Cap Fund",
    "Bajaj Finserv Consumption Fund",
    "Bajaj Finserv Small Cap Fund",
    "Bajaj Finserv ELSS Tax Saver Fund",
    "Bajaj Finserv Flexi Cap Fund",
    "Bajaj Finserv Multi Cap Fund",
    "Bajaj Finserv Healthcare Fund",
    "Bajaj Finserv Banking and Financial Services Fund",
    # Hybrid
    "Bajaj Finserv Balanced Advantage Fund",
    "Bajaj Finserv Arbitrage Fund",
    "Bajaj Finserv Multi Asset Allocation Fund",
    "Bajaj Finserv Equity Savings Fund",
    # Fixed Income / Debt
    "Bajaj Finserv Liquid Fund",
    "Bajaj Finserv Low Duration Fund",
    "Bajaj Finserv Overnight Fund",
    "Bajaj Finserv Money Market Fund",
    "Bajaj Finserv Gilt Fund",
    "Bajaj Finserv Banking and PSU Fund",
    # Passive
    "Bajaj Finserv Nifty 50 Index Fund",
    "Bajaj Finserv Nifty Next 50 Index Fund",
    "Bajaj Finserv Nifty 1D Rate Liquid ETF - Growth",
    "Bajaj Finserv Nifty 50 ETF",
    "Bajaj Finserv Nifty Bank ETF",
]

GENERAL_LABEL = "general"

# Chunk sizing: roughly 600-900 characters, ~100 character overlap between
# consecutive chunks in the same section.
CHUNK_TARGET_MIN_CHARS = 600
CHUNK_TARGET_MAX_CHARS = 900
CHUNK_OVERLAP_CHARS = 100

_TOC_LINE_RE = re.compile(r"^(.*\S)\s+(\d{1,3})$")


def build_chunks(pdf_path: str) -> list[dict]:
    """Extract, section, and chunk the factsheet PDF into retrievable pieces.

    Returns a list of {"text", "fund_name", "page_number", "chunk_id"}
    dicts. fund_name is either a name from CANONICAL_FUNDS or "general".
    """
    reader = PdfReader(pdf_path)
    pages_text = [reader.pages[i].extract_text() or "" for i in range(len(reader.pages))]
    num_pages = len(pages_text)

    toc_entries = _parse_toc(pages_text)
    _print_fund_diff(toc_entries)
    sections = _build_sections(toc_entries, num_pages)

    all_chunks = []
    seq_by_fund = {}
    for section in sections:
        section_pages = [
            (page_num, pages_text[page_num - 1])
            for page_num in range(section["start_page"], section["end_page"] + 1)
        ]
        section_chunks = _chunk_section(section_pages, section["fund_name"])
        for chunk in section_chunks:
            seq = seq_by_fund.get(chunk["fund_name"], 0)
            chunk["chunk_id"] = f"{_slugify(chunk['fund_name'])}-p{chunk['page_number']}-{seq}"
            seq_by_fund[chunk["fund_name"]] = seq + 1
        all_chunks.extend(section_chunks)

    print(f"built {len(all_chunks)} chunks across {len(sections)} sections "
          f"({num_pages} pages)")
    return all_chunks


def _parse_toc(pages_text: list[str]) -> list[tuple[str, int]]:
    """Find the table-of-contents page and parse it into (title, page) pairs.

    Looks for a page whose first non-empty line is exactly "Index" (as in
    this factsheet template) within the first 6 pages. Falls back to page 2
    with a loud warning if that pattern isn't found, rather than silently
    mis-segmenting the whole document.
    """
    toc_page_idx = None
    for i in range(min(6, len(pages_text))):
        lines = [line.strip() for line in pages_text[i].splitlines() if line.strip()]
        if lines and lines[0].lower() == "index":
            toc_page_idx = i
            break

    if toc_page_idx is None:
        toc_page_idx = 1 if len(pages_text) > 1 else 0
        print(f"WARNING: could not find a page starting with 'Index' in the "
              f"first 6 pages — falling back to page {toc_page_idx + 1} as "
              f"the table of contents. Section boundaries may be wrong; "
              f"inspect the output below closely.")

    entries = []
    for line in pages_text[toc_page_idx].splitlines():
        line = line.strip()
        if not line or line.lower() == "index":
            continue
        match = _TOC_LINE_RE.match(line)
        if not match:
            continue
        title, page_num = match.group(1).strip(), int(match.group(2))
        entries.append((title, page_num))

    entries.sort(key=lambda entry: entry[1])
    return entries


def _build_sections(toc_entries: list[tuple[str, int]], num_pages: int) -> list[dict]:
    """Turn ordered TOC entries into page ranges, classified fund vs general."""
    canonical_by_norm = {_normalize_title(name): name for name in CANONICAL_FUNDS}

    if not toc_entries:
        # No parseable TOC at all -- degrade to "everything is general"
        # rather than crash or guess at boundaries.
        return [{"title": "whole document", "fund_name": GENERAL_LABEL,
                 "start_page": 1, "end_page": num_pages}]

    sections = []
    if toc_entries[0][1] > 1:
        sections.append({
            "title": "front matter (before first TOC entry)",
            "fund_name": GENERAL_LABEL,
            "start_page": 1,
            "end_page": toc_entries[0][1] - 1,
        })

    for idx, (title, start_page) in enumerate(toc_entries):
        if idx + 1 < len(toc_entries):
            end_page = toc_entries[idx + 1][1] - 1
        else:
            end_page = num_pages
        end_page = max(end_page, start_page)

        fund_name = canonical_by_norm.get(_normalize_title(title), GENERAL_LABEL)
        sections.append({
            "title": title,
            "fund_name": fund_name,
            "start_page": start_page,
            "end_page": end_page,
        })

    return sections


def _print_fund_diff(toc_entries: list[tuple[str, int]]) -> None:
    """Sanity check: diff TOC-derived 'Bajaj Finserv ...' entries against
    CANONICAL_FUNDS. Prints the diff either way but never blocks ingestion
    on it — a mismatch is something to review, not a hard failure."""
    found_normalized = {
        _normalize_title(title)
        for title, _ in toc_entries
        if title.strip().lower().startswith("bajaj finserv")
    }
    canonical_by_norm = {_normalize_title(name): name for name in CANONICAL_FUNDS}

    missing = [name for norm, name in canonical_by_norm.items() if norm not in found_normalized]
    extra = [title for title, _ in toc_entries
             if title.strip().lower().startswith("bajaj finserv")
             and _normalize_title(title) not in canonical_by_norm]

    print("=" * 60)
    print("CANONICAL FUND LIST SANITY CHECK (vs. PDF table of contents)")
    print("=" * 60)
    if missing:
        print(f"in canonical list but NOT found in the PDF TOC ({len(missing)}):")
        for name in missing:
            print(f"  - {name}")
    else:
        print("all canonical funds found in the PDF table of contents.")
    if extra:
        print(f"found in the PDF TOC but NOT in the canonical list ({len(extra)}):")
        for name in extra:
            print(f"  - {name}")
    else:
        print("no unexpected 'Bajaj Finserv ...' TOC entries beyond the canonical list.")


def _chunk_section(section_pages: list[tuple[int, str]], fund_name: str) -> list[dict]:
    """Chunk one section's pages, tracking which page each chunk falls on.

    Pages are concatenated (so overlap/boundary-finding can cross a page
    break within the same section) while tracking character offsets, so
    each resulting chunk can still be attributed to the one page its
    content mostly starts on.
    """
    full_text = ""
    page_offsets = []  # (start_offset, end_offset, page_number)
    for page_num, text in section_pages:
        start = len(full_text)
        full_text += text + "\n"
        page_offsets.append((start, len(full_text), page_num))

    raw_chunks = _split_into_chunks(full_text)

    chunks = []
    for chunk_text, chunk_start in raw_chunks:
        page_num = next(
            (p for s, e, p in page_offsets if s <= chunk_start < e),
            page_offsets[-1][2],
        )
        chunks.append({
            "text": chunk_text,
            "fund_name": fund_name,
            "page_number": page_num,
        })
    return chunks


def _split_into_chunks(text: str) -> list[tuple[str, int]]:
    """Split text into ~600-900 char pieces with ~100 char overlap.

    Prefers splitting on paragraph breaks, then sentence boundaries, over
    a hard mid-sentence cut. Returns (chunk_text, start_offset) pairs.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_TARGET_MAX_CHARS, n)
        if end < n:
            search_from = start + CHUNK_TARGET_MIN_CHARS
            if search_from < end:
                boundary = _find_boundary(text, search_from, end)
                if boundary:
                    end = boundary
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, start))
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

    return chunks


def _find_boundary(text: str, search_from: int, end: int) -> int | None:
    """Look backward from `end` (down to `search_from`) for a paragraph
    break, then a sentence end, then a line break. None if none found."""
    window = text[search_from:end]

    para_break = window.rfind("\n\n")
    if para_break != -1:
        return search_from + para_break + 2

    for sep in (". ", "! ", "? "):
        idx = window.rfind(sep)
        if idx != -1:
            return search_from + idx + len(sep)

    line_break = window.rfind("\n")
    if line_break != -1:
        return search_from + line_break + 1

    return None


def _normalize_title(title: str) -> str:
    title = title.replace("–", "-").replace("—", "-")  # en/em dash
    title = re.sub(r"\s+", " ", title).strip().lower()
    return title


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "chunk"
