import re
import json
import csv
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup, Tag, NavigableString
from urllib.parse import urljoin, urlparse
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

BASE = "https://opinion.lawmaking.go.kr"
LIST_URL = "https://opinion.lawmaking.go.kr/nl4li/lsItptEmp/list?mode=all&pageIndex={page}&blockStartPage={block}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LawInterpCrawler/1.0; +https://gist.ac.kr)"
}

# -------------------------------
# Basic utility helpers
# -------------------------------
REL_MARKER_LINE_RE = re.compile(r"^(?:[<\[]|&lt;)?\s*관계\s*법령\s*(?:[>\]]|&gt;)?\s*$", re.I)
REL_MARKER_RE      = re.compile(r"(?:[<\[]|&lt;)?\s*관계\s*법령\s*(?:[>\]]|&gt;)?", re.I)

def clean_text(x: str) -> str:
    return " ".join((x or "").replace("\xa0", " ").split())

def clean_lines(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return [clean_text(ln) for ln in text.split("\n")]

def extract_text_with_breaks(node: Tag | BeautifulSoup) -> str:
    """Preserve line breaks for <br> tags and block-level elements."""
    if not node:
        return ""
    frag = BeautifulSoup(str(node), "lxml")
    for br in frag.find_all("br"):
        br.replace_with("\n")
    text = frag.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(clean_text(ln) for ln in text.split("\n")).strip()

def get_row_text_by_label(soup: BeautifulSoup, label_ko: str) -> str:
    ul = soup.select_one("ul.detailInfo")
    if not ul:
        return ""
    for li in ul.select("li"):
        lab = li.select_one("span.label_tit")
        if lab and lab.get_text(strip=True) == label_ko:
            spans = li.find_all("span")
            if len(spans) >= 2:
                return clean_text(spans[1].get_text(" ", strip=True))
    return ""

def get_id_from_url(url: str) -> str:
    try:
        path = urlparse(url).path.strip("/")
        return path.split("/")[-1]
    except Exception:
        return ""

def _find_subtitle_h3(soup: BeautifulSoup, keywords: list[str]) -> Tag | None:
    for h3 in soup.select("div.subTitle h3"):
        title = clean_text(h3.get_text(" ", strip=True))
        if any(k in title for k in keywords):
            return h3
    return None

def _get_section_block_text(soup: BeautifulSoup, keywords: list[str]) -> str:
    """
    Extract text content starting from the matching <div class="subTitle"><h3>keyword</h3></div> up to the next subTitle section.
    """
    h3 = _find_subtitle_h3(soup, keywords)
    if not h3:
        return ""
    sub_div = h3.find_parent("div", class_="subTitle")
    if not sub_div:
        return ""
    collected = []
    for sib in sub_div.next_siblings:
        if isinstance(sib, Tag):
            if sib.name == "div" and "subTitle" in (sib.get("class") or []):
                break
            collected.append(str(sib))
        elif isinstance(sib, NavigableString):
            if sib.strip():
                collected.append(str(sib))
    if not collected:
        return ""
    frag = BeautifulSoup("".join(collected), "lxml")
    return extract_text_with_breaks(frag)

# -------------------------------
# split content strictly by marker position
# -------------------------------
def split_reason_related_by_marker(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Rules:
      - Search for the '<Related Laws>' marker inside the 'Reason' section.
      - Everything before the marker is treated as reason.
      - Everything after the marker becomes related_laws if content exists.
      - If the marker appears at the end with no meaningful content after it,
        related_laws is set to an empty string.
      - No additional heuristics or fallback parsing are applied.
    """
    reason_block = _get_section_block_text(soup, ["이유"])
    if not reason_block:
        return "", ""  # Stop processing if the section does not exist

    lines = clean_lines(reason_block)
    marker_idx = None
    for i, ln in enumerate(lines):
        if REL_MARKER_RE.search(ln):
            marker_idx = i  # Use the last detected marker
    if marker_idx is None:
        # Leave related_laws empty if no marker is found
        return "\n".join(lines).strip(), ""

    head = "\n".join(lines[:marker_idx]).strip()
    tail = "\n".join(lines[marker_idx + 1:]).strip()

    # Treat as empty related_laws if no meaningful content exists after the marker
    if not tail or not tail.strip():
        return head, ""

    # Preserve everything after the marker as related_laws
    # Remove duplicate marker lines if they appear again in the tail
    tail_lines = [ln for ln in clean_lines(tail) if not REL_MARKER_LINE_RE.match(ln)]
    return head, "\n".join(tail_lines).strip()

# -------------------------------
# List/detail page parsing
# -------------------------------
async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url, headers=HEADERS) as r:
        r.raise_for_status()
        return await r.text()

async def parse_list_page(session: aiohttp.ClientSession, page: int, block_start: int = 1) -> list[dict]:
    url = LIST_URL.format(page=page, block=block_start)
    html = await fetch(session, url)
    soup = BeautifulSoup(html, "lxml")
    items = []
    tbody = soup.select_one(".gridCnt table tbody")
    if not tbody:
        return items
    for row in tbody.select("tr"):
        a = row.select_one("td:nth-of-type(2) a")
        dept_cell = row.select_one("td:nth-of-type(3)")
        if not a:
            continue
        detail_url = urljoin(BASE, a.get("href"))
        title = clean_text(a.get_text(" ", strip=True))
        department = ""
        if dept_cell:
            department = clean_text(dept_cell.get_text(" ", strip=True))
            if department.startswith("(") and department.endswith(")"):
                department = department[1:-1].strip()
        items.append({
            "id": get_id_from_url(detail_url),
            "url": detail_url,
            "title": title,
            "department": department,
        })
    return items

async def parse_detail(session: aiohttp.ClientSession, detail_url: str) -> dict:
    html = await fetch(session, detail_url)
    soup = BeautifulSoup(html, "lxml")

    # Metadata
    title = ""
    h2 = soup.select_one("#pageTitle")
    if h2:
        title = clean_text(h2.get_text(" ", strip=True))
    law_name    = get_row_text_by_label(soup, "법령")
    case_no     = get_row_text_by_label(soup, "안건번호")
    request_org = get_row_text_by_label(soup, "요청기관")
    reply_date  = get_row_text_by_label(soup, "회신일자")

    # Full body text
    main = soup.select_one("#containerWrap") or soup
    full_text = extract_text_with_breaks(main)

    # Split content strictly based on marker position
    reason, related_laws = split_reason_related_by_marker(soup)

    # Attachments
    attachments = []
    for a in soup.select("a[href*='/save/pdf/']"):
        name = clean_text(a.get_text(" ", strip=True) or "PDF")
        url = urljoin(BASE, a.get("href"))
        attachments.append({"name": "PDF" if "PDF" in name.upper() else name, "url": url})
    for a in soup.select("a[href*='/atchFile/download/']"):
        name = clean_text(a.get("title") or a.get_text(" ", strip=True) or "첨부")
        url = urljoin(BASE, a.get("href"))
        attachments.append({"name": name, "url": url})

    return {
        "id": get_id_from_url(detail_url),
        "url": detail_url,
        "title": title,
        "case_no": case_no,
        "request_org": request_org,
        "reply_date": reply_date,
        "law_name": law_name,
        "attachments": attachments,
        "reason": reason,
        "related_laws": related_laws,
        "content": full_text,  # for reference
    }

# -------------------------------
# Crawler
# -------------------------------
class Crawler:
    def __init__(
        self,
        start_page=1,
        end_page=None,
        block_start=1,
        out_jsonl="lawinterp.jsonl",
        out_csv="lawinterp.csv",
        concurrency=8,
    ):
        self.start_page = start_page
        self.end_page = end_page
        self.block_start = block_start
        self.out_jsonl = Path(out_jsonl)
        self.out_csv = Path(out_csv)
        self.sem = asyncio.Semaphore(concurrency)

    async def gather_list(self, session: aiohttp.ClientSession) -> list[dict]:
        all_items = []
        seen_ids: set[str] = set()
        page = self.start_page
        
        if self.end_page is not None:
            # Fixed page range mode
            pages = list(range(self.start_page, self.end_page + 1))
            for p in tqdm(pages, desc="List pages"):
                # Page blocks are typically grouped in units of 10 pages
                block = ((p - 1) // 10) * 10 + 1
                items = await parse_list_page(session, p, block)
                all_items.extend(items)
            return all_items

        # Auto-expand mode: stop when an empty page is encountered
        pbar = tqdm(desc="List pages (auto)", unit="page")
        try:
            while True:
                block = ((page - 1) // 10) * 10 + 1
                items = await parse_list_page(session, page, block)
                if not items:
                    break
                # Stop if every item on the page is duplicated
                new_items = [it for it in items if it.get("id") not in seen_ids]
                if not new_items:
                    break
                for it in new_items:
                    if it.get("id"):
                        seen_ids.add(it["id"])
                all_items.extend(new_items)
                page += 1
                pbar.update(1)
        finally:
            pbar.close()
        return all_items

    async def handle_one(self, session: aiohttp.ClientSession, list_item: dict) -> dict | None:
        async with self.sem:
            try:
                detail = await parse_detail(session, list_item["url"])
                # Merge metadata from the list page
                detail["department"] = list_item.get("department", "")
                if list_item.get("title"):
                    detail["title"] = list_item["title"]
                return detail
            except Exception as e:
                print(f"[ERROR] {list_item.get('url')} -> {e}")
                return None

    async def run(self):
        timeout = ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            list_items = await self.gather_list(session)

            self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.out_jsonl.open("w", encoding="utf-8") as fout, \
                 self.out_csv.open("w", newline="", encoding="utf-8-sig") as fcsv:

                w = csv.writer(fcsv)
                w.writerow([
                    "id","url","title","law_name","case_no",
                    "request_org","reply_date","department",
                    "attachments_count","content_len"
                ])

                tasks = [self.handle_one(session, it) for it in list_items]
                # Use a for-await pattern instead of async for
                for coro in tqdm_asyncio.as_completed(tasks, desc="Detail pages", total=len(tasks)):
                    data = await coro
                    if not data:
                        continue
                    fout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
                    w.writerow([
                        data.get("id",""),
                        data.get("url",""),
                        data.get("title",""),
                        data.get("law_name",""),
                        data.get("case_no",""),
                        data.get("request_org",""),
                        data.get("reply_date",""),
                        data.get("department",""),
                        len(data.get("attachments", [])),
                        len(data.get("content","")),
                    ])

# -------------------------------
# Execution
# -------------------------------
if __name__ == "__main__":
    crawler = Crawler(
        start_page=1,
        end_page=None,   # None= all pages
        block_start=1,
        out_jsonl="crawl_lawmaking/lawinterp.jsonl",
        out_csv="crawl_lawmaking/lawinterp.csv",
        concurrency=10,
    )
    asyncio.run(crawler.run())
