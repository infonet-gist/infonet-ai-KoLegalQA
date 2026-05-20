"""
MOLEG Law Interpretation Case Crawler
- List page: https://www.moleg.go.kr/lawinfo/nwLwAnList.mo?mid=a10106020000&pageCnt=10&currentPage=1&keyField=&keyWord=&sort=date
- Detail page: https://www.moleg.go.kr/lawinfo/nwLwAnInfo.mo?mid=a10106020000&cs_seq=443208&...

Output files:
- crawl_moleg/opinions.csv
- crawl_moleg/opinions.jsonl
- crawl_moleg/seen_ids.txt
"""

import re
import csv
import json
import random
import asyncio
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
import argparse

import httpx
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

# -------------------- Configuration --------------------
BASE = "https://www.moleg.go.kr"
LIST_PATH = "/lawinfo/nwLwAnList.mo"
DETAIL_PATH = "/lawinfo/nwLwAnInfo.mo"
MID = "a10106020000"

OUTDIR = Path("crawl_moleg")
OUTDIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTDIR / "opinions.csv"
JSONL_PATH = OUTDIR / "opinions.jsonl"
SEEN_PATH = OUTDIR / "seen_ids.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MOLEG-Law-Interp-Crawler/1.0; +https://example.org)",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# -------------------- Utilities --------------------
def norm(u: str) -> str:
    return u.strip()

def text(el) -> str:
    if not el:
        return ""
    for tag in el.find_all(["script", "style", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()

def build_list_url(page: int, page_cnt: int, key_field: str, keyword: str, sort: str) -> str:
    qs = {
        "mid": MID,
        "pageCnt": page_cnt,
        "currentPage": page,
        "keyField": key_field or "",
        "keyWord": keyword or "",
        "sort": sort or "date",  # Latest first
    }
    return f"{BASE}{LIST_PATH}?{urlencode(qs, doseq=False)}"

def extract_id_from_url(url: str) -> str:
    """
    Extract MOLEG detail identifier from the cs_seq query parameter.
    """
    qs = parse_qs(urlparse(url).query)
    if "cs_seq" in qs and qs["cs_seq"]:
        return qs["cs_seq"][0]
    # Fallback query parameters
    for k in ("id", "no", "seq"):
        if k in qs and qs[k]:
            return qs[k][0]
    # Handle rare cases where the identifier appears in the URL path
    m = re.search(r"/(\d+)(?:\D|$)", urlparse(url).path)
    return m.group(1) if m else ""

def thtd_to_dict(soup: BeautifulSoup) -> dict:
    """
    Convert table- or dl-based metadata into a dictionary:
    {label: value or attachment_links}

    - Attachment/file fields are stored as link lists.
    """
    data = {}
    # <table> (th/td)
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            th = tr.find(["th", "dt"])
            td = tr.find(["td", "dd"])
            k = text(th)
            v = text(td)
            if not k:
                continue
            # Collect attachment/file links
            if ("첨부" in k or "파일" in k) and td:
                links = []
                for a in td.select("a[href]"):
                    href = norm(urljoin(BASE, a.get("href")))
                    if not href:
                        continue
                    links.append({
                        "name": text(a) or a.get("download") or href.split("/")[-1],
                        "url": href
                    })
                data[k] = links
            else:
                if v:
                    data[k] = v

    # Parse <dl><dt>/<dd> structures
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        if len(dts) != len(dds):
            continue
        for dt_el, dd_el in zip(dts, dds):
            k = text(dt_el)
            if not k:
                continue
            # Handle attachment/file entries
            if ("첨부" in k or "파일" in k):
                links = []
                for a in dd_el.select("a[href]"):
                    href = norm(urljoin(BASE, a.get("href")))
                    if href:
                        links.append({
                            "name": text(a) or href.split("/")[-1],
                            "url": href
                        })
                data[k] = links
            else:
                v = text(dd_el)
                if v:
                    data[k] = v
    return data

# -------------------- Parsers --------------------
def parse_list(html: str) -> list:
    """
    Extract detail page URLs (`nwLwAnInfo.mo?...cs_seq=`) from a list page.
    """
    soup = BeautifulSoup(html, "lxml")
    detail_urls = set()

    # Common pattern: a[href*="nwLwAnInfo.mo"] containing cs_seq
    for a in soup.select('a[href*="nwLwAnInfo.mo"]'):
        href = a.get("href")
        if not href:
            continue
        absu = norm(urljoin(BASE, href))
        if "nwLwAnInfo.mo" in absu and "cs_seq=" in absu:
            detail_urls.add(absu)

    # Additional pass through all table links
    for a in soup.select("table a[href]"):
        href = a.get("href")
        if not href:
            continue
        absu = norm(urljoin(BASE, href))
        if "nwLwAnInfo.mo" in absu and "cs_seq=" in absu:
            detail_urls.add(absu)

    return sorted(detail_urls)

def parse_detail(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # ----- Title -----
    title = ""
    for sel in [
        # Common title selectors
        "div.titArea h3", "div.titArea h2", "div.view_tit h3", "div.view_tit h2",
        ".bbs_view .tit", ".board_view .tit", "h1.tit", "h2.tit",
        "h1", "h2", "h3", ".subject", "title"
    ]:
        el = soup.select_one(sel)
        if el:
            if sel == "title" and soup.title:
                title = (soup.title.string or "").strip()
            else:
                title = text(el)
            if title:
                break

    # ----- Status / labels -----
    status = ""
    for sel in [".status", ".badge", ".label", ".state", ".stt", ".tag"]:
        st = text(soup.select_one(sel))
        if st:
            status = st
            break

    # ----- Table-based metadata -----
    kv = thtd_to_dict(soup)

    def first_match(keys, contains=False):
        for k in kv.keys():
            for want in keys:
                if (contains and want in k) or (not contains and k == want):
                    return kv[k]
        return ""

    department = first_match(["소관부처","소관 부처","소관기관","주관부처","담당부서"], contains=True) or ""
    contact    = first_match(["연락처","담당자","문의처"], contains=True) or ""
    period     = first_match(["의견제출기간","의견 제출 기간","공고기간","기간","회신일자"], contains=True) or ""

    # ----- Attachments (fallback blocks) -----
    attachments = None
    if not attachments:
        links = []
        for a in soup.select('.file, .attach, .file_down, .download, .fileArea a[href], .attach_area a[href], .file_list a[href]'):
            href = a.get("href")
            if not href:
                continue
            absu = norm(urljoin(url, href))
            if absu.lower().startswith("http"):
                links.append({"name": text(a) or absu.split("/")[-1], "url": absu})
        attachments = links

    # ----- Main content -----
    content = ""
    for sel in [
        "div.view_con", "div.bbs_view", "div.board_view", "div.con_view",
        "div.view-cont", "div.article", "div.contents",
        "#content", "article", "main"
    ]:
        el = soup.select_one(sel)
        content = text(el)
        if content:
            break
    if not content:
        body = soup.find("body")
        content = text(body)

    return {
        "id": extract_id_from_url(url),
        "url": url,
        "title": title,
        "status": status,
        "department": department,
        "contact": contact,
        "period": period,
        "attachments": attachments or [],
        "content": content,
        "raw_fields": kv,
        "source_site": "moleg",
    }

# -------------------- Crawler --------------------
class Crawler:
    def __init__(self, start_page=1, end_page=1, page_cnt=10, max_tasks=10, timeout=20.0,
                 key_field="", keyword="", sort="date", force=False):
        self.start_page = start_page
        self.end_page = end_page
        self.page_cnt = page_cnt
        self.max_tasks = max_tasks
        self.timeout = timeout
        self.key_field = key_field
        self.keyword = keyword
        self.sort = sort
        self.force = force

        self.seen = set()
        if SEEN_PATH.exists():
            self.seen |= set(x.strip() for x in SEEN_PATH.read_text(encoding="utf-8").splitlines() if x.strip())

    async def fetch(self, client: httpx.AsyncClient, url: str) -> str:
        backoff = 1.0
        for attempt in range(6):
            try:
                r = await client.get(url, headers=HEADERS, timeout=self.timeout, follow_redirects=True)
                if r.status_code in (200, 201):
                    r.encoding = r.encoding or "utf-8"
                    return r.text
                if r.status_code in (429, 503):
                    await asyncio.sleep(backoff + random.uniform(0, 0.4))
                    backoff = min(backoff * 2, 10)
                else:
                    raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
            except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPStatusError):
                if attempt == 5:
                    raise
                await asyncio.sleep(backoff + random.uniform(0, 0.4))
                backoff = min(backoff * 2, 10)
        return ""

    async def run(self):
        # CSV header
        if not CSV_PATH.exists():
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "id","url","title","status","department","contact","period",
                    "attachments_count","content_len","source_site"
                ])
        # Ensure JSONL file exists even if every item gets skipped
        JSONL_PATH.touch(exist_ok=True)

        limits = httpx.Limits(max_connections=self.max_tasks, max_keepalive_connections=self.max_tasks)
        async with httpx.AsyncClient(http2=True, limits=limits) as client:

            # 1) Traverse list pages and collect detail URLs
            all_detail_urls = []
            for page in range(self.start_page, self.end_page + 1):
                list_url = build_list_url(page, self.page_cnt, self.key_field, self.keyword, self.sort)
                html = await self.fetch(client, list_url)
                urls = parse_list(html)
                all_detail_urls.extend(urls)

                # Respectful delay between list page requests
                await asyncio.sleep(random.uniform(0.1, 0.3))

            # Remove duplicates
            all_detail_urls = sorted(set(all_detail_urls))

            # 2) Fetch detail pages concurrently
            sem = asyncio.Semaphore(self.max_tasks)

            async def handle(url, pbar):
                async with sem:
                    # Skip already-seen IDs before making requests
                    _id = extract_id_from_url(url)
                    if _id and (not self.force) and _id in self.seen:
                        pbar.write(f"[skip] seen {_id} {url}")
                        pbar.update(1)
                        return

                    html = await self.fetch(client, url)
                    data = parse_detail(html, url)

                    # Persist output
                    with open(JSONL_PATH, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(data, ensure_ascii=False) + "\n")
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as cf:
                        w = csv.writer(cf)
                        w.writerow([
                            data["id"], data["url"], data["title"], data["status"], data["department"],
                            data["contact"], data["period"], len(data["attachments"]), len(data["content"]),
                            data.get("source_site","moleg")
                        ])

                    # Update seen ID cache
                    if data["id"]:
                        self.seen.add(data["id"])
                        with open(SEEN_PATH, "a", encoding="utf-8") as sf:
                            sf.write(data["id"] + "\n")

                    pbar.update(1)
                    pbar.set_description(f"{data['id'] or ''}")

                    # Respectful delay between detail requests
                    await asyncio.sleep(random.uniform(0.05, 0.2))

            with tqdm(total=len(all_detail_urls), desc="Detail pages", ncols=100) as pbar:
                tasks = [asyncio.create_task(handle(url, pbar)) for url in all_detail_urls]
                if tasks:
                    await asyncio.gather(*tasks)

# -------------------- Entry point --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MOLEG Law Interpretation Crawler")
    ap.add_argument("--start", type=int, default=1, help="시작 목록 페이지")
    ap.add_argument("--end", type=int, default=852, help="끝 목록 페이지")
    ap.add_argument("--pagecnt", type=int, default=10, help="목록 한 페이지 게시물 수 (기본 10)")
    ap.add_argument("--concurrency", type=int, default=4, help="동시 상세 요청 수")
    ap.add_argument("--timeout", type=float, default=20.0, help="요청 타임아웃 (초)")
    ap.add_argument("--keyfield", type=str, default="", help="검색 필드 (공란=전체)")
    ap.add_argument("--keyword", type=str, default="", help="검색어")
    ap.add_argument("--sort", type=str, default="date", choices=["date","hit","title"], help="정렬 기준")
    ap.add_argument("--force", action="store_true", help="seen 무시하고 재수집")
    ap.add_argument("--reset-seen", action="store_true", help="seen_ids.txt 초기화")

    args = ap.parse_args()

    if args.reset_seen:
        try:
            SEEN_PATH.unlink(missing_ok=True)
            print("seen_ids.txt reset completed")
        except Exception as e:
            print(f"Failed to reset seen IDs: {e}")

    crawler = Crawler(
        start_page=args.start,
        end_page=args.end,
        page_cnt=args.pagecnt,
        max_tasks=args.concurrency,
        timeout=args.timeout,
        key_field=args.keyfield,
        keyword=args.keyword,
        sort=args.sort,
        force=args.force,
    )

    asyncio.run(crawler.run())

    print(f"\nDone! Output files:\n- {CSV_PATH}\n- {JSONL_PATH}\n")