import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json 
import os
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "https://www.easylaw.go.kr/CSP/"
LIST_URL = "https://www.easylaw.go.kr/CSP/OnhunqueansLstRetrieve.laf"

domain_num = {
    25: "가정법률",
    89: "아동청소년/교육",
    84: "부동산/임대차",
    92: "금융/금전",
    83: "사업",
    91: "창업",
    100: "무역/출입국",
    88: "소비자",
    87: "문화/여가생활",
    85: "민형사/소송",
    90: "교통/운전",
    82: "근로/노동",
    97: "복지",
    81: "국방/보훈",
    94: "정보통신/기술",
    96: "환경/에너지",
    86: "사회안전/범죄",
    95: "국가및지자체"
    }


def get_retrying_session():
    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        backoff_factor=2
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_detail_content(detail_url):
    try:
        response = requests.get(detail_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        answer_area = soup.select_one("div.ans")
        
        if not answer_area:
            return ""
        
        return answer_area.get_text(separator="\n", strip=True)
    
    except Exception as e:
        print(f"[ERROR] Failed to fetch detail page: {detail_url}\n{e}")
        return ""


def get_articles(domain_code, domain_name):
    articles = []
    session = get_retrying_session()
    page_size = 20
    
    page = 1
    while True:
        print(f"[{domain_name}] Processing page {page}...")

        data = {
            'pageType': page_size,
            'curPage': page,
            'astSeq': f'QNA{domain_code}',
            'onhunqnaAstSeq': domain_code
        }

        res = session.post(LIST_URL, data=data)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        qa_items = soup.select("ul.question li.qa")
        item_count = len(qa_items)
        
        if item_count == 0:
            print(f"[{domain_name}] No posts found on page {page}")
            break

        for qa in qa_items:
            title_tag = qa.select_one("div.ttl a")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            relative_link = title_tag.get("href")
            full_link = urljoin(BASE_URL, relative_link)
            full_text = get_detail_content(full_link)

            if full_text:
                articles.append({
                    "title": title,
                    "content": full_text,
                    "url": full_link
                })

        print(f"[{domain_name}] Finished processing page {page} (Total collected: {len(articles)})")
        time.sleep(1)

        if item_count < page_size:
            break   # Reached the last page

        page += 1

    return articles



def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved successfully: {filename}")

if __name__ == "__main__":
    os.makedirs("crawling_output", exist_ok=True)

    for code, name in domain_num.items():
        print(f"\n[{name}] Starting data collection")
        articles = get_articles(code, name)
        filename = f"crawling_output/{name.replace('/', '_')}.json"
        save_to_json(articles, filename)
        print(f"[{name}] Data collection completed")