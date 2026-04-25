# ik_it_act_scraper_min.py
import os, time, re, json, random
from pathlib import Path
from urllib.parse import urljoin, quote_plus
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

IT_SEARCH_QUERIES = [
    # Core Acts & broad terms
    "Information Technology Act, 2000",
    "Information Technology (Amendment) Act, 2008",
    "IT Act cyber crime",
    "electronic record Information Technology Act",
    "electronic evidence Information Technology Act",
    "intermediary liability section 79",
    "IT Rules 2021 intermediary guidelines",
    "CERT-In directions Information Technology Act",

    # Sections (offences & key provisions)
    "Section 43 Information Technology Act",
    "Section 65 Information Technology Act",
    "Section 66 Information Technology Act",
    "Section 66A Information Technology Act",
    "Section 66B Information Technology Act",
    "Section 66C Information Technology Act",
    "Section 66D Information Technology Act",
    "Section 66E Information Technology Act",
    "Section 67 Information Technology Act",
    "Section 67A Information Technology Act",
    "Section 67B Information Technology Act",
    "Section 68 Information Technology Act",
    "Section 69 Information Technology Act",
    "Section 69A Information Technology Act",
    "Section 70 Information Technology Act",
    "Section 72 Information Technology Act",
    "Section 72A Information Technology Act",
    "Section 73 Information Technology Act",
    "Section 74 Information Technology Act",
    "Section 79 Information Technology Act safe harbour",

    # Evidence linkage
    "Section 65B Evidence Act electronic record",
    "admissibility of electronic evidence Section 65B",

    # Task-style/real-world cyber disputes (boost recall)
    "phishing Information Technology Act",
    "UPI fraud Information Technology Act",
    "OTP fraud Information Technology Act",
    "cyber stalking Information Technology Act",
    "morphing images Information Technology Act",
    "child sexual content Section 67B",
    "identity theft Section 66C",
    "cheating by personation Section 66D",
    "unauthorised access Section 43",
    "hacking Section 66",
    "privacy violation Section 66E",
    "blocking orders Section 69A",
    "takedown orders intermediary Section 79",
    "bail Information Technology Act offences"
]

OUTCOME_REGEX = {
    "PETITION_ALLOWED": [
        r"\b(writ\s+)?petition\s+(?:is\s+)?allowed\b", r"\bappeal(?:s)?\s+(?:is\s+)?allowed\b",
        r"\bimpugned\s+order\s+(?:quashed|set\s+aside)\b",
        r"\bpetition\s+is\s+hereby\s+allowed\b", r"\ballowed\s+accordingly\b"
    ],
    "PETITION_DISMISSED": [
        r"\b(writ\s+)?petition\s+(?:is\s+)?dismissed\b", r"\bappeal(?:s)?\s+(?:is\s+)?dismissed\b",
        r"\bpetition\s+rejected\b", r"\bpetition\s+stands\s+dismissed\b", r"\bdismissed\s+accordingly\b"
    ],
    "PARTLY_ALLOWED": [r"\bpartly\s+allowed\b", r"\ballowed\s+in\s+part\b"],
    "DISPOSED_OF": [r"\bdisposed\s+of\b", r"\bdisposed\s+with\s+directions\b", r"\bwith\s+liberty\b",
                    r"\bpetition\s+is\s+disposed\s+of\b"],
    "REMANDED": [r"\bremand(?:ed)?\b", r"\bremitted\b"],
    "CONVICTION_UPHELD": [r"\bconviction\s+(?:upheld|affirmed|confirmed)\b", r"\bsentence\s+affirmed\b"],
    "CONVICTION_SET_ASIDE": [r"\bconviction\s+set\s+aside\b", r"\bacquitted\b", r"\bbenefit\s+of\s+doubt\b",
                             r"\bset\s+aside\s+and\s+quashed\b", r"\bquashed\s+and\s+set\s+aside\b"],
    "BAIL_GRANTED": [r"\bbail\s+granted\b", r"\binterim\s+bail\s+granted\b", r"\banticipatory\s+bail\s+granted\b"],
    "BAIL_REJECTED": [r"\bbail\s+(?:rejected|denied)\b", r"\bbail\s+application\s+dismissed\b"]
}

IT_SECTIONS_SET = {
    "43","65","66","66A","66B","66C","66D","66E","67","67A","67B",
    "68","69","69A","70","72","72A","73","74","79"
}
SECTION_PATTERNS = [
    r'(?:under\s+)?[Ss](?:ec(?:tion)?)?\.?\s*(\d+[A-Z]?)\s*(?:of\s+the\s+)?(?:Information\s+Technology\s+Act|IT\s*Act)',
    r'(?:u/s|u\.s\.)\s*(\d+[A-Z]?)\s*(?:IT\s*Act|Information\s*Technology\s*Act)',
    r'IT\s*Act\s*[Ss](?:ec(?:tion)?)?\.?\s*(\d+[A-Z]?)',
    r'Information\s*Technology\s*Act\s*[Ss](?:ec(?:tion)?)?\.?\s*(\d+[A-Z]?)'
]

class ITActScraper:
    def __init__(self, download_dir="scraped_it_cases", headless=True):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        opts = Options()
        if headless: opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1600,1000")
        opts.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        prefs = {
            "download.default_directory": str(self.download_dir.resolve()),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True
        }
        opts.add_experimental_option("prefs", prefs)

        self.driver = webdriver.Chrome(options=opts)
        self.wait = WebDriverWait(self.driver, 20)

    def close(self):
        try: self.driver.quit()
        except: pass

    # ---------- URL Collection ----------
    def search(self, query, max_urls=400):
        """Return case URLs for one query (ordered, deduped per query)."""
        search_url = f"https://indiankanoon.org/search/?formInput={quote_plus(query)}"
        self.driver.get(search_url)
        time.sleep(1.0 + random.random()*0.6)

        urls = []
        seen = set()
        while len(urls) < max_urls:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            links = soup.select('div.result_title a[href^="/doc/"]')
            if not links:
                links = soup.select('a[href^="/doc/"]')

            added = 0
            for a in links:
                href = (a.get('href') or '').strip()
                if not href: continue
                url = urljoin("https://indiankanoon.org", href)
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
                    added += 1

            if added == 0:  # no new items → stop
                break

            # Try "Next"
            try:
                nxt = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(., 'Next')]")))
                nxt.click()
                time.sleep(0.9 + random.random()*0.5)
            except Exception:
                break
        return urls[:max_urls]

    def collect_urls(self, per_query_limit=300, urls_csv="scraped_it_cases/it_act_urls.csv"):
        """Collect URLs for all queries, dedupe globally, save a CSV."""
        all_rows = []
        global_seen = set()
        print("🔎 Collecting IT Act URLs …")
        for q in IT_SEARCH_QUERIES:
            q_urls = self.search(q, max_urls=per_query_limit)
            for u in q_urls:
                if u not in global_seen:
                    global_seen.add(u)
                    all_rows.append({"query": q, "url": u})
        df_urls = pd.DataFrame(all_rows)
        Path(urls_csv).parent.mkdir(parents=True, exist_ok=True)
        df_urls.to_csv(urls_csv, index=False)
        print(f"✅ Collected {len(df_urls)} unique URLs → {urls_csv}")
        return df_urls

    # ---------- Parsing ----------
    def extract_case(self, url):
        """Return dict with parsed fields or None if failed."""
        try:
            self.driver.get(url)
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(0.6 + random.random()*0.4)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # title
            title = None
            for sel in ['div.doc_title', 'h1', 'title']:
                node = soup.select_one(sel)
                if node:
                    title = node.get_text(strip=True); break
            title = title or "Unknown Case"

            # court/date meta
            meta_text = ""
            for sel in ['div.doc_subtitle', 'div.sub_title', '.subtitle', '.doc_source']:
                node = soup.select_one(sel)
                if node:
                    meta_text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)); break

            # body (several fallbacks)
            body = None
            for sel in ['#judgment', '#document', '.judgments', 'pre#pre_', '#content', 'article']:
                node = soup.select_one(sel)
                if node:
                    body = node.get_text(" ", strip=True); break
            if not body:
                body = soup.get_text(" ", strip=True)
            if len(body) < 500:
                return None

            # operative paragraph (tail)
            operative = body[-1500:]

            outcome_label, outcome_phrase = self.detect_outcome(operative)
            sections = self.extract_sections(body)
            sections_str = ", ".join(sorted(sections)) if sections else "Unknown"

            # best-effort date
            date = ""
            m = re.search(r"Decided on\s*[:\-]\s*([A-Za-z0-9 ,\-]+)", body, flags=re.I)
            if m: date = m.group(1).strip()

            return {
                "title": title,
                "url": url,
                "court_meta": meta_text,
                "date": date,
                "outcome_label": outcome_label,
                "outcome_phrase": outcome_phrase,
                "it_sections": sections_str,
                "text_length": len(body),
                "operative_snippet": operative[:750]
            }
        except Exception:
            return None

    def detect_outcome(self, text):
        t = text.lower()
        for label, patterns in OUTCOME_REGEX.items():
            for rx in patterns:
                m = re.search(rx, t)
                if m:
                    return label, m.group(0)
        return "INCONCLUSIVE", ""

    def extract_sections(self, text):
        secs = set()
        for pat in SECTION_PATTERNS:
            for s in re.findall(pat, text, flags=re.IGNORECASE):
                s = s.strip().upper()
                if s in IT_SECTIONS_SET:
                    secs.add(s)
        return secs

    def download_pdf(self, url):
        try:
            self.driver.get(url)
            for locator in [(By.ID, "pdfdoc"), (By.ID, "pdf"), (By.XPATH, "//a[contains(., 'PDF')]")]:
                try:
                    btn = self.wait.until(EC.element_to_be_clickable(locator))
                    btn.click()
                    time.sleep(2.0)
                    return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    # ---------- Main run ----------
    def run(self,
            per_query_limit=300,
            save_pdfs=False,
            urls_csv="scraped_it_cases/it_act_urls.csv",
            out_csv="scraped_it_cases/it_act_dataset.csv",
            test_cap=None,
            resume=True):
      
        # 1) URLs
        if not os.path.exists(urls_csv):
            self.collect_urls(per_query_limit=per_query_limit, urls_csv=urls_csv)
        df_urls = pd.read_csv(urls_csv)
        urls = df_urls["url"].dropna().astype(str).tolist()

        # 2) Test cap
        if test_cap is not None and len(urls) > test_cap:
            urls = urls[:test_cap]
        print(f"✅ URLs to process: {len(urls)}")

        # 3) Resume: skip already-scraped
        if resume and os.path.exists(out_csv):
            try:
                done = pd.read_csv(out_csv, usecols=["url"])
                done_set = set(done["url"].dropna().tolist())
                urls = [u for u in urls if u not in done_set]
                print(f"🔁 Resume enabled → skipping {len(done_set)} already scraped → {len(urls)} pending")
            except Exception:
                pass

        # 4) Parse
        rows, failed = [], 0
        for i, url in enumerate(tqdm(urls, desc="Scraping cases")):
            data = self.extract_case(url)
            if data:
                rows.append(data)
                if save_pdfs:
                    self.download_pdf(url)
            else:
                failed += 1

            if (i + 1) % 10 == 0:
                existing = []
                if os.path.exists(out_csv):
                    try:
                        existing = [pd.read_csv(out_csv)]
                    except Exception:
                        existing = []
                df_now = pd.concat([*(existing or []), pd.DataFrame(rows)], ignore_index=True)
                df_now.drop_duplicates(subset="url", inplace=True)
                Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
                df_now.to_csv(out_csv, index=False)

            time.sleep(0.4 + random.random()*0.4)  # light throttling

        # Final save/merge
        final_parts = []
        if os.path.exists(out_csv):
            try:
                final_parts.append(pd.read_csv(out_csv))
            except Exception:
                pass
        final_parts.append(pd.DataFrame(rows))
        df = pd.concat(final_parts, ignore_index=True)
        df.drop_duplicates(subset="url", inplace=True)
        df.to_csv(out_csv, index=False)

        print("\n======== SUMMARY ========")
        print(f"Total URLs in list: {len(df_urls)} | Newly parsed this run: {len(rows)} | Failed (this run): {failed}")
        if not df.empty:
            print("\nOutcome label counts:")
            print(df["outcome_label"].value_counts())
            print("\nTop IT sections:")
            print(df["it_sections"].value_counts().head(10))
            print(f"\nSaved URLs CSV → {urls_csv}")
            print(f"Saved DATA CSV → {out_csv}")
        else:
            print("No rows parsed. Try headless=False once to inspect DOM.")

        return df


if __name__ == "__main__":
    scraper = ITActScraper(download_dir="scraped_it_cases", headless=True)
    try:
        df = scraper.run(
            per_query_limit=300,             
            save_pdfs=False,
            urls_csv="scraped_it_cases/it_act_urls.csv",
            out_csv="scraped_it_cases/it_act_dataset.csv",
            resume=True                     
        )
    finally:
        scraper.close()
