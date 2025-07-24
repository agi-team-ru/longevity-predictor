import requests
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

class PubMedParser:
    def __init__(self, start_year=2021, end_year=2025):
        self.start_year = start_year
        self.end_year = end_year

    @staticmethod
    def extract_iso_date(article):
        """
        Попытается достать дату публикации в порядке приоритета:
        1) <ArticleDate DateType="Electronic">
        2) <PubDate> в JournalIssue
        3) <DateCompleted> в MedlineCitation
        Вернёт строку "YYYY-MM-DD" или None, если не найдётся.
        """
        ad = article.find("ArticleDate", {"DateType": "Electronic"})
        if ad:
            y = ad.find("Year").text
            m = ad.find("Month").text
            d = ad.find("Day").text
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

        pd = article.find("JournalIssue").find("PubDate")
        if pd:
            y_tag = pd.find("Year")
            m_tag = pd.find("Month")
            d_tag = pd.find("Day")
            y = int(y_tag.text) if y_tag else 1
            m = int(m_tag.text) if m_tag and m_tag.text.isdigit() else 1
            d = int(d_tag.text) if d_tag and d_tag.text.isdigit() else 1
            return f"{y:04d}-{m:02d}-{d:02d}"

        dc = article.find("DateCompleted")
        if dc:
            y = int(dc.find("Year").text)
            m = int(dc.find("Month").text)
            d = int(dc.find("Day").text)
            return f"{y:04d}-{m:02d}-{d:02d}"
        return None

    def get_uids(self, date_range=None):
        if date_range is None:
            date_range = f"{self.start_year}:{self.end_year}[pdat]"
        print(f"[PubMed] Получение UID статей за период: {date_range}")
        query = f'''(("Aging"[Majr] OR "Longevity"[Majr])
            AND 
            (
            "research priority"[TIAB] OR "research agenda"[TIAB] OR 
            "future direction*"[TIAB] OR "challenge*"[TIAB] OR 
            "open question*"[TIAB] OR "knowledge gaps"[TIAB] OR 
            "research needs"[TIAB] OR "unanswered questions"[TIAB] OR 
            "critical issues"[TIAB] OR "scientific priorities"[TIAB]
            ) 
            AND
            ("Cellular Senescence"[MeSH] OR "DNA Damage"[MeSH] OR "Epigenesis, Genetic"[MeSH] OR "Mitochondria/metabolism"[MeSH] OR "Aging/physiology"[MeSH] OR "Biological Clocks"[MeSH] OR "drug*"[MeSH] OR "biomarker*"[MeSH])
            NOT
            (geriatr*[TIAB] OR sociolog*[TIAB] OR "resource allocation"[TIAB] OR "health policy"[TIAB] OR "health economics"[TIAB] OR "insurance coverage"[TIAB]) 
            AND 
            {date_range}
            AND
            (review[pt] OR systematic review[pt]))'''
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=1000&retmode=json"
        response = requests.get(url)
        data = response.json()
        print(f"[PubMed] Получено {len(data['esearchresult']['idlist'])} UID")
        return data

    def get_json(self, uids_of_articles):
        ids = uids_of_articles['esearchresult']['idlist']
        if not ids:
            print("[PubMed] Нет статей для скачивания.")
            return []
        ids_str = ','.join(ids)
        print(f"[PubMed] Скачивание XML для {len(ids)} статей...")
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "xml")
        info = []
        for article in soup.find_all("PubmedArticle"):
            pmid = article.find("PMID").text
            title = article.find("ArticleTitle").text
            abstract = article.find("AbstractText").text
            iso_date = self.extract_iso_date(article) or "0000-00-00"
            journal = article.find("Title").text
            affiliation = article.find("Affiliation")
            grant_id = article.find("GrantID")
            founders = article.find("Agency")
            info.append({
                "pmid": int(pmid),
                "title" : title,
                "abstract": abstract,
                "pubdate": iso_date,
                "journal": journal,
                "affiliation": affiliation.text if affiliation else None,
                "GrantId": grant_id.text if grant_id else None,
                "founders" : founders.text if founders else None,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
            })
        print(f"[PubMed] Распарсено {len(info)} статей.")
        return info

    def daily_parse(self):
        today = datetime.today()
        yesterday = today - timedelta(days=1)
        date_range = f"{yesterday.strftime('%Y/%m/%d')}:{today.strftime('%Y/%m/%d')}[pdat]"
        print(f"[PubMed] Ежедневный парсинг за {date_range}")
        uids_of_articles = self.get_uids(date_range=date_range)
        return self.get_json(uids_of_articles)

if __name__ == "__main__":
    parser = PubMedParser()
    parser.get_json(parser.get_uids(), output_file="pubmed_articles.json")
    # Для ежедневного парсинга:
    # parser.daily_parse()