import requests
import json

def fast_multiword_biorxiv_search(
    keywords=None,
    from_date="2022-01-01",
    to_date="2025-12-31",
    max_results=300,
    output="biorxiv_multiword.json"
):
    if keywords is None:
        keywords = [
            "aging", "longevity", "senescence"
        ]
    url = f"https://api.biorxiv.org/details/biorxiv/{from_date}/{to_date}/0"
    results = []
    while url and len(results) < max_results:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        for record in data.get("collection", []):
            title = record["title"].lower()
            abstract = record["abstract"].lower()
            if any(word in title or word in abstract for word in keywords):
                doi = record["doi"]
                results.append({
                    "doi": doi,
                    "title": record["title"],
                    "abstract": record["abstract"],
                    #"authors": record["authors"],
                    "date": record["date"],
                    "category": record["category"],
                    "affiliation": record["author_corresponding_institution"],
                    "url": f"https://www.biorxiv.org/content/{doi}",
                })
                if len(results) >= max_results:
                    break
        # Пагинация
        if int(data.get("messages", [{}])[0].get("total", 0)) > len(results):
            url = f"https://api.biorxiv.org/details/biorxiv/{from_date}/{to_date}/{len(results)}"
        else:
            url = None
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(results)} препринтов в {output}")
    with open("razmetka.json", "w", encoding="utf-8") as f:
        f.write(resp.text)

if __name__ == "__main__":
    fast_multiword_biorxiv_search(
        keywords=[
            "aging", "longevity", "senescence"
        ],
        from_date="2022-01-01",
        to_date="2025-12-31",
        max_results=300,
        output="biorxiv_multiword.json"
    )