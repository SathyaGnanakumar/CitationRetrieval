import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"

class SemanticScholarPaper:
    def __init__(self, data):
        self.paper_id = data.get("paperId")
        self.title = data.get("title")
        self.year = data.get("year")
        self.authors = [a.get("name") for a in data.get("authors", [])]
        self.url = data.get("url")
        self.venue = data.get("venue")
        self.external_ids = data.get("externalIds", {})
        self.publication_types = data.get("publicationTypes", {})

class SemanticScholarLookupService:

    def lookup(self, query, year=None):
        params = {
            "query": query,
            "fields": "title,year,authors,venue,url,externalIds,publicationTypes",
            "limit": 10,
        }
        if year:
            params["year"] = year

        response = requests.get(f"{BASE_URL}/paper/search", params=params)
        data = response.json()
        papers = data.get("data", [])
        return [SemanticScholarPaper(p) for p in papers]
