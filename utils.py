from bs4 import BeautifulSoup
import requests

def fetch_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Failed to fetch page")
        return None

    soup = BeautifulSoup(response.text, "lxml")

    # find the element with id="eng-abstract"
    abstract_div = soup.find(id="eng-abstract")

    if abstract_div is None:
        print("No abstract section found")
        return None

    content = abstract_div.get_text(separator=" ", strip=True)

    return content
    