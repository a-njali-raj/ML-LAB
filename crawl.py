import requests
def simple_scraper(url):
    response = requests.get(url)
    if response.status_code == 200:
       print("Content:")
       print(response.text)
    else:
       print("failed to fetch the page, status code:",response.status_code)
url_to_scrape="https://ajce.in"
simple_scraper(url_to_scrape)
