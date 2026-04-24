import requests
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding='utf-8')
def decode_message(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    data = []
    rows = soup.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) == 3:
            try:
                x = int(cols[0].get_text(strip=True))
                char = cols[1].get_text(strip=True)
                y = int(cols[2].get_text(strip=True))
                data.append((char, x, y))
            except:
                continue
    if not data:
        print("Still failed — no table data found.")
        return
    max_x = max(x for _, x, _ in data)
    max_y = max(y for _, _, y in data)
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    for char, x, y in data:
        grid[y][x] = char
    for row in reversed(grid):
        print("".join(row))
if __name__ == "__main__":
    url = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
    decode_message(url)