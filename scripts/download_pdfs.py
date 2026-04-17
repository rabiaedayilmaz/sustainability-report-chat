import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.nttdata.com/global/en/about-us/sustainability/report"
DATA_DIR = "data" 

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def extract_year_from_text(text):
    # 4 haneli yıl ara (2000-2029 arası)
    match = re.search(r'(20[0-2][0-9])', text)
    if match:
        return match.group(1)
    return None

def download_file(url, year):
    # data/YEAR formatında klasör oluştur
    folder = os.path.join(DATA_DIR, year)
    os.makedirs(folder, exist_ok=True)
    
    # Dosya adını URL'den al
    filename = url.split("/")[-1].split("?")[0]  # Query string'i temizle
    
    # Eğer .pdf uzantısı yoksa ekle
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'
    
    filepath = os.path.join(folder, filename)

    # Dosya zaten varsa atla
    if os.path.exists(filepath):
        print(f"[SKIP] {year}/{filename} - Already exists")
        return

    try:
        print(f"[DOWNLOAD] {year}/{filename}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # PDF olduğunu doğrula
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower():
            print(f"[WARNING] {filename} - Not a PDF (Content-Type: {content_type})")
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"[SUCCESS] {year}/{filename} - {len(response.content)} bytes")
        
    except Exception as e:
        print(f"[ERROR] {filename} - {str(e)}")


def scrape():
    print(f"Scraping: {BASE_URL}\n")
    
    try:
        response = requests.get(BASE_URL, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"[ERROR] Could not fetch page: {str(e)}")
        return

    # Tüm linkleri bul
    links = soup.find_all("a", href=True)
    pdf_count = 0

    for a in links:
        href = a.get("href")
        text = a.get_text(strip=True)

        if not href:
            continue

        # Sadece PDF linkleri
        if ".pdf" not in href.lower():
            continue

        pdf_count += 1
        full_url = urljoin(BASE_URL, href)

        # Önce link metninden yıl çıkarmayı dene
        year = extract_year_from_text(text)
        
        # Bulamazsan URL'den dene
        if not year:
            year = extract_year_from_text(href)
        
        # Hala bulamadıysan dosya adından dene
        if not year:
            filename = href.split("/")[-1]
            year = extract_year_from_text(filename)
        
        # Son çare: "unknown" klasörüne koy
        if not year:
            year = "unknown"

        download_file(full_url, year)

    print(f"\n[DONE] Total PDFs found: {pdf_count}")
    print(f"Files saved in '{DATA_DIR}' directory with year-based structure")


if __name__ == "__main__":
    scrape()