import time
import pandas as pd
from firecrawl import FirecrawlApp

# Load Firecrawl API key from environment variable


def scrape_page(url):
    """Scrape a website using Firecrawl."""
    scrape_status = app.scrape_url(
        url,
        params={'formats': ['markdown']}
    )
    return scrape_status

def scrape_links(csv_file, output_file):
    df = pd.read_csv(csv_file)
    
    if 'url' not in df.columns:
        print("CSV file must have a column named 'url'")
        return
    
    scraped_data = []
    
    for i, row in df.iterrows():
        url = row['url']
        print(f"Scraping: {url}")
        content = scrape_page(url)
        
        if content:
            scraped_data.append({
                "url": url,
                "content": content
            })
        
        # To avoid rate limits
        time.sleep(1)
    
    # Save to CSV (LLM-ready format)
    output_df = pd.DataFrame(scraped_data)
    output_df.to_csv(output_file, index=False)
    print(f"\n✅ Scraped data saved to {output_file}")

if __name__ == "__main__":
    input_csv = "data/links.csv"
    output_csv = "data/scraped_data.csv"
    
    scrape_links(input_csv, output_csv)
