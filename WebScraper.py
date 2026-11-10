import httpx
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import argparse
from typing import Optional


def scrape_website(url: str, output_dir: str = "scraped_content") -> str:
    """
    Fetches the content from the given URL and saves it as a .txt file.

    Args:
        url: The website URL to scrape.
        output_dir: Directory to store the output text files.

    Returns:
        Path to the saved text file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Fetch the webpage content
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(follow_redirects=True, timeout=15) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            # Detect encoding
            response.encoding = response.charset_encoding or "utf-8"
            html_content = response.text

    except httpx.HTTPError as e:
        print(f"❌ HTTP Error fetching {url}: {e}")
        return ""

    # Parse HTML and extract text
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # Get text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    # Generate filename from URL
    sanitized_url = re.sub(r'[^\w]', '_', url.strip("https://").strip("http://"))[:80]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitized_url}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Scraped on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write(text)

    print(f"✅ Content saved to: {filepath}")
    print(f"📄 Total characters: {len(text):,}")

    return filepath

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Web Scraper - Extract content from websites and save to text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py https://example.com
  python scraper.py https://example.com --output my_scrapes
  python scraper.py https://httpbin.org/html --verbose
        """
    )
    
    parser.add_argument(
        "url", 
        help="Website URL to scrape (e.g., https://example.com)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default="documents",
        help="Output directory for text files (default: scraped_content)"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🔍 Starting scraper for URL: {args.url}")
        print(f"📁 Output directory: {args.output}")
    
    # Validate URL format
    if not args.url.startswith(('http://', 'https://')):
        print("❌ Error: URL must start with http:// or https://")
        return
    
    # Perform scraping
    result = scrape_website(args.url, args.output)
    
    if result:
        if args.verbose:
            print(f"🎉 Scraping completed successfully!")
    else:
        print("❌ Scraping failed. Check the URL and try again.")

# --- Example Usage ---

if __name__ == "__main__":
    main()