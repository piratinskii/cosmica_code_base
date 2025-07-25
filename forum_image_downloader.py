import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
import re
import random
import time
from typing import Set, Optional


class ForumImageDownloader:
    """
    Automated image downloader for astronomy forum threads.
    Designed to collect astronomical images from forum attachments for dataset creation.
    """
    
    def __init__(self, base_url: str, output_folder: Optional[str] = None):
        """
        Initialize the downloader with forum URL and output settings.
        
        Args:
            base_url: Base URL of the forum thread
            output_folder: Custom output folder name (defaults to timestamp)
        """
        self.base_url = base_url
        self.output_folder = output_folder or datetime.now().strftime('%m%d%Y_%H%M%S')
        self.downloaded_urls: Set[str] = set()
        self.session = requests.Session()
        
        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)
        
    def download_thread_images(self, start_page: int, end_page: int, delay: float = 1.0):
        """
        Download all images from a forum thread across specified page range.
        
        Args:
            start_page: Starting page number
            end_page: Ending page number
            delay: Delay between requests to avoid overwhelming the server
        """
        print(f"Starting download from page {start_page} to {end_page}")
        print(f"Output folder: {self.output_folder}")
        
        for page in range(start_page, end_page + 1):
            page_url = f"{self.base_url}.{(page - 1) * 20}.html"
            print(f"Processing page: {page_url}")
            
            try:
                self._process_page(page_url)
                time.sleep(delay)  # Be respectful to the server
            except Exception as e:
                print(f"Error processing page {page}: {e}")
                continue
                
        print(f"Download completed. Total images: {len(self.downloaded_urls)}")
    
    def _process_page(self, page_url: str):
        """Process a single forum page and extract image attachments."""
        response = self.session.get(page_url)
        if response.status_code != 200:
            print(f"Failed to load page: {page_url}")
            return
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find main content block
        main_block = soup.find("td", id="sp_center")
        if not main_block:
            print("Main content block not found on page.")
            return
            
        # Find all attachment blocks
        attachments = main_block.find_all("div", class_="attachments smalltext")
        
        for attachment in attachments:
            for link in attachment.find_all("a", href=True):
                href = link["href"]
                if ";image" not in href:  # Skip thumbnails
                    full_image_url = urljoin(page_url, href)
                    if full_image_url not in self.downloaded_urls:
                        self._save_image(full_image_url, link.text.strip())
                        self.downloaded_urls.add(full_image_url)
    
    def _save_image(self, img_url: str, fallback_name: str):
        """Download and save a single image."""
        try:
            response = self.session.get(img_url, stream=True)
            if response.status_code == 200:
                filename = self._generate_filename(img_url, fallback_name)
                filepath = os.path.join(self.output_folder, filename)
                
                with open(filepath, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                        
                print(f"Saved: {filepath}")
            else:
                print(f"Failed to download image: {img_url}")
        except Exception as e:
            print(f"Error saving {img_url}: {e}")
    
    def _generate_filename(self, url: str, fallback_name: str) -> str:
        """
        Generate a unique filename from URL or fallback name.
        
        Args:
            url: Image URL
            fallback_name: Alternative name from link text
            
        Returns:
            Clean, unique filename
        """
        # Extract attachment ID from URL
        query_params = parse_qs(urlparse(url).query)
        attach_id = query_params.get("attach", [""])[0]
        
        # Use fallback name or generate from attachment ID
        filename = fallback_name or f"attachment_{attach_id}"
        
        # Clean filename of invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Ensure unique filename
        base, ext = os.path.splitext(filename)
        counter = 0
        while os.path.exists(os.path.join(self.output_folder, filename)):
            counter += 1
            filename = f"{base}_{counter}{ext}"
            
        return filename


def main():
    """Main execution function for automated forum image collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download images from astronomy forum threads')
    parser.add_argument('--url', required=True, help='Base URL of the forum thread')
    parser.add_argument('--start', type=int, required=True, help='Starting page number')
    parser.add_argument('--end', type=int, required=True, help='Ending page number')
    parser.add_argument('--output', help='Output folder name (default: timestamp)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests in seconds')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ForumImageDownloader(args.url, args.output)
    
    # Download images
    downloader.download_thread_images(args.start, args.end, delay=args.delay)


if __name__ == "__main__":
    main() 