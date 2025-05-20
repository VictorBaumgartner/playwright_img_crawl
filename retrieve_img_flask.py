import asyncio
import json
import os
import hashlib
import logging
import csv
import aiohttp # For downloading images asynchronously
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from flask import Flask, request, jsonify

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

async def download_image(session, img_url, output_path):
    """Downloads an image asynchronously and saves it to a file."""
    try:
        async with session.get(img_url, timeout=30) as response:
            if response.status == 200:
                # Get the image content type to determine file extension
                content_type = response.headers.get('Content-Type', '').split('/')
                extension = content_type[1] if len(content_type) > 1 else 'jpg' # Default to jpg

                # Use a hash of the URL for a unique filename
                img_name = hashlib.md5(img_url.encode()).hexdigest()
                file_path = os.path.join(output_path, f"{img_name}.{extension}")

                async with open(file_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(4096) # Read in chunks
                        if not chunk:
                            break
                        f.write(chunk)
                logger.info(f"Downloaded: {img_url} to {file_path}")
                return {"src": img_url, "downloaded_path": file_path, "status": "success"}
            else:
                logger.warning(f"Failed to download {img_url}: Status {response.status}")
                return {"src": img_url, "status": f"failed_http_{response.status}"}
    except asyncio.TimeoutError:
        logger.error(f"Timeout downloading {img_url}")
        return {"src": img_url, "status": "failed_timeout"}
    except Exception as e:
        logger.error(f"Error downloading {img_url}: {e}")
        return {"src": img_url, "status": f"failed_error_{str(e)}"}

async def crawl_website(url):
    """Extract all image URLs from a webpage."""
    image_srcs = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url, timeout=30000)  # Increased timeout
            
            # Extract all image sources
            elements = await page.query_selector_all('img')
            for element in elements:
                src = await element.get_attribute('src')
                if src and (src.startswith('http://') or src.startswith('https://')):
                    image_srcs.append(src)

        except Exception as e:
            logger.error(f"Crawling failed for {url}: {str(e)}")
        finally:
            await browser.close()

    return image_srcs

def create_sanitized_folder_name(url):
    """Creates a sanitized folder name from a URL."""
    parsed_url = urlparse(url)
    # Use netloc (domain) and path for a more specific folder name
    sanitized_name = f"{parsed_url.netloc}{parsed_url.path}".replace('.', '_').replace('/', '_').replace(':', '')
    return sanitized_name.strip('_') # Remove any leading/trailing underscores

async def scrape_and_save_images(url):
    """Scrapes image URLs from a URL, downloads them, and saves them to a dedicated folder."""
    logger.info(f"Crawling {url} for image URLs...")
    image_urls = await crawl_website(url)
    
    if not image_urls:
        logger.warning(f"No image URLs found on {url}.")
        return {"url": url, "status": "no image URLs found"}
    
    logger.info(f"Found {len(image_urls)} image URLs on {url}. Starting downloads...")
    
    # Create a folder named by the sanitized URL
    folder_name = create_sanitized_folder_name(url)
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    download_results = []
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, img_url, output_dir) for img_url in image_urls]
        download_results = await asyncio.gather(*tasks)
    
    # Save a JSON file with download results for reference
    summary_file_path = os.path.join(output_dir, "download_summary.json")
    with open(summary_file_path, 'w') as f:
        json.dump(download_results, f, indent=2)

    logger.info(f"All images for {url} processed. Summary saved to {summary_file_path}")
    return {"url": url, "status": "completed image download", "total_images_processed": len(image_urls), "output_folder": output_dir}

## Flask Endpoint for Image Download

@app.route('/scrape-and-download-images-from-csv', methods=['POST'])
async def scrape_and_download_images_from_csv_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        urls_to_scrape = []
        # Read the CSV file
        csv_content = file.stream.read().decode('utf-8').splitlines()
        csv_reader = csv.reader(csv_content)
        for row in csv_reader:
            if row: # Ensure row is not empty
                url = row[0].strip()
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                urls_to_scrape.append(url)
        
        if not urls_to_scrape:
            return jsonify({"message": "No URLs found in the CSV file"}), 200

        logger.info(f"Received {len(urls_to_scrape)} URLs from CSV. Starting image download process.")
        results = await asyncio.gather(*[scrape_and_save_images(url) for url in urls_to_scrape])
        return jsonify(results), 200
    else:
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:5000"]

    print("---")
    print("Flask app is running. Send a POST request to /scrape-and-download-images-from-csv with your CSV file.")
    print("Example: curl -X POST -F 'file=@your_urls.csv' [http://127.0.0.1:5000/scrape-and-download-images-from-csv](http://127.0.0.1:5000/scrape-and-download-images-from-csv)")
    print("---")

    asyncio.run(hypercorn.asyncio.serve(app, config))