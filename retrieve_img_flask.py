import asyncio
import json
import os
import hashlib
import logging
import csv
import aiohttp
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from flask import Flask, jsonify, send_from_directory
from flask_restx import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Initialize Flask-RESTX API
api = Api(app,
          version='1.0',
          title='Image Scraper API',
          description='API to scrape and download images from websites listed in a CSV file.',
          doc='/docs') # This sets the Swagger UI endpoint

# Define a Namespace for your scraping operations
scraper_ns = api.namespace('scraper', description='Website image scraping operations')

# --- Your existing asynchronous functions (no changes needed here for logic) ---

async def download_image(session, img_url, output_path):
    """Downloads an image asynchronously and saves it to a file."""
    try:
        async with session.get(img_url, timeout=30) as response:
            if response.status == 200:
                content_type = response.headers.get('Content-Type', '').split('/')
                extension = content_type[1] if len(content_type) > 1 else 'jpg'
                img_name = hashlib.md5(img_url.encode()).hexdigest()
                file_path = os.path.join(output_path, f"{img_name}.{extension}")

                async with open(file_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(4096)
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
            await page.goto(url, timeout=30000)
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
    sanitized_name = f"{parsed_url.netloc}{parsed_url.path}".replace('.', '_').replace('/', '_').replace(':', '')
    return sanitized_name.strip('_')

async def scrape_and_save_images(url):
    """Scrapes image URLs from a URL, downloads them, and saves them to a dedicated folder."""
    logger.info(f"Crawling {url} for image URLs...")
    image_urls = await crawl_website(url)
    
    if not image_urls:
        logger.warning(f"No image URLs found on {url}.")
        return {"url": url, "status": "no image URLs found"}
    
    logger.info(f"Found {len(image_urls)} image URLs on {url}. Starting downloads...")
    
    folder_name = create_sanitized_folder_name(url)
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    download_results = []
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, img_url, output_dir) for img_url in image_urls]
        download_results = await asyncio.gather(*tasks)
    
    summary_file_path = os.path.join(output_dir, "download_summary.json")
    with open(summary_file_path, 'w') as f:
        json.dump(download_results, f, indent=2)

    logger.info(f"All images for {url} processed. Summary saved to {summary_file_path}")
    return {"url": url, "status": "completed image download", "total_images_processed": len(image_urls), "output_folder": output_dir}

# --- Flask-RESTX Endpoint ---

# Define the parser for file upload
csv_upload_parser = reqparse.RequestParser()
csv_upload_parser.add_argument('file',
                                type=FileStorage,
                                location='files',
                                required=True,
                                help='CSV file containing URLs (one URL per line)')

@scraper_ns.route('/download-images-from-csv')
class ScrapeAndDownloadImages(Resource):
    @api.expect(csv_upload_parser)
    @api.doc(description='Upload a CSV file with URLs to scrape images and download them.')
    async def post(self):
        """
        Uploads a CSV file and triggers image scraping and downloading.
        """
        args = csv_upload_parser.parse_args()
        uploaded_file: FileStorage = args['file']

        if not uploaded_file.filename.endswith('.csv'):
            api.abort(400, "Invalid file type. Please upload a CSV file.")
        
        urls_to_scrape = []
        csv_content = uploaded_file.stream.read().decode('utf-8').splitlines()
        csv_reader = csv.reader(csv_content)
        for row in csv_reader:
            if row:
                url = row[0].strip()
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                urls_to_scrape.append(url)
        
        if not urls_to_scrape:
            return {"message": "No URLs found in the CSV file"}, 200

        logger.info(f"Received {len(urls_to_scrape)} URLs from CSV. Starting image download process.")
        # Note: Using asyncio.run_coroutine_threadsafe if running in a sync Flask context,
        # but with hypercorn (which handles async), direct await is fine in an async resource.
        results = await asyncio.gather(*[scrape_and_save_images(url) for url in urls_to_scrape])
        return jsonify(results), 200

# Optional: Add a simple home page to see if the server is running
@app.route('/')
def home():
    return "Welcome to the Image Scraper API! Go to <a href='/docs'>/docs</a> for the Swagger UI."

# --- Main execution block ---
if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:5000"]

    print("---")
    print("Flask app is running. Open your browser to http://127.0.0.1:5000/docs for Swagger UI.")
    print("---")

    asyncio.run(hypercorn.asyncio.serve(app, config))