import asyncio
import json
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path("./image_cache")
CACHE_DIR.mkdir(exist_ok=True)
MAX_WORKERS = 4
MODEL_CHOICE = "base"  # "base" or "large"

class ImageAnalyzer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        try:
            # Initialize image captioning model
            model_name = f"Salesforce/blip-image-captioning-{MODEL_CHOICE}"
            logger.info(f"Loading model: {model_name}")
            self.image_to_text = pipeline(
                "image-to-text", 
                model=model_name,
                use_fast=True
            )
            
            # Initialize NSFW detector
            logger.info("Loading NSFW detector")
            self.nsfw_detector = pipeline(
                "image-classification", 
                model="Falconsai/nsfw_image_detection",
                use_fast=True
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def _get_image_from_url(self, url):
        """Cache downloaded images with local storage"""
        cache_file = CACHE_DIR / f"{hashlib.md5(url.encode()).hexdigest()}.jpg"
        
        if cache_file.exists():
            return Image.open(cache_file)
            
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img.save(cache_file)
            return img
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            raise

    async def describe_image(self, url):
        """Generate image description"""
        try:
            img = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._get_image_from_url,
                url
            )
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.image_to_text,
                img
            )
            return result[0]['generated_text']
        except Exception as e:
            logger.warning(f"Description failed for {url}: {str(e)}")
            return "Could not generate description"

    async def moderate_image(self, url):
        """Classify image content"""
        try:
            img = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._get_image_from_url,
                url
            )
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.nsfw_detector,
                img
            )
            
            # Process results
            scores = {pred['label']: pred['score'] for pred in result}
            if scores.get('nsfw', 0) > 0.7:
                return "sexual", scores['nsfw']
            elif scores.get('violence', 0) > 0.7:
                return "violent", scores['violence']
            return "normal", 0.0
        except Exception as e:
            logger.warning(f"Moderation failed for {url}: {str(e)}")
            return "normal", 0.0

async def crawl_website(url):
    """Alternative to AsyncWebCrawler using Playwright"""
    from playwright.async_api import async_playwright
    
    images = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        try:
            await page.goto(url, timeout=15000)
            
            # Extract all images
            elements = await page.query_selector_all('img')
            for element in elements:
                src = await element.get_attribute('src')
                alt = await element.get_attribute('alt') or ''
                if src and src.startswith(('http://', 'https://')):
                    images.append({"src": src, "alt": alt})
                    
        except Exception as e:
            logger.error(f"Crawling failed: {str(e)}")
        finally:
            await browser.close()
    
    return images

async def main():
    try:
        analyzer = ImageAnalyzer()
        url = input("Enter website URL: ").strip()
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        logger.info(f"Crawling {url}...")
        images = await crawl_website(url)
        
        if not images:
            logger.warning("No images found on the page")
            return
            
        logger.info(f"Found {len(images)} images. Analyzing...")
        
        output = {
            "url": url,
            "model_used": f"blip-image-captioning-{MODEL_CHOICE}",
            "images": [],
            "stats": {"total_images": 0, "sensitive_content": 0}
        }
        
        # Process with progress bar
        tasks = [process_image(analyzer, img) for img in images]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            image_data = await future
            if image_data:
                output['images'].append(image_data)
                if image_data['analysis']['content_type'] != "normal":
                    output['stats']['sensitive_content'] += 1
        
        output['stats']['total_images'] = len(output['images'])
        
        # Save results
        result_file = r"C:\Users\victo\Desktop\CS\Job\img_crawling\results_{hashlib.md5(url.encode()).hexdigest()}.json"        
        with open(result_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {result_file}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")

async def process_image(analyzer, img_data):
    """Process single image with error handling"""
    try:
        src = img_data.get('src', '')
        if not src.startswith(('http://', 'https://')):
            return None
            
        description, (content_type, confidence) = await asyncio.gather(
            analyzer.describe_image(src),
            analyzer.moderate_image(src)
        )
        
        return {
            "src": src,
            "alt": img_data.get('alt', ''),
            "analysis": {
                "description": description,
                "content_type": content_type,
                "confidence": float(confidence)
            }
        }
    except Exception as e:
        logger.warning(f"Failed to process image {img_data.get('src')}: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(main())