import asyncio
import json
import base64
import requests
from pathlib import Path
from typing import List, Dict, Set
import hashlib
import logging
from tqdm import tqdm
from playwright.async_api import async_playwright

# Configuration
OLLAMA_ENDPOINT = "http://192.168.0.58:11434"
CACHE_DIR = Path("./image_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = 4
TIMEOUT = 30  # Seconds

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaVisionClient:
    def __init__(self, vision_model="llava", text_model="llama3.2"):
        """
        Initialize the client with specified models
        
        Args:
            vision_model: Model name for vision tasks
            text_model: Model name for text/moderation tasks
        """
        self.vision_model = vision_model
        self.text_model = text_model
        self.session = requests.Session()
        
        # Just verify connection to Ollama server without model checking
        self.verify_ollama_connection()
        
    def verify_ollama_connection(self):
        """Check if Ollama server is running"""
        try:
            # Verify server connection only
            response = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
                
        except Exception as e:
            logger.error(f"Ollama server connection failed: {str(e)}")
            raise

    async def analyze_image(self, image_url: str) -> Dict:
        """Process image through local LLM pipeline"""
        try:
            img_data = await self._get_image(image_url)
            base64_img = base64.b64encode(img_data).decode('utf-8')
            
            # Use available vision model
            logger.info(f"Analyzing image with {self.vision_model}...")
            description = await self._query_ollama_vision(
                model=self.vision_model,
                image=base64_img,
                prompt="Describe this image objectively and concisely."
            )
            
            # Use available text model for moderation
            logger.info(f"Running moderation with {self.text_model}...")
            moderation = await self._query_ollama_moderation(
                model=self.text_model,
                image=base64_img
            )
            
            return {
                "image_url": image_url,
                "description": description,
                "moderation": moderation,
                "hash": hashlib.md5(img_data).hexdigest()
            }
        except Exception as e:
            logger.error(f"Image analysis failed for {image_url}: {str(e)}")
            return {"image_url": image_url, "error": str(e)}

    async def _get_image(self, image_url: str) -> bytes:
        """Get image data from URL or local path"""
        # Check if this is a URL or local file
        if image_url.startswith(('http://', 'https://')):
            # It's a URL
            try:
                logger.info(f"Downloading image from URL: {image_url}")
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(image_url, timeout=TIMEOUT)
                )
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.error(f"Failed to download image: {str(e)}")
                raise
        else:
            # Assume it's a local file
            try:
                logger.info(f"Reading local image file: {image_url}")
                path = Path(image_url)
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_url}")
                return path.read_bytes()
            except Exception as e:
                logger.error(f"Failed to read local image: {str(e)}")
                raise

    async def _query_ollama_vision(self, model: str, image: str, prompt: str) -> str:
        """Query vision model with correct API endpoint"""
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image],
            "stream": False,
            "options": {"temperature": 0.2}
        }
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(
                    f"{OLLAMA_ENDPOINT}/api/generate",
                    json=payload,
                    timeout=TIMEOUT
                )
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Vision query failed: {str(e)}")
            return "Vision analysis failed"

    async def _query_ollama_moderation(self, model: str, image: str) -> Dict:
        """Query moderation model with specialized endpoint"""
        payload = {
            "model": model,
            "prompt": """Analyze for NSFW content. Respond with JSON: 
                        {"nsfw": boolean, "confidence": 0-1, "categories": []}""",
            "images": [image] if model.startswith("llava") else [],  # Only send image if it's a vision model
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(
                    f"{OLLAMA_ENDPOINT}/api/generate",
                    json=payload,
                    timeout=TIMEOUT
                )
            )
            response.raise_for_status()
            
            # Handle possible JSON parsing errors
            try:
                result = json.loads(response.json().get("response", "{}"))
                return result
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON response from moderation model")
                return {"nsfw": False, "confidence": 0.0, "categories": ["parsing_failed"]}
                
        except Exception as e:
            logger.error(f"Moderation query failed: {str(e)}")
            return {"nsfw": False, "confidence": 0.0, "categories": ["analysis_failed"]}

    async def process_batch(self, image_urls: List[str]) -> List[Dict]:
        """Process a batch of images concurrently"""
        tasks = [self.analyze_image(url) for url in image_urls]
        results = []
        
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing images"):
            result = await task
            results.append(result)
            
        return results

async def extract_images_with_playwright(url: str) -> List[str]:
    image_urls = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)
        await page.wait_for_load_state('networkidle')

        # Scroll to the bottom to load lazy images
        previous_height = None
        while True:
            current_height = await page.evaluate("document.body.scrollHeight")
            if previous_height == current_height:
                break
            previous_height = current_height
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)  # Wait for new images to load

        # Extract all image sources
        img_elements = await page.query_selector_all("img")
        for img in img_elements:
            src = await img.get_attribute("src")
            if src and not src.startswith("data:"):
                image_urls.append(src)

        await browser.close()
    return image_urls


async def main():
    """Main entry point for the script"""
    # Get website URL from user
    website_url = input("Enter website URL to extract and analyze images from: ")
    
    # Extract images from website
    image_urls = await extract_images_with_playwright(website_url)
    
    if not image_urls:
        logger.error("No images found on the website")
        return
        
    logger.info(f"Found {len(image_urls)} images")
    logger.info("Image URLs:")
    for i, url in enumerate(image_urls):
        logger.info(f"{i+1}. {url}")
    
    logger.info("Initializing LlamaVision Client...")
    
    client = LlamaVisionClient(
        vision_model="llava", 
        text_model="llama3.2"
    )
    
    logger.info(f"Processing {len(image_urls)} images...")
    results = await client.process_batch(image_urls)
    
    # Save results to file
    output_file = Path("analysis_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Analysis complete! Results saved to {output_file}")
    
    # Print sample result
    if results:
        logger.info("\nSample result:")
        logger.info(json.dumps(results[0], indent=2))

if __name__ == "__main__":
    """Execute the main function when the script is run"""
    asyncio.run(main())