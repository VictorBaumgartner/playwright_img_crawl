import asyncio
import json
from playwright.async_api import async_playwright
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO

# Load CLIP model for image description
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# Function to fetch image from URL and get description using CLIP model
def get_image_description(image_url):
    try:
        # Download image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Process image and generate description using CLIP model
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        outputs = clip_model.get_text_features(**inputs)
        description = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy().tolist()

        return description
    except Exception as e:
        return {"error": f"Failed to describe image: {e}"}

# Function to perform image moderation using OpenNSFW model (dummy example here)
def moderate_image(image_url):
    # A real moderation service could be plugged here.
    # Dummy moderation logic: images with "adult" content or similar can be flagged
    try:
        # Using a simple approach: Check if the image URL contains any "adult" terms
        if "adult" in image_url.lower() or "sex" in image_url.lower():
            return "sensitive"
        else:
            return "normal"
    except Exception as e:
        return {"error": f"Failed to moderate image: {e}"}

async def crawl_and_process_images(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        # Wait for images to load
        await page.wait_for_selector('img')

        # Extract all image elements
        images = await page.query_selector_all('img')

        # Prepare JSON output structure
        output = {
            "url": url,
            "images": [],
            "stats": {
                "total_images": 0,
                "sensitive_content": 0
            }
        }

        # Process each image
        for img in images:
            src = await img.get_attribute('src')
            alt = await img.get_attribute('alt')
            title = await img.get_attribute('title')
            width = await img.get_attribute('width')
            height = await img.get_attribute('height')

            # Get image description
            description = get_image_description(src)

            # Perform moderation
            moderation_status = moderate_image(src)

            # Image data structure
            image_data = {
                "src": src,
                "alt": alt or '',
                "title": title or '',
                "dimensions": {
                    "width": width or '',
                    "height": height or ''
                },
                "analysis": {
                    "description": description,
                    "content_type": moderation_status
                }
            }

            # Update sensitive content stats
            if moderation_status != "normal":
                output['stats']['sensitive_content'] += 1

            output['images'].append(image_data)

        # Update total image count
        output['stats']['total_images'] = len(output['images'])

        # Close the browser
        await browser.close()

        return output

# Main function to run the crawler
async def main():
    # Get the website URL from user input
    url = input("Enter the website URL to crawl: ")

    # Run the crawling and processing function
    result = await crawl_and_process_images(url)

    # Print the JSON output
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Optionally, save to a JSON file
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
