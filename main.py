import asyncio
import os
import random
import time
import logging
import sys
import base64
from pathlib import Path
from urllib.parse import unquote
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from playwright.async_api import async_playwright
import aiohttp
import aiofiles
from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

try:
    from config import (PROXY_ENABLED, PROXY_LIST, PROXY_ROTATION_MODE, PROXY_MAX_RETRIES,
                       AI_MODEL, OPENAI_API_KEY, GEMINI_API_KEY, OPENAI_MODEL, GEMINI_MODEL)
except ImportError:
    PROXY_ENABLED = False
    PROXY_LIST = []
    PROXY_ROTATION_MODE = 'round-robin'
    PROXY_MAX_RETRIES = 2
    AI_MODEL = 'openai'
    OPENAI_API_KEY = None
    GEMINI_API_KEY = None
    OPENAI_MODEL = 'gpt-5-nano'
    GEMINI_MODEL = 'gemini-1.5-flash'

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = '1WCmFLdioC-mm_QtWdA3qNo_sEdEHZ3HTyCdqCaz_2MA'
SHEET_NAME = 'Daily Results'
COLUMN_NAME = 'Tag Account'
COUNT_COLUMN_NAME = 'count'

DOWNLOAD_FOLDER = 'instagram_downloads'
BASE_URL = 'https://fastdl.app/en2'
MAX_POSTS_PER_ACCOUNT = 12

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                f'instagram_downloader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                encoding='utf-8',
                errors='replace'
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

class ProxyManager:
    def __init__(self, proxy_list, rotation_mode='round-robin'):
        self.proxy_list = proxy_list if proxy_list else []
        self.rotation_mode = rotation_mode
        self.current_index = 0
        self.proxy_stats = {proxy: {'success': 0, 'failures': 0} for proxy in self.proxy_list}

        if self.proxy_list:
            logger.info(f"Proxy Manager initialized with {len(self.proxy_list)} proxies")
            logger.info(f"Rotation mode: {self.rotation_mode}")
        else:
            logger.info("Proxy Manager: No proxies configured (direct connection)")

    def get_next_proxy(self):
        if not self.proxy_list:
            return None

        if self.rotation_mode == 'random':
            proxy = random.choice(self.proxy_list)
        else:
            proxy = self.proxy_list[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxy_list)

        logger.debug(f"Selected proxy: {self._mask_proxy(proxy)}")
        return proxy

    def mark_proxy_success(self, proxy):
        if proxy and proxy in self.proxy_stats:
            self.proxy_stats[proxy]['success'] += 1

    def mark_proxy_failure(self, proxy):
        if proxy and proxy in self.proxy_stats:
            self.proxy_stats[proxy]['failures'] += 1
            logger.warning(f"Proxy failed: {self._mask_proxy(proxy)} (failures: {self.proxy_stats[proxy]['failures']})")

    def get_proxy_stats(self):
        return self.proxy_stats

    def _mask_proxy(self, proxy):
        if not proxy:
            return "None"
        if '@' in proxy:
            parts = proxy.split('@')
            protocol_creds = parts[0].split('://')
            if len(protocol_creds) == 2:
                protocol = protocol_creds[0]
                return f"{protocol}://***:***@{parts[1]}"
        return proxy

class AIAnalyzer:
    def __init__(self, model_type='openai'):
        self.model_type = model_type.lower()
        
        if self.model_type == 'openai':
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found")
            self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            self.model_name = OPENAI_MODEL
            logger.info(f"AI Analyzer initialized with OpenAI model: {self.model_name}")
        elif self.model_type == 'gemini':
            if not GEMINI_API_KEY:
                raise ValueError("Gemini API key not found")
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            self.model_name = GEMINI_MODEL
            logger.info(f"AI Analyzer initialized with Gemini model: {self.model_name}")
            if 'exp' in GEMINI_MODEL:
                logger.info("Using experimental model - FREE but may change in future")
        else:
            raise ValueError(f"Invalid model type: {model_type}. Choose 'openai' or 'gemini'")

    def encode_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    async def analyze_image_openai(self, image_path):
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return False

        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Does this image contain swimwear, bikini, or swimming suit? This includes both people wearing swimwear and just the swimwear items themselves. Answer with only 'YES' or 'NO'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )

        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer

    async def analyze_image_gemini(self, image_path):
        try:
            import PIL.Image
            img = PIL.Image.open(image_path)
            
            prompt = "Does this image contain swimwear, bikini, or swimming suit? This includes both people wearing swimwear and just the swimwear items themselves. Answer with only 'YES' or 'NO'."
            
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                [prompt, img]
            )
            
            answer = response.text.strip().upper()
            has_swimwear = "YES" in answer
            
            logger.debug(f"Gemini raw response: {answer}")
            
            return has_swimwear
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return False

    async def detect_swimwear(self, image_path, image_number=None, total_images=None):
        try:
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return False

            if image_path.stat().st_size == 0:
                logger.error(f"Image file is empty: {image_path}")
                return False

            progress = f"[{image_number}/{total_images}] " if image_number and total_images else ""
            logger.info(f"{progress}🔍 Starting {self.model_type.upper()} analysis: {image_path.name}")

            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    if self.model_type == 'openai':
                        has_swimwear = await self.analyze_image_openai(image_path)
                    else:
                        has_swimwear = await self.analyze_image_gemini(image_path)

                    result_emoji = "✓ SWIMWEAR" if has_swimwear else "✗ NO SWIMWEAR"
                    logger.info(f"{progress}✅ {self.model_type.upper()} Result: {result_emoji}")
                    return has_swimwear

                except asyncio.CancelledError:
                    logger.warning(f"{progress}Analysis cancelled for {image_path.name}")
                    raise
                except Exception as api_error:
                    if attempt < max_retries - 1:
                        logger.warning(f"{progress}{self.model_type.upper()} API error (attempt {attempt + 1}/{max_retries}): {api_error}, retrying...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"{progress}{self.model_type.upper()} API error after {max_retries} attempts: {api_error}")
                        raise

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return False

class InstagramDownloader:
    def __init__(self):
        self.download_folder = Path(DOWNLOAD_FOLDER)
        self.download_folder.mkdir(exist_ok=True)
        self.sheet_client = None
        self.worksheet = None
        self.column_index = None
        self.count_column_index = None
        
        self.ai_analyzer = AIAnalyzer(model_type=AI_MODEL)

        if PROXY_ENABLED and PROXY_LIST:
            self.proxy_manager = ProxyManager(PROXY_LIST, PROXY_ROTATION_MODE)
            self.current_proxy = None
        else:
            self.proxy_manager = None
            self.current_proxy = None

        self.stats = {
            'total_usernames': 0,
            'processed_usernames': 0,
            'total_downloads': 0,
            'failed_downloads': 0,
            'skipped_videos': 0,
            'total_swimwear_detected': 0,
            'proxy_switches': 0
        }
    
    def _parse_proxy_url(self, proxy_url):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(proxy_url)
            proxy_dict = {
                'server': f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
            }
            if parsed.username:
                proxy_dict['username'] = parsed.username
            if parsed.password:
                proxy_dict['password'] = parsed.password
            return proxy_dict
        except Exception as e:
            logger.error(f"Error parsing proxy URL {proxy_url}: {e}")
            return None

    def init_google_sheets(self):
        if self.sheet_client is None:
            logger.info("Connecting to Google Sheets...")
            creds = Credentials.from_service_account_file(
                'credentials.json',
                scopes=SCOPES
            )
            self.sheet_client = gspread.authorize(creds)
            sheet = self.sheet_client.open_by_key(SPREADSHEET_ID)
            self.worksheet = sheet.worksheet(SHEET_NAME)

            headers = self.worksheet.row_values(1)
            try:
                self.column_index = headers.index(COLUMN_NAME) + 1
                logger.info(f"Column '{COLUMN_NAME}' found at index {self.column_index}")
            except ValueError:
                logger.error(f"Column '{COLUMN_NAME}' not found!")
                raise

            try:
                self.count_column_index = headers.index(COUNT_COLUMN_NAME) + 1
                logger.info(f"Column '{COUNT_COLUMN_NAME}' found at index {self.count_column_index}")
            except ValueError:
                logger.warning(f"Column '{COUNT_COLUMN_NAME}' not found, creating it...")
                next_col = len(headers) + 1
                self.worksheet.update_cell(1, next_col, COUNT_COLUMN_NAME)
                self.count_column_index = next_col
                logger.info(f"Column '{COUNT_COLUMN_NAME}' created at index {self.count_column_index}")
    
    def get_next_username(self, row_number):
        try:
            cell_value = self.worksheet.cell(row_number, self.column_index).value
            if cell_value and cell_value.strip():
                username = cell_value.strip().replace('@', '')
                if not username.startswith('#'):
                    return username
        except Exception as e:
            logger.debug(f"Row {row_number}: {e}")
        return None
    
    def count_total_usernames(self):
        try:
            column_values = self.worksheet.col_values(self.column_index)
            count = 0
            for value in column_values[1:]:
                if value and value.strip() and not value.strip().startswith('#'):
                    count += 1
            return count
        except:
            return 0

    async def analyze_images_parallel(self, downloaded_images, username):
        if not downloaded_images:
            logger.warning(f"No images to analyze for @{username}")
            return 0, 0

        total_images = len(downloaded_images)
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Collecting {AI_MODEL.upper()} analysis results for @{username}")
        logger.info(f"Total images: {total_images}")
        logger.info(f"{'='*60}\n")

        swimwear_count = 0
        successful_analyses = 0
        failed_analyses = 0

        for i, (filepath, task) in enumerate(downloaded_images, 1):
            try:
                logger.info(f"⏳ [{i}/{total_images}] Waiting for analysis: {filepath.name}")
                has_swimwear = await task
                successful_analyses += 1

                if has_swimwear:
                    swimwear_count += 1
                    logger.info(f"    ✅ SWIMWEAR DETECTED!")
                else:
                    logger.info(f"    ❌ No swimwear")

            except asyncio.CancelledError:
                logger.warning(f"[{i}/{total_images}] Analysis was cancelled for {filepath.name}")
                failed_analyses += 1
            except Exception as e:
                logger.error(f"[{i}/{total_images}] Error getting analysis result for {filepath.name}: {e}")
                failed_analyses += 1

        logger.info(f"\n{'-'*60}")
        logger.info(f"📈 Analysis Summary for @{username}:")
        logger.info(f"  🏊 Swimwear detected: {swimwear_count}/{total_images}")
        logger.info(f"  ✅ Successful: {successful_analyses}")
        logger.info(f"  ❌ Failed: {failed_analyses}")
        logger.info(f"{'-'*60}\n")

        return swimwear_count, total_images

    def update_count_in_sheet(self, row_number, swimwear_count, total_count):
        try:
            count_value = f"{swimwear_count}/{total_count}"
            self.worksheet.update_cell(row_number, self.count_column_index, count_value)
            logger.info(f"Updated Google Sheets row {row_number}: count = {count_value}")
            return True
        except Exception as e:
            logger.error(f"Error updating Google Sheets: {e}")
            return False
    
    async def download_image(self, session, url, filepath):
        try:
            if not isinstance(filepath, Path):
                filepath = Path(filepath)

            filepath = filepath.absolute()
            parent_dir = filepath.parent
            logger.info(f"Creating directory: {parent_dir}")

            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Directory ready: {parent_dir}")
            except Exception as mkdir_error:
                logger.error(f"✗ Failed to create directory {parent_dir}: {mkdir_error}")
                return False

            if not parent_dir.exists():
                logger.error(f"✗ Directory does not exist after mkdir: {parent_dir}")
                return False

            logger.info(f"Downloading from URL to: {filepath}")

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    async with aiofiles.open(filepath, 'wb') as f:
                        await f.write(await response.read())
                    file_size = filepath.stat().st_size / 1024
                    logger.info(f"✓ Downloaded {filepath.name} ({file_size:.1f} KB)")
                    return True
                else:
                    logger.warning(f"✗ Failed to download: HTTP {response.status}")
                    return False
        except asyncio.TimeoutError:
            logger.error(f"✗ Timeout downloading image")
            return False
        except Exception as e:
            logger.error(f"✗ Error downloading: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def random_wait(self, min_sec=2, max_sec=4):
        wait_time = random.uniform(min_sec, max_sec)
        await asyncio.sleep(wait_time)
    
    async def slow_scroll_and_load_posts(self, page, target_count):
        previous_count = 0
        no_change_attempts = 0
        max_no_change = 4
        
        logger.debug(f"Starting slow scroll, target: {target_count} posts")
        
        while no_change_attempts < max_no_change:
            current_posts = await page.locator('.profile-media-list__item').count()
            
            if current_posts >= target_count:
                logger.debug(f"Reached target: {current_posts} posts")
                break
            
            if current_posts == previous_count:
                no_change_attempts += 1
                logger.debug(f"No new posts (attempt {no_change_attempts}/{max_no_change})")
            else:
                no_change_attempts = 0
                previous_count = current_posts
                logger.debug(f"Loaded {current_posts} posts...")
            
            current_scroll = await page.evaluate('window.pageYOffset')
            target_scroll = current_scroll + 800
            
            await page.evaluate(f'window.scrollTo({{top: {target_scroll}, behavior: "smooth"}})')
            
            await self.random_wait(2.5, 3.5)
        
        final_count = await page.locator('.profile-media-list__item').count()
        logger.debug(f"Scrolling complete: {final_count} posts available")
    
    async def process_username(self, page, username, session, row_number, max_posts=MAX_POSTS_PER_ACCOUNT):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: @{username}")
        logger.info(f"{'='*60}")
        
        user_folder = self.download_folder / username
        user_folder.mkdir(exist_ok=True)
        logger.debug(f"Folder: {user_folder}")
        
        try:
            logger.info(f"Opening {BASE_URL}...")
            await page.goto(BASE_URL, wait_until='domcontentloaded', timeout=30000)
            await self.random_wait(2, 3)
            
            logger.info(f"Entering username: @{username}")
            input_selector = 'input#search-form-input'
            await page.wait_for_selector(input_selector, timeout=10000)
            
            await page.fill(input_selector, '')
            await self.random_wait(0.3, 0.5)
            
            await page.fill(input_selector, username)
            await self.random_wait(1.5, 2)
            
            logger.info("Clicking Download button...")
            download_button = 'button.search-form__button[type="submit"]'
            await page.click(download_button)
            
            await asyncio.sleep(2)

            logger.debug("Checking for ad modal...")
            try:
                close_button = page.locator('.ad-modal__close')
                if await close_button.count() > 0:
                    logger.debug("Ad modal detected - clicking close button")
                    await close_button.first.click()
                    await asyncio.sleep(1)
                    logger.debug("Ad modal closed")
                else:
                    logger.debug("No ad modal found")
            except Exception as e:
                logger.debug(f"Ad modal handling: {e}")

            await asyncio.sleep(1)
            
            logger.info("Waiting for results...")
            try:
                await page.wait_for_selector('.search-result', state='attached', timeout=20000)
                await asyncio.sleep(2)
                
                await page.evaluate("""
                    () => {
                        const result = document.querySelector('.search-result');
                        if (result) {
                            result.style.display = 'block';
                            result.style.visibility = 'visible';
                            result.style.opacity = '1';
                        }
                        
                        const modal = document.querySelector('.ad-modal__wrapper');
                        if (modal) {
                            modal.style.display = 'none';
                        }
                    }
                """)
                
                await asyncio.sleep(1)
                await self.random_wait(2, 3)
            except Exception as e:
                logger.error(f"No results for @{username}: {e}")
                return
            
            account_exists = await page.locator('.user-info__username').count() > 0
            if not account_exists:
                logger.warning(f"Account @{username} not found or private")
                return
            
            logger.info(f"Collecting up to {max_posts} images (1 per unique post)...")
            logger.info(f"📥 Download & AI analysis will run in parallel for speed")
            logger.info("")

            downloaded_count = 0
            skipped_videos = 0
            processed_post_captions = set()
            current_scroll_attempt = 0
            max_scroll_attempts = 20
            downloaded_images = []

            while downloaded_count < max_posts and current_scroll_attempt < max_scroll_attempts:
                posts = await page.locator('.profile-media-list__item').all()
                total_available = len(posts)

                logger.debug(f"Found {total_available} items, processing for unique posts...")

                found_new_post = False

                for post in posts:
                    if downloaded_count >= max_posts:
                        break

                    try:
                        video_tag = await post.locator('.tags__item--video').count()
                        is_video = video_tag > 0

                        if is_video:
                            try:
                                download_link = await post.locator('img.media-content__image').get_attribute('src')
                                if download_link:
                                    logger.debug(f"Video post - using thumbnail image")
                                    skipped_videos += 1
                                    self.stats['skipped_videos'] += 1
                                else:
                                    logger.debug(f"Video post without thumbnail - skipping")
                                    continue
                            except:
                                logger.debug(f"Could not get thumbnail from video post")
                                continue
                        else:
                            download_link = await post.locator('a.button__download').get_attribute('href')

                        if not download_link:
                            logger.debug(f"No download link found")
                            continue

                        try:
                            caption_elem = await post.locator('.media-content__caption').text_content()
                            time_elem = await post.locator('.media-content__meta-time').get_attribute('title')
                            post_identifier = f"{caption_elem[:50]}_{time_elem}"
                        except:
                            post_identifier = download_link[:100]

                        if post_identifier in processed_post_captions:
                            logger.debug(f"Skipping carousel image from already processed post")
                            continue

                        processed_post_captions.add(post_identifier)
                        found_new_post = True

                        filename = f"img_{downloaded_count + 1:03d}.jpg"
                        filepath = user_folder / filename

                        logger.info(f"Downloading image {downloaded_count + 1}/{max_posts}...")
                        success = await self.download_image(session, download_link, filepath)

                        if success:
                            downloaded_count += 1
                            self.stats['total_downloads'] += 1

                            logger.info(f"📷 Image {downloaded_count}: ✅ Downloaded")
                            logger.info(f"🤖 Starting {AI_MODEL.upper()} analysis in background...")
                            analysis_task = asyncio.create_task(
                                self.ai_analyzer.detect_swimwear(filepath, downloaded_count, max_posts)
                            )
                            downloaded_images.append((filepath, analysis_task))

                            await asyncio.sleep(0.3)
                        else:
                            self.stats['failed_downloads'] += 1

                        await self.random_wait(1.5, 2.5)

                    except Exception as e:
                        logger.error(f"Error processing post: {e}")
                        self.stats['failed_downloads'] += 1
                        continue

                if downloaded_count < max_posts:
                    if not found_new_post:
                        logger.debug(f"No new unique posts found, scrolling for more... (attempt {current_scroll_attempt + 1})")
                        current_scroll = await page.evaluate('window.pageYOffset')
                        target_scroll = current_scroll + 800
                        await page.evaluate(f'window.scrollTo({{top: {target_scroll}, behavior: "smooth"}})')
                        await self.random_wait(2.5, 3.5)
                        current_scroll_attempt += 1
                    else:
                        current_scroll_attempt = 0
                else:
                    break
            
            self.stats['processed_usernames'] += 1
            logger.info(f"\n{'✅'*30}")
            logger.info(f"✅ Completed @{username}")
            logger.info(f"{'✅'*30}")
            logger.info(f"  📷 Images downloaded: {downloaded_count}")
            logger.info(f"  🎥 Videos skipped: {skipped_videos}")
            logger.info(f"  📁 Saved to: {user_folder}")
            logger.info(f"{'='*60}\n")

            if downloaded_images:
                logger.info(f"\n{'='*60}")
                logger.info(f"🤖 {AI_MODEL.upper()} Analysis Results for @{username}")
                logger.info(f"(All images were analyzed in parallel)")
                logger.info(f"{'='*60}")

                swimwear_count, total_count = await self.analyze_images_parallel(downloaded_images, username)

                self.stats['total_swimwear_detected'] += swimwear_count

                logger.info(f"💾 Updating Google Sheets for @{username}...")
                self.update_count_in_sheet(row_number, swimwear_count, total_count)

                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 FINAL RESULT for @{username}:")
                logger.info(f"  🏊 Swimwear Count: {swimwear_count}/{total_count}")
                logger.info(f"  📊 Percentage: {(swimwear_count/total_count*100):.1f}%")
                logger.info(f"{'='*60}\n")
            elif downloaded_count > 0:
                logger.warning(f"Images were downloaded but no analysis tasks were created for @{username}")
            else:
                logger.warning(f"No images were downloaded for @{username}")

        except KeyboardInterrupt:
            logger.warning(f"\nUser interrupted processing of @{username}")

            if 'downloaded_images' in locals() and downloaded_images:
                logger.info(f"Cancelling {len(downloaded_images)} pending analysis tasks...")
                for filepath, task in downloaded_images:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
            raise

        except Exception as e:
            logger.error(f"Error processing @{username}: {e}", exc_info=True)

            if 'downloaded_images' in locals() and downloaded_images:
                logger.warning(f"Cancelling {len(downloaded_images)} pending analysis tasks due to error")
                for filepath, task in downloaded_images:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
    
    async def run(self, max_posts_per_account=MAX_POSTS_PER_ACCOUNT, start_row=2):
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("🚀 Instagram Downloader Started")
        logger.info("="*60)
        logger.info(f"🤖 AI Model: {AI_MODEL.upper()}")
        logger.info(f"📋 Model Name: {self.ai_analyzer.model_name}")
        if 'exp' in self.ai_analyzer.model_name:
            logger.info(f"💰 Cost: FREE (experimental)")
        logger.info("="*60)
        
        self.init_google_sheets()
        
        total_count = self.count_total_usernames()
        self.stats['total_usernames'] = total_count
        
        logger.info(f"\nTotal usernames in sheet: {total_count}")
        logger.info(f"Starting from row: {start_row}")
        logger.info(f"Max posts per account: {max_posts_per_account}\n")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-web-security',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-infobars',
                    '--window-position=0,0',
                    '--ignore-certifcate-errors',
                    '--ignore-certifcate-errors-spki-list',
                    '--disable-popup-blocking'
                ]
            )

            context_options = {
                'viewport': {'width': 1920, 'height': 1080},
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            if self.proxy_manager:
                self.current_proxy = self.proxy_manager.get_next_proxy()
                if self.current_proxy:
                    proxy_dict = self._parse_proxy_url(self.current_proxy)
                    if proxy_dict:
                        context_options['proxy'] = proxy_dict
                        logger.info(f"Using proxy: {self.proxy_manager._mask_proxy(self.current_proxy)}")
                    else:
                        logger.warning(f"Invalid proxy format: {self.current_proxy}, using direct connection")

            context = await browser.new_context(**context_options)
            
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                window.chrome = {
                    runtime: {}
                };
            """)
            
            ad_block_patterns = [
                '**/ads/**',
                '**/doubleclick.net/**',
                '**/google-analytics.com/**',
                '**/googletagmanager.com/**',
                '**/facebook.net/**',
                '**/analytics/**',
                '**/*ad*.js',
                '**/*analytics*.js',
                '**/*tracking*.js',
                '**/pagead2.googlesyndication.com/**',
                '**/adservice.google.com/**',
                '**/static.ads-twitter.com/**',
                '**/ads-twitter.com/**',
                '**/amazon-adsystem.com/**',
                '**/adnxs.com/**',
                '**/advertising.com/**',
                '**/bidswitch.net/**',
                '**/outbrain.com/**',
                '**/taboola.com/**',
                '**/criteo.com/**',
                '**/pubmatic.com/**',
                '**/*banner*.js',
                '**/*sponsor*.js',
                '**/*popup*.js',
                '**/googleads.g.doubleclick.net/**',
                '**/adsbygoogle.js',
                '**/*adsbygoogle*'
            ]
            
            for pattern in ad_block_patterns:
                await context.route(pattern, lambda route: route.abort())
            
            await context.add_init_script("""
                const blockAds = () => {
                    const adSelectors = [
                        'iframe[src*="doubleclick"]',
                        'iframe[src*="googleads"]',
                        'iframe[src*="adservice"]',
                        'ins.adsbygoogle[data-ad-client]'
                    ];
                    
                    adSelectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => {
                            if (el && el.parentElement) {
                                el.parentElement.remove();
                            }
                        });
                    });
                };
                
                setTimeout(blockAds, 2000);
                setTimeout(blockAds, 4000);
            """)
            
            page = await context.new_page()
            
            async with aiohttp.ClientSession() as session:
                current_row = start_row
                processed_count = 0
                
                while True:
                    username = self.get_next_username(current_row)
                    
                    if username is None:
                        consecutive_empty = 0
                        for check_row in range(current_row, current_row + 10):
                            if self.get_next_username(check_row) is None:
                                consecutive_empty += 1
                        
                        if consecutive_empty >= 10:
                            logger.info("\nReached end of usernames (10 empty rows)")
                            break
                        
                        current_row += 1
                        continue
                    
                    processed_count += 1
                    logger.info(f"\n[{processed_count}] Row {current_row}: Processing @{username}")

                    await self.process_username(page, username, session, current_row, max_posts_per_account)
                    
                    current_row += 1
                    
                    wait_time = random.randint(3, 5)
                    logger.info(f"Waiting {wait_time} seconds before next account...")
                    await asyncio.sleep(wait_time)
            
            await context.close()
            await browser.close()
        
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("🎉 DOWNLOAD & ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"🤖 AI Model Used: {AI_MODEL.upper()} ({self.ai_analyzer.model_name})")
        logger.info(f"📊 Total usernames: {self.stats['total_usernames']}")
        logger.info(f"✅ Processed successfully: {self.stats['processed_usernames']}")
        logger.info(f"📷 Images downloaded: {self.stats['total_downloads']}")
        logger.info(f"❌ Failed downloads: {self.stats['failed_downloads']}")
        logger.info(f"🎥 Videos skipped: {self.stats['skipped_videos']}")
        logger.info(f"🏊 Total swimwear detected: {self.stats['total_swimwear_detected']}")
        logger.info(f"⏱️  Total time: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
        logger.info(f"📁 Downloads saved to: {self.download_folder.absolute()}")

        if self.proxy_manager:
            logger.info(f"\n🌐 Proxy Statistics:")
            proxy_stats = self.proxy_manager.get_proxy_stats()
            for proxy, stats in proxy_stats.items():
                masked_proxy = self.proxy_manager._mask_proxy(proxy)
                logger.info(f"  {masked_proxy}: Success={stats['success']}, Failures={stats['failures']}")

        logger.info("="*60)

async def main():
    print("\n" + "="*60)
    print("Instagram Downloader - AI Powered")
    print(f"Current AI Model: {AI_MODEL.upper()}")
    print("="*60)
    
    downloader = InstagramDownloader()
    
    max_posts = input("\nHow many posts per account? (default: 12): ").strip()
    max_posts = int(max_posts) if max_posts.isdigit() and int(max_posts) > 0 else MAX_POSTS_PER_ACCOUNT
    
    start_row = input("Start from which row? (default: 2): ").strip()
    start_row = int(start_row) if start_row.isdigit() and int(start_row) >= 2 else 2
    
    print(f"\nSettings:")
    print(f"  - AI Model: {AI_MODEL.upper()}")
    print(f"  - Posts per account: {max_posts}")
    print(f"  - Starting from row: {start_row}")
    print(f"  - Download folder: {DOWNLOAD_FOLDER}")
    print(f"  - Google Sheet: {SPREADSHEET_ID}")
    print(f"  - Sheet name: {SHEET_NAME}")
    
    confirm = input("\nStart downloading? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print()
    await downloader.run(max_posts_per_account=max_posts, start_row=start_row)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
