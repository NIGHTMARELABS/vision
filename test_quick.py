import asyncio
import sys
from main import InstagramDownloader

TEST_USERNAMES_COUNT = 50
TEST_POSTS_PER_ACCOUNT = 10
TEST_START_ROW = 29

async def quick_test():
    print("\n" + "="*60)
    print(f"Quick Test - {TEST_USERNAMES_COUNT} usernames")
    print("="*60)

    downloader = InstagramDownloader()

    print("\nTest settings:")
    print(f"  - Posts per account: {TEST_POSTS_PER_ACCOUNT}")
    print(f"  - Starting from row: {TEST_START_ROW}")
    print(f"  - Will process only {TEST_USERNAMES_COUNT} usernames")
    
    confirm = input("\nStart test? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Test cancelled.")
        return
    
    downloader.init_google_sheets()

    test_usernames = []
    current_row = TEST_START_ROW

    while len(test_usernames) < TEST_USERNAMES_COUNT:
        username = downloader.get_next_username(current_row)
        if username:
            test_usernames.append((current_row, username))
        current_row += 1

        if current_row > 100:
            break
    
    if not test_usernames:
        print("\nNo usernames found in sheet!")
        return
    
    print(f"\nTest usernames:")
    for row, username in test_usernames:
        print(f"  Row {row}: @{username}")
    
    print()
    
    from playwright.async_api import async_playwright
    import aiohttp
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-web-security'
            ]
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        # NO AD BLOCKING - AdGuard Chrome extension handles all ad blocking
        # Code-based ad blocking was interfering with site functionality

        page = await context.new_page()
        
        async with aiohttp.ClientSession() as session:
            for row, username in test_usernames:
                print(f"\nTesting @{username} (row {row})...")
                await downloader.process_username(page, username, session, row, max_posts=TEST_POSTS_PER_ACCOUNT)
                await asyncio.sleep(3)
        
        await context.close()
        await browser.close()
    
    print("\n" + "="*60)
    print("✅ Test completed!")
    print("="*60)
    print(f"\n📊 Results:")
    print(f"  ✅ Processed: {downloader.stats['processed_usernames']} accounts")
    print(f"  📷 Downloaded: {downloader.stats['total_downloads']} images")
    print(f"  🎥 Skipped videos: {downloader.stats['skipped_videos']}")
    print(f"  🏊 Swimwear detected: {downloader.stats['total_swimwear_detected']}")
    print(f"  📁 Location: {downloader.download_folder.absolute()}")
    print("="*60)

if __name__ == '__main__':
    try:
        asyncio.run(quick_test())
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
