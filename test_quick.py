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
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
        )
        
        await context.add_init_script("""
            // ULTRA STEALTH MODE
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };

            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );

            Object.defineProperty(navigator, 'mediaDevices', {
                get: () => ({
                    enumerateDevices: () => Promise.resolve([]),
                    getUserMedia: () => Promise.reject(new Error('Not allowed'))
                })
            });

            Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
            Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
            Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
            Object.defineProperty(navigator, 'getBattery', {
                get: () => () => Promise.resolve({
                    charging: true,
                    chargingTime: 0,
                    dischargingTime: Infinity,
                    level: 1
                })
            });

            Object.defineProperty(screen, 'availWidth', { get: () => 1920 });
            Object.defineProperty(screen, 'availHeight', { get: () => 1040 });
            Object.defineProperty(screen, 'width', { get: () => 1920 });
            Object.defineProperty(screen, 'height', { get: () => 1080 });
            Object.defineProperty(screen, 'colorDepth', { get: () => 24 });
            Object.defineProperty(screen, 'pixelDepth', { get: () => 24 });

            window.outerWidth = 1920;
            window.outerHeight = 1080;

            // Ad blocking
            window.adsbygoogle = [];
            Object.defineProperty(window, 'adsbygoogle', {
                configurable: false,
                get: function() { return []; },
                set: function() {}
            });

            window.googletag = window.googletag || {};
            window.googletag.cmd = window.googletag.cmd || [];
            window.googletag.cmd.push = function() { return 1; };

            window.ga = function() {};
            window.gtag = function() {};

            const cleanAds = () => {
                try {
                    document.querySelectorAll('iframe[src*="doubleclick"], iframe[src*="google"], iframe[src*="ads"]').forEach(el => {
                        try { el.remove(); } catch(e) {}
                    });
                    document.querySelectorAll('ins.adsbygoogle, [data-ad-client], .adsbygoogle').forEach(el => {
                        try { el.remove(); } catch(e) {}
                    });
                } catch(e) {}
            };

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', cleanAds);
            } else {
                cleanAds();
            }
            setInterval(cleanAds, 1000);
        """)

        # Block ad network requests
        await context.route('**/*doubleclick*/**', lambda route: route.abort())
        await context.route('**/*googleads*/**', lambda route: route.abort())
        await context.route('**/*google-analytics*/**', lambda route: route.abort())
        await context.route('**/*googletagmanager*/**', lambda route: route.abort())
        await context.route('**/*fundingchoices*/**', lambda route: route.abort())
        await context.route('**/pagead/**', lambda route: route.abort())

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
