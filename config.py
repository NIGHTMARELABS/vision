OPENAI_API_KEY = 'your-openai-api-key-here'
GEMINI_API_KEY = 'AIzaSyCMMMzfmPaD9USFgnu0FPrxXQnBdToI4FU'

AI_MODEL = 'gemini'

OPENAI_MODEL = 'gpt-5-nano'

GEMINI_MODEL = 'gemini-2.0-flash-exp'
# GEMINI_MODEL = 'gemini-1.5-flash-8b'
# GEMINI_MODEL = 'gemini-1.5-flash'
# GEMINI_MODEL = 'gemini-1.5-pro'

SPREADSHEET_ID = '1WCmFLdioC-mm_QtWdA3qNo_sEdEHZ3HTyCdqCaz_2MA'
SHEET_NAME = 'Daily Results'
COLUMN_NAME = 'Tag Account'
COUNT_COLUMN_NAME = 'count'

DOWNLOAD_FOLDER = 'instagram_downloads'
MAX_POSTS_PER_ACCOUNT = 12

BASE_URL = 'https://fastdl.app/en2'

WAIT_AFTER_GOTO = (2, 3)
WAIT_AFTER_INPUT = (1.5, 2)
WAIT_AFTER_CLICK = (4, 5)
WAIT_AFTER_RESULTS = (2, 3)
WAIT_BETWEEN_DOWNLOADS = (1.5, 2.5)
WAIT_BETWEEN_ACCOUNTS = (3, 5)

SCROLL_PIXELS = 800
SCROLL_WAIT = (2.5, 3.5)
MAX_NO_CHANGE_ATTEMPTS = 4

HEADLESS = False
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080

PAGE_LOAD_TIMEOUT = 30000
SELECTOR_TIMEOUT = 10000
RESULT_TIMEOUT = 15000
DOWNLOAD_TIMEOUT = 30000

LOG_LEVEL = 'INFO'
LOG_FILE_PREFIX = 'instagram_downloader'

ADDITIONAL_AD_PATTERNS = [
    '**/googleads.g.doubleclick.net/**',
    '**/adsbygoogle.js',
    '**/*adsbygoogle*',
]

AD_MODAL_SELECTORS = [
    '.ad-modal__wrapper',
    '.ad-modal__content',
    '.ad-modal',
    '.adsbygoogle',
    'ins.adsbygoogle',
    '[data-ad-client]',
    'iframe[src*="doubleclick"]',
    'iframe[src*="googleads"]',
    'iframe[src*="adservice"]'
]

MAX_EMPTY_ROWS_STOP = 10

MAX_RETRIES = 3
RETRY_DELAY = (5, 10)

SELECTORS = {
    'input': 'input#search-form-input',
    'download_button': 'button.search-form__button[type="submit"]',
    'search_result': '.search-result',
    'username_display': '.user-info__username',
    'post_item': '.profile-media-list__item',
    'video_tag': '.tags__item--video',
    'download_link': 'a.button__download'
}

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

PROXY_ENABLED = False

PROXY_LIST = []

PROXY_ROTATION_MODE = 'round-robin'
PROXY_MAX_RETRIES = 2
