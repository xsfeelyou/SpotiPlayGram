import os

APP_NAME = "SpotiPlayGram"

LOG_FORMAT_SPOTIFY = " \033[92mSpotify\033[0m | "
LOG_FORMAT_TELEGRAM = "\033[94mTelegram\033[0m | "
LOG_FORMAT_GENIUS = "  \033[93mGenius\033[0m | "
LOG_FORMAT_JSON = "    \033[95mJSON\033[0m | "
LOG_FORMAT_MISTRAL = " \033[38;2;255;165;0mMistral\033[0m | "

ERROR_UPDATING_MESSAGES = f"{LOG_FORMAT_TELEGRAM}Failed to update messages: {{0}}"
ERROR_UNEXPECTED = "Unexpected error: {}"

ERROR_AUTH_TG = f"{LOG_FORMAT_TELEGRAM}Authentication failed"
ERROR_AUTH_SPOTIFY = f"{LOG_FORMAT_SPOTIFY}Authentication failed: {{0}}"
ERROR_AUTH_GENIUS = f"{LOG_FORMAT_GENIUS}Authentication failed"
ERROR_TG_AUTH_INVALID = f"{LOG_FORMAT_TELEGRAM}Invalid Telegram credentials or access denied"
ERROR_SPOTIFY_AUTH_INVALID = f"{LOG_FORMAT_SPOTIFY}Invalid Spotify credentials or access denied"
ERROR_GENIUS_ACCESS_TOKEN_INVALID = f"{LOG_FORMAT_GENIUS}Invalid GENIUS_ACCESS_TOKEN or access denied"
ERROR_AUTH_MISTRAL_API_KEY_MISSING = f"{LOG_FORMAT_MISTRAL}API key is missing. Set MISTRAL_API_KEY in .env"
ERROR_AUTH_MISTRAL = f"{LOG_FORMAT_MISTRAL}Authentication failed"

ERROR_AUTH_TG_API_ID_MISSING = f"{LOG_FORMAT_TELEGRAM}Environment variable TG_API_ID is not set in .env"
ERROR_AUTH_TG_API_HASH_MISSING = f"{LOG_FORMAT_TELEGRAM}Environment variable TG_API_HASH is not set in .env"
ERROR_AUTH_TG_BOT_TOKEN_MISSING = f"{LOG_FORMAT_TELEGRAM}Environment variable TG_BOT_TOKEN is not set in .env"
ERROR_AUTH_TG_CHANNEL_ID_MISSING = f"{LOG_FORMAT_TELEGRAM}Environment variable TG_CHANNEL_ID is not set in .env"
ERROR_AUTH_TG_MEDIA_MESSAGE_ID_MISSING = f"{LOG_FORMAT_TELEGRAM}Environment variable TG_MEDIA_MESSAGE_ID is not set in .env"
ERROR_AUTH_TG_TEXT_MESSAGE_ID_MISSING = f"{LOG_FORMAT_TELEGRAM}Environment variable TG_TEXT_MESSAGE_ID is not set in .env"
ERROR_AUTH_TG_ADMIN_USER_ID_MISSING = f"{LOG_FORMAT_TELEGRAM}Environment variable TG_ADMIN_USER_ID is not set in .env"

ERROR_AUTH_SPOTIFY_CLIENT_ID_MISSING = f"{LOG_FORMAT_SPOTIFY}Environment variable SPOTIFY_CLIENT_ID is not set in .env"
ERROR_AUTH_SPOTIFY_CLIENT_SECRET_MISSING = f"{LOG_FORMAT_SPOTIFY}Environment variable SPOTIFY_CLIENT_SECRET is not set in .env"
ERROR_AUTH_SPOTIFY_REDIRECT_URI_MISSING = f"{LOG_FORMAT_SPOTIFY}Environment variable SPOTIFY_REDIRECT_URI is not set in .env"

ERROR_ENV_INVALID_INT = "Invalid value for {0} in .env: expected an integer (got: {1})"
ERROR_ENV_INVALID_FLOAT = "Invalid value for {0} in .env: expected a number (got: {1})"

INFO_NO_TRACK_PLAYING = f"{LOG_FORMAT_SPOTIFY}Nothing is currently playing"
INFO_TRACK_PLAYING = f"{LOG_FORMAT_SPOTIFY}Now playing: {{0}} – {{1}}"

INFO_COVER_DOWNLOADED_PRIMARY = f"{LOG_FORMAT_TELEGRAM}Cover art downloaded successfully (primary method)"
INFO_COVER_DOWNLOADED_FALLBACK1 = f"{LOG_FORMAT_TELEGRAM}Cover art downloaded successfully (fallback method #1)"
INFO_COVER_DOWNLOADED_FALLBACK2 = f"{LOG_FORMAT_TELEGRAM}Cover art downloaded successfully (fallback method #2)"
INFO_MEDIA_UPDATED = f"{LOG_FORMAT_TELEGRAM}Media message updated successfully"
INFO_MEDIA_DEFAULT_UPDATED = f"{LOG_FORMAT_TELEGRAM}Media message reset to default"
INFO_CHANNEL_DEFAULT_UPDATED = f"{LOG_FORMAT_TELEGRAM}Channel title and photo reset to default"
INFO_CHANNEL_UPDATED = f"{LOG_FORMAT_TELEGRAM}Channel title and photo updated successfully"

INFO_AUTH_TG_USER = f"{LOG_FORMAT_TELEGRAM}Authorizing user..."
INFO_AUTH_TG_USER_SUCCESS = f"{LOG_FORMAT_TELEGRAM}User authorized successfully"
INFO_AUTH_TG_BOT = f"{LOG_FORMAT_TELEGRAM}Authorizing bot..."
INFO_AUTH_TG_BOT_SUCCESS = f"{LOG_FORMAT_TELEGRAM}Bot authorized successfully"
INFO_AUTH_SPOTIFY = f"{LOG_FORMAT_SPOTIFY}Authenticating..."
INFO_AUTH_SPOTIFY_SUCCESS = f"{LOG_FORMAT_SPOTIFY}Authentication successful"
INFO_AUTH_GENIUS = f"{LOG_FORMAT_GENIUS}Authenticating..."
INFO_AUTH_GENIUS_SUCCESS = f"{LOG_FORMAT_GENIUS}Authentication successful"
INFO_AUTH_MISTRAL = f"{LOG_FORMAT_MISTRAL}Authenticating..."
INFO_AUTH_MISTRAL_SUCCESS = f"{LOG_FORMAT_MISTRAL}Authentication successful"
INFO_AUTH_RETRY = "{0}Retrying authentication in {1:.1f}s (attempt {2}/{3})"

INFO_STARTUP_COMPLETE = "Initialization completed"
INFO_ENTERING_MAIN_LOOP = "Starting the main status update loop..."
INFO_APP_SHUTDOWN = "Application has shut down"

INFO_DEP_UPDATES_FOUND = "Dependency updates found: {0}"
INFO_DEP_UPDATES_DONE = "Dependencies updated, restarting application..."
INFO_DEP_UPDATES_ENABLED = "Dependency auto-updates are enabled"
INFO_DEP_UPDATES_DISABLED = "Dependency auto-updates are disabled"
ERROR_DEP_UPDATES = "Dependency update error: {0}"

INFO_AUTOSTART_ENABLED = "Autostart is enabled ({0})"
WARNING_AUTOSTART_DISABLED = "Autostart is disabled"
WARNING_AUTOSTART_ALREADY_DISABLED = "Autostart is already disabled"
INFO_AUTOSTART_INSTALLED = "Autostart entry installed: {0}"
INFO_AUTOSTART_REMOVED = "Autostart entry removed: {0}"
INFO_AUTOSTART_ALREADY_OK = "Autostart state is already up to date"
INFO_AUTOSTART_XDG_DETAIL = "XDG autostart (GUI session)"
INFO_AUTOSTART_CRON_DETAIL = "cron @reboot (headless)"
INFO_AUTOSTART_WINDOWS_SHORTCUT = "Windows Startup shortcut (SpotiPlayGram.exe, background)"
INFO_AUTOSTART_WINDOWS_REGISTRY = "Windows registry Run (background)"
WARNING_AUTOSTART_WINDOWS_FALLBACK = "Cannot create the Startup shortcut; using the registry Run key: {0}"
INFO_AUTOSTART_CRON_ENTRY = "crontab @reboot"
INFO_AUTOSTART_LINGER_ENABLED = "Autostart linger enabled (starts at boot without login)"
INFO_AUTOSTART_ENABLE_DONE = "Autostart enabled"
INFO_AUTOSTART_DISABLE_DONE = "Autostart disabled"
ERROR_AUTOSTART_ENABLE = "Failed to enable autostart"
ERROR_AUTOSTART_DISABLE = "Failed to disable autostart"
PROMPT_PRESS_ENTER = "Press Enter to exit and close the console..."
ENV_AUTOSTART_SUPERVISED = "SPG_SUPERVISED"
ERROR_AUTOSTART = "Autostart sync failed: {0}"
ERROR_AUTOSTART_LINGER = "Autostart linger not enabled (starts after login only): {0}"
ERROR_AUTOSTART_CRONTAB_READ = "Cannot read crontab, existing entries are left intact: {0}"
ERROR_AUTOSTART_CRONTAB_WRITE = "Cannot update crontab: {0}"

INFO_SINGLE_INSTANCE = "Another application instance is already running"
INFO_SINGLE_INSTANCE_REMOTE = 'Another application instance is already running on device "{0}"'
ENV_LOCAL_LOCK_HELD = "SPG_LOCAL_LOCK_HELD"
ENV_NO_PAUSE = "SPG_NO_PAUSE"
WARNING_DEVICE_LOCK_UNAVAILABLE = "Cross-device lock is unavailable: {0}"
WARNING_DEVICE_LOCK_LOST = "Another device took over the cross-device lock; shutting down"
ERROR_DEVICE_LOCK = "Cross-device lock error: {0}"
INFO_START_RESTART = "Application exited with code {0}, restarting in {1} seconds (attempt {2}/{3})"
ERROR_START_RESTART_LIMIT = "Application keeps failing, giving up after {0} restart attempts"

UPDATE_CHECK_TIME_UTC = "00:00"
STATUS_IDLE_POLL_SECONDS = 60

INFO_MESSAGES_DELETED = f"{LOG_FORMAT_TELEGRAM}Deleted {{0}} messages"
INFO_FALLBACK_TRACK_INFO = f"{LOG_FORMAT_SPOTIFY}Using cached track data to build the message"

NOW_PLAYING_ON_SPOTIFY = "Now Playing on Spotify"
LYRICS_ON_GENIUS = "Lyrics on Genius"
ALBUM_LABEL_SINGLE = "Single: "
ALBUM_LABEL_ALBUM = "Album: "
RELEASE_DATE_PREFIX = "Release Date: "

INFO_STATUS_ENABLED = "✅ Playback status is enabled"
INFO_STATUS_DISABLED = "🚫 Playback status is disabled"
INFO_STATUS_ENABLING = "⏳ Enabling playback status..."
INFO_STATUS_DISABLING = "⏳ Disabling playback status..."
INFO_BUTTON_ENABLE_PRESSED = "✅ Enable playback status button pressed..."
INFO_BUTTON_DISABLE_PRESSED = "🚫 Disable playback status button pressed..."
BTN_ENABLE = "▶️ Enable"
BTN_DISABLE = "⏹️ Disable"
ACCESS_DENIED_MESSAGE = "Access denied"

INFO_MAIN_MENU = "\u2063🏠\u2063"
BTN_MENU_STATUS = "📢 Playback status"
BTN_BACK = "\u2063↩️\u2063"
BTN_MENU_GENIUS_SETTINGS = "⚙️ Genius settings"
BTN_MENU_MISTRAL = "🤖 Mistral AI models"
BTN_REFRESH_MISTRAL = "\u2063🔄\u2063"

BTN_MENU_TAG_PATTERNS = "Version patterns"
BTN_MENU_FEAT_PATTERNS = "Feat patterns"
BTN_PATTERNS_ADD = "Add"
BTN_PATTERNS_DELETE = "Delete"
BTN_PATTERNS_CONFIRM_ADD = "Confirm add"
BTN_PATTERNS_CONFIRM_DELETE = "Confirm delete"

VALUE_NOT_SET = "not set"
PATTERNS_KIND_TAG_LABEL = "Version patterns"
PATTERNS_KIND_FEAT_LABEL = "Feat patterns"
PATTERNS_KIND_GENERIC_LABEL = "Patterns"

INFO_GENIUS_SETTINGS_MENU = "Select a setting.\n\n<b>🤖 Mistral AI models</b> — choose Mistral models used to select the correct Genius lyrics URL.\n<b>Version patterns</b> — list of words/phrases used to detect versions/remixes in track titles.\n<b>Feat patterns</b> — list of words/phrases used to detect feat/with/ft etc."
INFO_GENIUS_SETTINGS_MENU_NO_MISTRAL = "Select a setting.\n\n<b>Version patterns</b> — list of words/phrases used to detect versions/remixes in track titles.\n<b>Feat patterns</b> — list of words/phrases used to detect feat/with/ft etc."
INFO_PATTERNS_MENU_TAG = "<b>Version patterns.</b>\n\nHere you can add or delete patterns used to detect versions/remixes."
INFO_PATTERNS_MENU_FEAT = "<b>Feat patterns.</b>\n\nHere you can add or delete patterns used to detect featuring/with/ft etc."
INFO_PATTERNS_APPLIED = f"{LOG_FORMAT_JSON}Applied: {{0}}"

INFO_PATTERNS_ENTER_ADD_TAG = "Enter a version pattern to add.\n\nYou can enter multiple patterns separated by two dots \"..\"\n\nExample:\n<blockquote><code>remix</code></blockquote>\n<blockquote><code>remix .. warm up .. slowed & reverb</code></blockquote>"
INFO_PATTERNS_ENTER_DELETE_TAG = "Enter a version pattern to delete.\n\nYou can enter multiple patterns separated by two dots \"..\"\n\nExample:\n<blockquote><code>remix</code></blockquote>\n<blockquote><code>remix .. warm up .. slowed & reverb</code></blockquote>"
INFO_PATTERNS_CONFIRM_ADD_TAG = "Review the version patterns and confirm adding:\n\n{0}"
INFO_PATTERNS_CONFIRM_DELETE_TAG = "Review the version patterns and confirm deleting:\n\n{0}"
INFO_PATTERNS_SAVED_WILL_APPLY_TAG = "✅ Version patterns: changes saved. The updated pattern list will be applied on the next track update."

INFO_PATTERNS_ENTER_ADD_FEAT = "Enter a feat pattern to add.\n\nYou can enter multiple patterns separated by two dots \"..\"\n\nExample:\n<blockquote><code>ft</code></blockquote>\n<blockquote><code>ft .. w/ .. feat.</code></blockquote>"
INFO_PATTERNS_ENTER_DELETE_FEAT = "Enter a feat pattern to delete.\n\nYou can enter multiple patterns separated by two dots \"..\"\n\nExample:\n<blockquote><code>ft</code></blockquote>\n<blockquote><code>ft .. w/ .. feat.</code></blockquote>"
INFO_PATTERNS_CONFIRM_ADD_FEAT = "Review the feat patterns and confirm adding:\n\n{0}"
INFO_PATTERNS_CONFIRM_DELETE_FEAT = "Review the feat patterns and confirm deleting:\n\n{0}"
INFO_PATTERNS_SAVED_WILL_APPLY_FEAT = "✅ Feat patterns: changes saved. The updated pattern list will be applied on the next track update."

ERROR_PATTERNS_EMPTY = "⚠️ Empty input. Please enter a pattern."
ERROR_PATTERNS_DUPLICATES = "⚠️ Your input contains duplicate patterns. Please re-enter the batch without duplicates."
ERROR_PATTERNS_ALREADY_EXIST = "⚠️ One or more patterns already exist. Please enter a new batch of patterns."
ERROR_PATTERNS_NOT_FOUND = "⚠️ One or more patterns were not found in the list. Please enter an existing batch of patterns."
ERROR_PATTERNS_EXISTS_IN_TAG = "⚠️ Patterns:\n\n{0}\n\nAlready exist in version patterns. Remove them there or enter a different pattern."
ERROR_PATTERNS_EXISTS_IN_FEAT = "⚠️ Patterns:\n\n{0}\n\nAlready exist in feat patterns. Remove them there or enter a different pattern."

ERROR_CHANNEL_INFO_UPDATE = f"{LOG_FORMAT_TELEGRAM}Failed to update channel info: {{0}}"
ERROR_COVER_UPLOAD = f"{LOG_FORMAT_TELEGRAM}Failed to upload cover art: {{0}}"
ERROR_MEDIA_UPDATE = f"{LOG_FORMAT_TELEGRAM}Failed to update media message: {{0}}"
ERROR_TEXT_UPDATE = f"{LOG_FORMAT_TELEGRAM}Failed to update text message: {{0}}"
ERROR_CHANNEL_TITLE_UPDATE = f"{LOG_FORMAT_TELEGRAM}Failed to update channel title: {{0}}"
ERROR_CHANNEL_PHOTO_UPDATE = f"{LOG_FORMAT_TELEGRAM}Failed to update channel photo: {{0}}"
ERROR_MESSAGE_DELETE = f"{LOG_FORMAT_TELEGRAM}Failed to delete messages: {{0}}"
ERROR_UPDATE_LOOP = f"{LOG_FORMAT_TELEGRAM}Update loop error: {{0}}"
ERROR_TG_DISCONNECTED_RETRY = f"{LOG_FORMAT_TELEGRAM}Disconnected; retrying in {{0}}s"
INFO_TG_RECONNECTED = f"{LOG_FORMAT_TELEGRAM}Reconnected successfully"
ERROR_FALLBACK_RESET = f"{LOG_FORMAT_TELEGRAM}Resetting to default state due to error: {{0}}"

REASON_TELEGRAM_UPDATE_FAILED = "Telegram update failed"
REASON_EMPTY_TELEGRAM_RESPONSE = "empty response from Telegram API"
REASON_EMPTY_JSON_RESPONSE = "empty response body"

INFO_GENIUS_DISABLED = f"{LOG_FORMAT_GENIUS}Genius is disabled"
ERROR_AUTH_GENIUS_INVALID_TOKEN = f"{LOG_FORMAT_GENIUS}Token is missing. Set GENIUS_ACCESS_TOKEN in .env"

INFO_GENIUS_SEARCH_QUERY = f"{LOG_FORMAT_GENIUS}GET /search ({{0}}): {{1}}"
INFO_TRANSLATION_REPLACED = f"{LOG_FORMAT_GENIUS}Translation replaced with original: {{0}} → {{1}}"
INFO_GENIUS_NO_RESULTS = f"{LOG_FORMAT_GENIUS}No suitable songs found"
INFO_GENIUS_SCORING_CHOSEN = f"{LOG_FORMAT_GENIUS}Chosen URL: {{0}}"
ERROR_GENIUS_HTTP = f"{LOG_FORMAT_GENIUS}Genius HTTP error: {{0}} — {{1}}"
ERROR_GENIUS_TIMEOUT = f"{LOG_FORMAT_GENIUS}Request timeout: {{0}} sec"

INFO_GENIUS_CANDIDATE_SUMMARY = f"{LOG_FORMAT_GENIUS}  #{{0:02d}} {{1}} | album: {{2}} | date: {{3}} | score: {{4}} | url: {{5}}"
INFO_GENIUS_CANDIDATE_REASON = f"{LOG_FORMAT_GENIUS}       {{0}}"
INFO_GENIUS_DETAILS_FETCHED = f"{LOG_FORMAT_GENIUS}Fetched details for {{0}} out of {{1}} candidates"
ERROR_GENIUS_DETAILS_FAILED = f"{LOG_FORMAT_GENIUS}Failed to fetch details for song ID={{0}}"
INFO_GENIUS_TAG_FOUND_FROM_TRACK = f"{LOG_FORMAT_GENIUS}Tag found: track_base='{{0}}' | tag='{{1}}' | tag_pattern='{{2}}'"
INFO_GENIUS_FEAT_REMOVED_FROM_TRACK = f"{LOG_FORMAT_GENIUS}Feat removed: track_base='{{0}}' | feat='{{1}}' | feat_pattern='{{2}}'"
INFO_GENIUS_TAG_FOUND_AND_FEAT_REMOVED_FROM_TRACK = f"{LOG_FORMAT_GENIUS}Tag found and feat removed: track_base='{{0}}' | tag='{{1}}' | tag_pattern='{{2}}' | feat='{{3}}' | feat_pattern='{{4}}'"

INFO_MISTRAL_MENU_TITLE = "<b><u>Mistral AI model management</u></b> (for selecting the correct Genius lyrics URL)"
INFO_MISTRAL_CURRENT_MODEL = "<i>Current model:</i> <code>{0}</code>"
INFO_MISTRAL_FALLBACK_MODELS = "<i>Fallback models:</i> {0}"
INFO_MISTRAL_MODELS_NON_REASONING = "<i>Available Non-Reasoning models:</i>"
INFO_MISTRAL_MODELS_REASONING = "<i>Available Reasoning models:</i>"
INFO_MISTRAL_LOADING = "⏳ Fetching and validating model list…"
INFO_MISTRAL_SAVING = "⏳ Saving and validating selected models…"
INFO_MISTRAL_ENTER_MODEL = "Enter model IDs separated by space or comma in priority order (copy from the list above).\nFor reasoning models you can add reasoning_effort as <code>model=effort</code>.\nIf <code>model=effort</code> is not specified, the model's default effort is used.\nExample: <code>codestral-latest, mistral-small-latest=high</code>"
INFO_MISTRAL_MODEL_SAVED = "✅ Models updated to:\n{0}\n\nThe change will take effect on the next Spotify status update."
INFO_MISTRAL_MODEL_UPDATED = f"{LOG_FORMAT_MISTRAL}Model updated: {{0}}"
ERROR_MISTRAL_FETCH_MODELS = "⚠️ Failed to fetch model list: {0}"
ERROR_MISTRAL_MODEL_SAVE = "⚠️ Failed to save the selected models. Please try again."
ERROR_MISTRAL_MODEL_INVALID = "⚠️ Invalid model. Please choose one from the list above"
ERROR_MISTRAL_MODEL_FORMAT = "⚠️ Invalid input format. Enter model IDs separated by space or comma.\nFor reasoning models use <code>model=effort</code>.\nIf <code>model=effort</code> is not specified, the model's default effort is used.\nExample: <code>codestral-latest, mistral-small-latest=high</code>"

INFO_MISTRAL_SWITCH_MODEL = f"{LOG_FORMAT_MISTRAL}Trying model: {{0}}"
ERROR_MISTRAL_SDK_MISSING = f"{LOG_FORMAT_MISTRAL}The 'mistralai' library is not installed"
ERROR_MISTRAL_TIMEOUT = f"{LOG_FORMAT_MISTRAL}Request timeout: model {{0}}, {{1}} sec"
INFO_MISTRAL_DISABLED = f"{LOG_FORMAT_MISTRAL}Disabled"
ERROR_MISTRAL_API_KEY_INVALID = f"{LOG_FORMAT_MISTRAL}Invalid MISTRAL_API_KEY or access denied"
ERROR_MISTRAL_MODEL_NOT_FOUND = f"{LOG_FORMAT_MISTRAL}Model is unavailable or not found: {{0}}"
ERROR_MISTRAL_REQUEST = f"{LOG_FORMAT_MISTRAL}Request error: {{0}}"
MISTRAL_TIMEOUT_SECONDS = 30

GENIUS_TIMEOUT_SECONDS = 30
GENIUS_SEARCH_CONCURRENCY = 128
SEARCH_VARIANTS_LIMIT = 128
GENIUS_SEARCH_PER_PAGE = 10
GENIUS_DETAILS_TOP_N = 10
GENIUS_FUZZY_TOKENS_PERCENT = 0.51
GENIUS_FUZZY_CHARS_PERCENT = 0.49
GENIUS_TAG_FUZZY_TOKENS_PERCENT = 0.9
GENIUS_TAG_FUZZY_CHARS_PERCENT = 0.9

INFO_MISTRAL_FINAL_SELECTION_START = f"{LOG_FORMAT_MISTRAL}Selecting the correct lyrics URL... (model {{0}})"
INFO_MISTRAL_FINAL_SELECTION_NONE = f"{LOG_FORMAT_MISTRAL}No candidate selected; assuming no suitable songs"
INFO_MISTRAL_FINAL_SELECTION_NONE_CACHED = f"{LOG_FORMAT_MISTRAL}Using cached 'no candidate selected' result; assuming no suitable songs"
INFO_MISTRAL_FINAL_SELECTION_NONE_INFLIGHT = f"{LOG_FORMAT_MISTRAL}Using shared in-flight 'no candidate selected' result; assuming no suitable songs"
ERROR_MISTRAL_FINAL_SELECTION = f"{LOG_FORMAT_MISTRAL}Final selection error: {{0}}"

GENIUS_SCORE_TRACK_EXACT_TITLE_BONUS = 25
GENIUS_SCORE_TAG_PHRASE_IN_TITLE = 25
GENIUS_SCORE_TAG_PHRASE_IN_TITLE_FEAT = 25
GENIUS_SCORE_TAG_TOKENS_PER = 25
GENIUS_SCORE_TAG_TOKENS_MAX = 100
GENIUS_PENALTY_TAG_MISSING = 10
GENIUS_SCORE_RELATIONSHIP_TAG_TOKENS = 5
GENIUS_SCORE_FEATURED_MATCH_PER = 15
GENIUS_SCORE_FEATURED_MATCH_MAX = 100
GENIUS_SCORE_FEATURED_IN_TITLE_PER = 10
GENIUS_SCORE_FEATURED_IN_TITLE_MAX = 100
GENIUS_PENALTY_TITLE_EXTRA_TOKEN_PER = 5
GENIUS_PENALTY_TITLE_MISSING_TOKEN_PER = 5
GENIUS_PENALTY_FEATURED_EXTRA_TOKEN_PER = 5
GENIUS_PENALTY_FEATURED_MISSING_TOKEN_PER = 5
GENIUS_PENALTY_TOKEN_CAP = 100
GENIUS_PENALTY_TAG_MISSING_TOKEN_PER = 5
GENIUS_TOP1_BONUS_PER = 1
GENIUS_TOP1_BONUS_MAX = 100
GENIUS_SCORE_ALBUM_MATCH = 5
GENIUS_PENALTY_ALBUM_MISMATCH = 5
GENIUS_ALBUM_MIN_TRACKS = 5
GENIUS_SCORE_CONSEC_TOKENS_PER = 5
GENIUS_SCORE_CONSEC_TOKENS_MAX = 100
GENIUS_SCORE_SLUG_TOKEN_PER = 5
GENIUS_SLUG_TOKEN_MIN_LEN = 2

GENIUS_HTTP_MAX_RETRIES = 5
GENIUS_HTTP_RETRY_DELAY_BASE = 0.5

DIRS = {"LOGS": "logs", "SESSION": "session"}
DEFAULT_LOGO_PATH = os.path.join("spotify", "default.png")
DEFAULT_MESSAGE = "\u2063"

ENV_LOG_DIR = "SPG_LOG_DIR"

ERROR_ENV_INVALID_BOOL = "Invalid value for {0} in .env: expected a boolean"
ERROR_ENV_OUT_OF_RANGE = "Invalid value for {0} in .env: expected a number between {1} and {2}"
ERROR_SINGLE_INSTANCE_LOCK = f"{LOG_FORMAT_TELEGRAM}Cannot acquire single-instance lock: {{0}}"
ERROR_SESSION_DIR = "Failed to create session directory: {0}"
ERROR_LOG_SETUP = "Failed to initialize file logging: {0}"
ERROR_SPOTIFY_TOKEN_REFRESH = f"{LOG_FORMAT_SPOTIFY}Failed to refresh access token: {{0}}"
INFO_MISTRAL_FALLBACK = f"{LOG_FORMAT_MISTRAL}Mistral selection unavailable; using the best Genius candidate"
ERROR_PATTERNS_SAVE = "⚠️ Failed to save patterns. Please try again."
ERROR_PATTERNS_CORRUPT = f"{LOG_FORMAT_JSON}Patterns file is corrupted and was backed up: {{0}}"
ERROR_STATUS_DISABLE_FAILED = "⚠️ Failed to reset the channel. Playback status is still enabled."
WARNING_CHDIR_FAILED = "Failed to change working directory: {0}"
WARNING_FLOOD_WAIT = f"{LOG_FORMAT_TELEGRAM}Flood wait {{0}}s; limiting retry"
ERROR_FILE_REFERENCE_EXPIRED = f"{LOG_FORMAT_TELEGRAM}File reference expired"
