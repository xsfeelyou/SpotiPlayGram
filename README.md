# SpotiPlayGram

[![README на русском](https://img.shields.io/badge/README-RU-blue)](README_RU.md)

**SpotiPlayGram** is a Python application that displays your current Spotify track in a Telegram channel in real time. It automatically updates the channel with album artwork, track information, and optionally provides a link to lyrics on Genius.

---

## 💡 Idea

The main idea is to **add the channel you created to your Telegram profile** (Settings → Edit Profile → Channel). When playing music on Spotify, the channel title automatically changes to **"Artist – Track Name"**, and the message displays **"Now Playing on Spotify"**.

This creates a **full-fledged Spotify playback status** right in your Telegram profile — other users can see what you're currently listening to, just like a classic Spotify status.

---

## 📸 Screenshots

![SpotiPlayGram](https://i.ibb.co/8Dk3rT1j/1.png)
![SpotiPlayGram](https://i.ibb.co/0Vf5653h/2.png) ![SpotiPlayGram](https://i.ibb.co/Y7RCckGx/3.png)
![SpotiPlayGram](https://i.ibb.co/s92xtydB/4.png) ![SpotiPlayGram](https://i.ibb.co/zWQ3G7z1/5.png)
![SpotiPlayGram](https://i.ibb.co/ycHJMrvK/6.png) ![SpotiPlayGram](https://i.ibb.co/ksCk1ymd/7.png)

---

## ✨ Features

### 🎵 Spotify Integration
- Real-time tracking of currently playing track
- Displays artist name, track title, album name, and release date
- Automatic album cover download and display

### 📢 Telegram Channel Updates
- Updates channel messages with track info and album artwork
- Dynamically changes channel title to current track
- Updates channel profile photo with album cover
- Auto-deletes unnecessary messages to keep the channel clean

### 📝 Genius Lyrics Integration
- Automatically searches for lyrics URL on Genius
- Advanced fuzzy matching algorithm for accurate results

### 🤖 Mistral AI Selection
- Uses Mistral AI to intelligently select the best lyrics URL
- Supports multiple AI models with automatic fallback
- Configurable model priority chain

### 🎛️ Bot Control Panel
- Telegram bot for administration
- Enable/disable playback status
- Manage Genius settings (version patterns, feat patterns)
- Configure Mistral AI models directly from Telegram

### ⚙️ Additional Features
- Cross-platform support (Windows, Linux, macOS)
- Automatic virtual environment creation
- Optional daily dependency update checks (at 00:00 UTC)

---

## 📋 Requirements

- Python 3.10 or higher
- Spotify Developer Account
- Spotify Premium (may be required for the playback status API)
- Telegram API credentials (API ID, API Hash)
- Telegram Bot Token
- Telegram user account (phone number login, plus cloud password if set)
- (Optional) Genius API Token
- (Optional) Mistral AI API Key

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/xsfeelyou/SpotiPlayGram.git
cd SpotiPlayGram
```

### 2. Configure environment variables

Copy the `.env-example` file to `.env` and fill in your credentials:

```bash
# Windows
copy .env-example .env

# Linux/macOS
cp .env-example .env
```

```env
# Spotify API
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
SPOTIFY_LANGUAGE=EN  # Language for Spotify API requests
SPOTIFY_REQUEST_TIMEOUT=5  # Spotify API request timeout in seconds

# Telegram API
TG_API_ID=your_telegram_api_id
TG_API_HASH=your_telegram_api_hash
TG_BOT_TOKEN=your_bot_token

# Telegram IDs
TG_CHANNEL_ID=1234567890
TG_ADMIN_USER_ID=your_user_id

# Telegram Channel Message IDs
TG_MEDIA_MESSAGE_ID=2
TG_TEXT_MESSAGE_ID=3
TG_AUTO_DELETE_MESSAGES=True  # Automatically delete bot service messages

# Genius API (optional)
GENIUS_ACCESS_TOKEN=your_genius_token
ENABLE_GENIUS=True  # Enable lyrics search through Genius API
GENIUS_DETAILED_LOG=False  # Detailed logging of Genius operations

# Mistral AI (optional)
MISTRAL_API_KEY=your_mistral_api_key
ENABLE_MISTRAL=True  # Enable Mistral AI for accurate lyrics URL selection
MISTRAL_MODEL=mistral-large-latest  # Mistral AI model (default)

# Auth retries
AUTH_RETRY_MAX_RETRIES=2  # Maximum number of authorization attempts
AUTH_RETRY_DELAY_BASE=1.0  # Base delay between authorization attempts in seconds

# Dependency auto-updates
ENABLE_DEP_UPDATES=False  # Automatic dependency updates on startup

# Update interval in seconds
UPDATE_INTERVAL=1  # Spotify API polling interval in seconds
```

---

## 🔧 Configuration

### Spotify Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application
3. In the application settings, add the Redirect URI: `http://localhost:8888/callback`
4. Copy the Client ID and Client Secret to `.env`

**Authorization Note:** Spotify uses **OAuth 2.0 authentication** — on the first launch, a browser will open to log in to your account. **Authorization via SSH is not possible**, as there is no access to a graphical interface. You must perform the initial authorization on a computer with a graphical interface (Windows, Linux with GUI, or macOS), and then transfer the session files to the target device.

**Language Note:** Choose the correct value for `SPOTIFY_LANGUAGE` in `.env` — it should match the language of your Spotify application. This parameter affects how artist names and track titles are displayed in your Telegram channel. For example, if an artist's name is written in a non-Latin alphabet (Cyrillic, Hangul, Kanji, etc.), with `SPOTIFY_LANGUAGE=EN` the name might display in Latin script, while with `SPOTIFY_LANGUAGE=RU` (or the corresponding language code) — in the original script.

**Limitations Note:** The Spotify Web API endpoint for playback status may require a Premium account (depending on Spotify limitations/policy).

### Telegram Setup

1. Go to [my.telegram.org](https://my.telegram.org)
2. Create a new application
3. Copy API ID and API Hash to `.env`
4. Create a bot via [@BotFather](https://t.me/BotFather)
5. Copy the bot token to `.env`

#### Preparing the Telegram channel

1. Create a **public** Telegram channel (a public channel is required to add it to your profile)
2. Add your bot as an **administrator** with full permissions (this is mandatory for the bot to work)
3. Get the channel ID (`TG_CHANNEL_ID`) — use any Telegram client or bot that can show chat IDs (for example, forward a message from the channel to a bot like [@userinfobot](https://t.me/userinfobot), or retrieve it programmatically via Telethon). Copy the resulting numeric ID (e.g. `1234567890`) into `.env`
4. Post **two** messages in the channel in the following order:
   - **First message** (`TG_MEDIA_MESSAGE_ID`) — send any image **as a compressed photo** (not as a file) and add any text as a caption. For example, you can send `default.png` from the `session/` folder. This message will be used for the album cover. The Telegram API only allows replacing media in a message that already contains an image, so a plain text message will not work
   - **Second message** (`TG_TEXT_MESSAGE_ID`) — send any text. This message will be used for track info
5. Get the ID of each message — right-click (or long-press) the message and select "Copy Message Link". The link looks like `https://t.me/channel_name/123` — the last number (`123`) is the message ID. Copy the first message ID (with the image) into `TG_MEDIA_MESSAGE_ID`, and the second (with text) into `TG_TEXT_MESSAGE_ID` in `.env`

#### Telegram Sessions

SpotiPlayGram uses **two separate Telethon sessions**:

- **User session** (`session/user.session*`) — requires full Telegram authorization (phone number and cloud password, if set). Used for actions that need a real user
- **Bot session** (`session/bot.session*`) — uses `TG_BOT_TOKEN`. Used for the control panel (`/start`) and channel updates

To run on another device (PC, server, VPS, etc.) after the initial authorization, copy the files `session/user.session*`, `session/bot.session*` and `session/.cache`.

### Genius Setup (Optional)

1. Go to [Genius API Clients](https://genius.com/api-clients)
2. Create a new API client
3. Generate an access token
4. Copy the token to `.env`

### Mistral AI Setup (Optional)

1. Go to [Mistral AI Console](https://console.mistral.ai)
2. Create an API key
3. Copy the key to `.env`

**Registration Note:** Mistral registration may require a phone number for verification. At the time of writing, Mistral provides a free tier (up to ~1B tokens/month), which is more than enough for SpotiPlayGram (even if you listen to music 24/7, usage is usually far below ~2M tokens/month).

**Accuracy Note:** Without Mistral AI, the selection of lyrics links from Genius will be less accurate. The built-in algorithm via the Genius API works well for most cases, but combined with Mistral AI, it significantly improves accuracy.

**Caching Note:** Mistral AI uses a 1-hour result cache for performance optimization and token consumption reduction. Repeated requests for the same track will be processed instantly without additional API calls.

**Model Selection Note:** You can compare Mistral model benchmarks at [Artificial Analysis](https://artificialanalysis.ai/providers/mistral) to choose a more accurate model for selecting the best lyrics URL from Genius search results. However, keep in mind that the smartest and heaviest models may start returning **HTTP 429** errors over time (`Service tier capacity exceeded for this model`) due to free tier rate limits — if this happens, switch to a lighter model.

---

## ▶️ Running the Application

**Windows:**
```bash
python start.py
```

You can also run the `start.py` file by double-clicking or via context menu "Open with Python".

**Linux:**
```bash
python3 start.py
```

**Linux (desktop / GUI):** It is recommended to run SpotiPlayGram from a regular desktop session (GNOME/KDE/etc.), because Spotify authorization requires opening a browser on the first run.

**Linux (SSH connection):** For continuous operation when connected via SSH, use `screen` or `tmux` to keep the application running after disconnecting from the session:
```bash
# Using screen
screen -S SpotiPlayGram
python3 start.py
# Press Ctrl+A, then D to detach
# To reconnect: screen -r SpotiPlayGram

# Using tmux
tmux new -s SpotiPlayGram
python3 start.py
# Press Ctrl+B, then D to detach
# To reconnect: tmux attach -t SpotiPlayGram
```

**macOS:**
```bash
python3 start.py
```

On first run:
- A virtual environment will be created automatically
- Required dependencies will be installed
- You will be prompted to authenticate with Telegram (phone number login, plus cloud password if set)
- You will be prompted to authenticate with Spotify (browser will open)

Dependency auto-updates are optional and controlled by `ENABLE_DEP_UPDATES` in `.env`.

### 24/7 Operation

If you want the application to run continuously 24/7, you have two options:
- **Keep the application running on your PC** — the computer must be on and the application must be running at all times
- **Use your own server** — deploy the application on a VPS or dedicated server for uninterrupted operation

---

## 🎮 Bot Commands

Send `/start` to your bot to access the control panel:

- **📢 Playback status** — Enable/disable Spotify playback status updates in your Telegram channel
  - **▶️ Enable** — Turn on playback status updates
  - **⏹️ Disable** — Turn off playback status updates
- **⚙️ Genius settings** — Configure lyrics search parameters through Genius API
  - **🤖 Mistral AI models** — Select Mistral AI models for intelligent lyrics URL selection from Genius search results
    - **🔄** — Refresh the Mistral model list
  - **Version patterns** — Manage list of words/phrases to detect remixes, covers, versions in track titles (e.g. "remix", "cover", "acoustic")
  - **Feat patterns** — Manage list of words/phrases to detect featuring artists in track titles (e.g. "feat", "ft", "with")
    - **Add** — Add a new pattern to the selected category
    - **Delete** — Remove an existing pattern from the selected category
    - **Confirm add** — Confirm adding the entered patterns
    - **Confirm delete** — Confirm deleting the selected patterns

---

## 🔄 How It Works

1. **Polling Loop**: The application polls Spotify API every second (configurable)
2. **Track Detection**: When a new track is detected, it fetches full metadata
3. **Lyrics Search**: Searches Genius for the track with fuzzy matching
4. **AI Selection**: Mistral AI helps Genius select the accurate lyrics URL
5. **Telegram Update**: Updates channel messages, title, and photo
6. **State Management**: Saves current state to avoid duplicate updates

---

## 🛠️ Troubleshooting

### Spotify

**Authorization error / invalid credentials**
- Make sure `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in `.env` match the values from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
- Verify that `SPOTIFY_REDIRECT_URI` in `.env` exactly matches the Redirect URI configured in your Dashboard app settings (default is `http://localhost:8888/callback`)

**Spotify token expired / unable to get playback data**
- Delete the `session/.cache` file and restart the application — authorization will be repeated via browser
- Check that the application in Spotify Developer Dashboard has not been deleted and is still active
- If you have a free Spotify account, the playback status API may not be available — Premium is required

### Telegram

**Authorization error / invalid credentials**
- Verify that `TG_API_ID` and `TG_API_HASH` in `.env` match the values from [my.telegram.org](https://my.telegram.org)
- Make sure `TG_BOT_TOKEN` is valid — you can regenerate the token via [@BotFather](https://t.me/BotFather)

**Channel not updating**
- Make sure the bot is added to the channel as an **administrator** with full permissions
- Verify that the messages with IDs `TG_MEDIA_MESSAGE_ID` and `TG_TEXT_MESSAGE_ID` actually exist in the channel and have not been deleted
- Make sure `TG_CHANNEL_ID` is correct (numeric channel ID, e.g. `1234567890`)

**Session file errors**
- If the application cannot connect to Telegram or throws session errors, delete `session/user.session*` and `session/bot.session*`, then restart the application to re-authorize
- Make sure another instance of the application is not running simultaneously — two processes cannot use the same session

### Genius

**Lyrics search not working**
- Check that `GENIUS_ACCESS_TOKEN` in `.env` is valid — you can regenerate the token at [Genius API Clients](https://genius.com/api-clients)
- Make sure `ENABLE_GENIUS=True` in `.env`

### Mistral AI

**Mistral AI not responding / request errors**
- Verify that `MISTRAL_API_KEY` in `.env` is correct and the key is active in [Mistral AI Console](https://console.mistral.ai)
- Make sure `ENABLE_MISTRAL=True` in `.env`
- If the model returns an error, make sure it is available — models are periodically updated and may be renamed or removed. You can check the current model list via the bot (⚙️ Genius settings → 🤖 Mistral AI models)
- If timeouts occur frequently, try switching to a lighter model (e.g. `mistral-small-latest`)

---

## 📞 Support

If you encounter any issues or have questions, please open an [issue](https://github.com/xsfeelyou/SpotiPlayGram/issues).
