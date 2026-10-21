# SpotiPlayGram

[![README in Russian](https://img.shields.io/badge/README-RU-blue)](README_RU.md)

**SpotiPlayGram** is a Python application that displays your current Spotify track in a Telegram channel in real time. It automatically updates the channel with album artwork and track information, and optionally provides a link to lyrics on Genius.

---

## 💡 Idea

The main idea is to **add the channel you created to your Telegram profile** (Settings → Edit profile → Personal channel). When playing music on Spotify, the channel name automatically changes to **"Artist – Track Name"**, and the message displays **"Now Playing on Spotify"**.

This creates a **full-fledged Spotify playback status** right in your Telegram profile — other users can see what you're currently listening to, just like a classic Spotify status.

---

## 📸 Screenshots

![SpotiPlayGram](https://i.ibb.co/GvCf78JP/1.png)
![SpotiPlayGram](https://i.ibb.co/0Vf5653h/2.png) ![SpotiPlayGram](https://i.ibb.co/Y7RCckGx/3.png)
![SpotiPlayGram](https://i.ibb.co/s92xtydB/4.png) ![SpotiPlayGram](https://i.ibb.co/zWQ3G7z1/5.png)
![SpotiPlayGram](https://i.ibb.co/8LXK7fhD/6.png) ![SpotiPlayGram](https://i.ibb.co/j96PPgSw/7.png)

---

## ✨ Features

### 🎵 Spotify Integration

- Real-time tracking of the currently playing track
- Displays artist name, track title, album name, and release date
- Automatic album cover download and display

### 📢 Telegram Channel Updates

- Updates channel messages with track info and album artwork
- Dynamically changes the channel name to the current track
- Updates the channel profile photo with the album cover
- Automatically removes all messages except the channel's first message and the two main status messages

### 📝 Genius Lyrics Integration

- Automatically searches for the lyrics URL on Genius
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

- Cross-platform support (Windows, Linux, macOS, FreeBSD)
- Automatic virtual environment creation
- Optional daily dependency update checks (at 00:00 UTC)
- Separate scripts to enable and disable autostart
- Protection against running on multiple devices at the same time

---

## 📋 Requirements

- Python 3.10 or higher
- Spotify Developer account
- Spotify Premium (may be required for the playback status API)
- Telegram API credentials (`api_id`, `api_hash`)
- Telegram bot token
- Telegram user account (phone number login, plus a two-step verification password if set)
- (Optional) Genius API token
- (Optional) Mistral AI API key

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/xsfeelyou/SpotiPlayGram.git
cd SpotiPlayGram
```

### 2. Configure Environment Variables

Copy the `.env-example` file to `.env` and fill in your credentials:

```bash
# Windows
copy .env-example .env

# Linux, macOS, FreeBSD
cp .env-example .env
```

```env
# Spotify API
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:47474/callback
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

# Genius API (optional)
GENIUS_ACCESS_TOKEN=your_genius_token
ENABLE_GENIUS=True  # Enable lyrics search through Genius API
GENIUS_DETAILED_LOG=False  # Detailed logging of Genius operations

# Mistral AI (optional)
MISTRAL_API_KEY=your_mistral_api_key
ENABLE_MISTRAL=True  # Enable Mistral AI for accurate lyrics URL selection
MISTRAL_MODEL=codestral-latest  # Mistral AI model (default)

# Auth retries
AUTH_RETRY_MAX_RETRIES=2  # Maximum number of authorization attempts
AUTH_RETRY_DELAY_BASE=1  # Base delay between authorization attempts in seconds

# Dependency auto-updates
ENABLE_DEP_UPDATES=False  # Automatic dependency updates on startup

# Update interval in seconds
UPDATE_INTERVAL=1  # Spotify API polling interval in seconds
```

---

## 🔧 Configuration

### 🎵 Spotify Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application
3. In the application settings, add the Redirect URI: `http://127.0.0.1:47474/callback`
4. Copy the Client ID and Client Secret to `.env`

**Authorization Note:** Spotify uses **OAuth 2.0 authorization** — on the first launch, a browser will open to log in to your account. **Authorization via SSH is not possible**, as there is no access to a graphical interface. You must perform the initial authorization on a computer with a graphical interface (Windows, Linux with GUI, macOS, or FreeBSD), and then transfer the session files to the target device.

**Language Note:** Choose the correct value for `SPOTIFY_LANGUAGE` in `.env` — it should match the language of your Spotify application. This parameter affects how artist names and track titles are displayed in your Telegram channel. For example, if an artist's name is written in a non-Latin script (Cyrillic, Hangul, CJK characters, etc.), with `SPOTIFY_LANGUAGE=EN` the name may be displayed in Latin script, while with `SPOTIFY_LANGUAGE=RU` (or the corresponding language code) it is displayed in the original script.

**Limitations Note:** The Spotify Web API endpoint for playback status may require a Premium account (depending on Spotify limitations/policy).

### 📢 Telegram Setup

1. Go to [my.telegram.org](https://my.telegram.org)
2. Create a new application
3. Copy `api_id` and `api_hash` to `.env`
4. Create a bot via [@BotFather](https://t.me/BotFather)
5. Copy the bot token to `.env`

#### Preparing the Telegram Channel

1. Create a **public** Telegram channel (a public channel is required to add it to your profile)
2. Add your bot as an **administrator** with full permissions (this is mandatory for the bot to work)
3. Get the channel ID (`TG_CHANNEL_ID`) — use any Telegram client or bot that can show chat IDs (for example, forward a message from the channel to a bot like [@userinfobot](https://t.me/userinfobot), or retrieve it programmatically via Telethon). Copy the resulting numeric ID (e.g. `1234567890`) into `.env`
4. Post **two** messages in the channel in the following order:
   - **First message** (`TG_MEDIA_MESSAGE_ID`) — send any image **as a compressed photo** (not as a file) and add any text as a caption. For example, you can send `default.png` from the `session/spotify/` folder. This message will be used for the album cover. The Telegram Bot API only allows replacing media in a message that already contains an image, so a plain text message will not work.
   - **Second message** (`TG_TEXT_MESSAGE_ID`) — send any text. This message will be used for track info.
5. Get the ID of each message — right-click (or long-press) the message and select "Copy Message Link". The link looks like `https://t.me/channel_name/123` — the last number (`123`) is the message ID. Copy the first message ID (with the image) into `TG_MEDIA_MESSAGE_ID`, and the second (with text) into `TG_TEXT_MESSAGE_ID` in `.env`.

#### Telegram Sessions

SpotiPlayGram uses **two separate Telethon sessions**:

- **User session** (`session/telegram/user.session*`) — requires full Telegram authorization (phone number and two-step verification password, if set). Used for actions that need a real user
- **Bot session** (`session/telegram/bot.session*`) — uses `TG_BOT_TOKEN`. Used for the control panel (`/start`) and channel updates

To run the app on another device (PC, server, VPS, etc.) after the initial authorization, copy the files `session/telegram/user.session*`, `session/telegram/bot.session*`, and `session/spotify/.cache`.

Only one device can run SpotiPlayGram at a time. On startup, the app checks a marker in both bot text fields. If the app is already running on another device, the second device shows a message and does not open the Telegram sessions, even if the session files and `.cache` were copied. If the previous run crashed, the protection stops working after about 90 seconds. For this protection to work, at least one of the two fields — About or Description — must be free.

#### Device Lock Marker in the Bot Fields

The bot has two text fields:

- **About** (`short_description`) — the short text under the bot name in its profile.
- **Description** (`description`) — the text shown in the chat with the bot before the first `/start`.

While the app is running, one of these fields contains a service string like:

`SPG1|7f3a9c21|b4e8d0c6a2f1|5c1e9a7b3d2f4608a1c9e5b7|1735689600|1735689480`

The app writes the marker to the About field and uses the Description field only if the About field already contains your own text. This is not user-facing text but a device lock marker that prevents the app from running on several devices at once. If the marker is in the About field, it is visible in the bot profile; if it is in the Description field, it is visible in the chat before the first `/start`. The parts are as follows:

| Part                       | Meaning                                        |
| -------------------------- | ---------------------------------------------- |
| `SPG1`                     | Marker format version                          |
| `7f3a9c21`                 | Current run identifier                         |
| `b4e8d0c6a2f1`             | Device identifier (hash)                       |
| `5c1e9a7b3d2f4608a1c9e5b7` | Machine identifier (hash)                      |
| `1735689600`               | Last heartbeat time (updated every 30 seconds) |
| `1735689480`               | Start time                                     |

On a normal shutdown, the app clears the field. If it was closed abnormally, the marker stays, but after 90 seconds it is considered stale and the next launch can overwrite it.

**Can I edit the About and Description fields while the app is running?**

The marker is only ever in one of the two fields — About or Description.

- Do not edit the field that currently contains the marker: the app checks it every 30 seconds, detects the replacement, and shuts down with a lock-lost warning. If you only clear the field, the app writes the marker back on the next check.
- You can edit the field without the marker: the app only reads it and does not change it.
- If both the About and Description fields contain your own text, the next launch cannot acquire the lock and runs without cross-device protection (a warning is written to the log).
- After the app stops, the marker field is cleared and you can edit it freely; the next launch puts the marker back into the About field, or into the Description field if the About field is taken.

### 📝 Genius Setup (Optional)

1. Go to [Genius API Clients](https://genius.com/api-clients)
2. Create a new API client
3. Generate an access token
4. Copy the token to `.env`

### 🤖 Mistral AI Setup (Optional)

1. Go to [Mistral AI Console](https://console.mistral.ai)
2. Create an API key
3. Copy the key to `.env`

**Registration Note:** Mistral registration may require a phone number for verification.

**Accuracy Note:** Without Mistral AI, the selection of lyrics links from Genius will be less accurate. The built-in algorithm via the Genius API works well in most cases, but when combined with Mistral AI, it is significantly more accurate.

**Caching Note:** Mistral AI uses a 1-hour result cache for performance optimization. Repeated requests for the same track will be processed instantly without additional API calls.

**Model Selection Note:** You can compare Mistral model benchmarks at [Artificial Analysis](https://artificialanalysis.ai/providers/mistral) to choose a more accurate model for selecting the best lyrics URL from Genius search results. Before choosing a model, check how many tokens per month are available for that model. Monthly token limits for each model are available at [Mistral AI Limits](https://admin.mistral.ai/plateforme/limits). During normal use, SpotiPlayGram usually uses up to 2 million tokens per month. In the worst case, if you listen to music continuously 24/7 and all tracks are no longer than 1 minute, usage will be about 50 million tokens per month. You can check actual token usage at [Mistral AI Usage](https://admin.mistral.ai/organization/usage?workspace_dialog_use_org_currency=false). However, keep in mind that the smartest and heaviest models may start returning **HTTP 429** errors over time due to free-tier rate limits — if this happens, switch to a lighter model.

---

## ▶️ Running the Application

| System                | Run                |
| --------------------- | ------------------ |
| Windows               | `python start.py`  |
| Linux, macOS, FreeBSD | `python3 start.py` |

On Windows, you can also run `start.py` by double-clicking it or via the context menu "Open with Python".

**Linux / FreeBSD (desktop / GUI):** It is recommended to run SpotiPlayGram from a regular desktop session (GNOME, KDE, etc.), because Spotify authorization requires opening a browser on the first run.

**Linux / FreeBSD (SSH connection):** For continuous operation when connected via SSH, use `screen` or `tmux` to keep the application running after disconnecting from the session. On FreeBSD, install them beforehand if needed: `pkg install screen` or `pkg install tmux`.

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

On first run:

- A virtual environment will be created automatically.
- Required dependencies will be installed.
- You will be prompted to authenticate with Telegram (phone number login, plus a two-step verification password if set).
- You will be prompted to authenticate with Spotify (browser will open).

Dependency auto-updates are optional and controlled by `ENABLE_DEP_UPDATES` in `.env`.

### 🔁 Autostart

`start.py` does not enable or disable autostart. Use the standalone scripts. Run `start.py` once before using them so that `venv` and the dependencies are created.

**Enable and disable**

| System                | Enable                                        | Disable                                        |
| --------------------- | --------------------------------------------- | ---------------------------------------------- |
| Windows               | `venv\Scripts\python.exe enable_autostart.py` | `venv\Scripts\python.exe disable_autostart.py` |
| Linux, macOS, FreeBSD | `venv/bin/python3 enable_autostart.py`        | `venv/bin/python3 disable_autostart.py`        |

The scripts select the right method for your system. After finishing, they wait for Enter so the console stays open. Add `--no-pause` to skip the wait.

**Check whether autostart is enabled**

| System  | Check                                                                                                                                                                                |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Windows | Task Manager → Startup apps → `SpotiPlayGram` (check whether it is enabled). Or in Command Prompt: `dir "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SpotiPlayGram.lnk"` |
| Linux   | `systemctl --user is-enabled spotiplaygram.service` (if enabled by the script) or `ls ~/.config/autostart/spotiplaygram.desktop` / `crontab -l` (if enabled manually)                |
| macOS   | `ls ~/Library/LaunchAgents/com.spotisplaygram.plist` (`SpotiPlayGram` is listed in System Settings → Login Items)                                                                    |
| FreeBSD | `ls ~/.config/autostart/spotiplaygram.desktop` or `crontab -l`                                                                                                                       |

If the entry or file is missing, autostart is disabled.

**Enable manually**

<details>
<summary>Windows</summary>

Open Command Prompt in the project folder (for example, type `cmd` in the File Explorer address bar) and run:

```bat
copy /y "venv\Scripts\pythonw.exe" "venv\Scripts\SpotiPlayGram.exe" >nul
powershell -NoProfile -Command "$startup=[Environment]::GetFolderPath('Startup'); $shell=New-Object -ComObject WScript.Shell; $link=$shell.CreateShortcut((Join-Path $startup 'SpotiPlayGram.lnk')); $link.TargetPath=(Resolve-Path 'venv\Scripts\SpotiPlayGram.exe').Path; $link.Arguments=[char]34+(Resolve-Path 'start.py').Path+[char]34; $link.WorkingDirectory=(Get-Location).Path; $link.Save()"
```

</details>

<details>
<summary>Linux and FreeBSD</summary>

Run from the project folder:

```bash
PROJECT_DIR="$(pwd)"
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/spotiplaygram.desktop <<EOF
[Desktop Entry]
Type=Application
Name=SpotiPlayGram
Exec="$PROJECT_DIR/venv/bin/python3" "$PROJECT_DIR/start.py"
Path=$PROJECT_DIR
Terminal=false
X-GNOME-Autostart-enabled=true
EOF
```

For a headless server, add an entry to `crontab`:

```bash
PROJECT_DIR="$(pwd)"
(crontab -l 2>/dev/null; echo "@reboot \"$PROJECT_DIR/venv/bin/python3\" \"$PROJECT_DIR/start.py\" # SpotiPlayGram") | crontab -
```

</details>

<details>
<summary>macOS</summary>

Run from the project folder:

```bash
PROJECT_DIR="$(pwd)"
LAUNCHER="$HOME/Library/Application Support/SpotiPlayGram/SpotiPlayGram"
mkdir -p "$(dirname "$LAUNCHER")" "$HOME/Library/LaunchAgents"
printf '#!/bin/sh\nexec "%s" "%s"\n' "$PROJECT_DIR/venv/bin/python3" "$PROJECT_DIR/start.py" > "$LAUNCHER"
chmod +x "$LAUNCHER"
cat > "$HOME/Library/LaunchAgents/com.spotisplaygram.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.spotisplaygram</string>
  <key>ProgramArguments</key><array>
    <string>$LAUNCHER</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
  <key>RunAtLoad</key><true/>
</dict></plist>
EOF
launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/com.spotisplaygram.plist"
```

</details>

**Disable manually**

<details>
<summary>Windows</summary>

```bat
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SpotiPlayGram.lnk"
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v SpotiPlayGram /f 2>nul
```

</details>

<details>
<summary>Linux and FreeBSD</summary>

```bash
rm -f ~/.config/autostart/spotiplaygram.desktop
crontab -l 2>/dev/null | grep -v SpotiPlayGram | crontab -
```

If autostart was enabled by the script through systemd:

```bash
systemctl --user disable --now spotiplaygram.service
rm -f ~/.config/systemd/user/spotiplaygram.service
systemctl --user daemon-reload
```

</details>

<details>
<summary>macOS</summary>

```bash
launchctl bootout gui/$(id -u)/com.spotisplaygram 2>/dev/null || true
rm -f ~/Library/LaunchAgents/com.spotisplaygram.plist
rm -f "$HOME/Library/Application Support/SpotiPlayGram/SpotiPlayGram"
```

</details>

If you move the program folder, enable autostart again — the entry stores the path to `start.py`.

### 🕒 24/7 Operation

If you want the application to run continuously 24/7, you have two options:

- **Keep the application running on your PC** — the computer must be on, and the application must be running at all times.
- **Use your own server** — deploy the application on a VPS or dedicated server for uninterrupted operation.

---

## 🎮 Bot Commands

Send `/start` to your bot to access the control panel:

- **📢 Playback status** — Enable/disable Spotify playback status updates in your Telegram channel
  - **▶️ Enable** — Turn on playback status updates
  - **⏹️ Disable** — Turn off playback status updates
- **⚙️ Genius settings** — Configure lyrics search parameters through the Genius API
  - **🤖 Mistral AI models** — Select Mistral AI models for intelligent lyrics URL selection from Genius search results
    - **🔄** — Refresh the Mistral model list
  - **Version patterns** — Manage the list of words/phrases to detect remixes, covers, and versions in track titles (e.g. "remix", "cover", "acoustic")
    - **Add** — Add new patterns to the selected category
    - **Delete** — Remove existing patterns from the selected category
    - **Confirm add** — Confirm adding the entered patterns
    - **Confirm delete** — Confirm deleting the selected patterns
  - **Feat patterns** — Manage the list of words/phrases to detect featuring artists in track titles (e.g. "feat", "ft", "with")
    - **Add** — Add new patterns to the selected category
    - **Delete** — Remove existing patterns from the selected category
    - **Confirm add** — Confirm adding the entered patterns
    - **Confirm delete** — Confirm deleting the selected patterns

---

## 🔄 How It Works

1. **Polling Loop:** The application polls the Spotify API every second (configurable).
2. **Track Detection:** When a new track is detected, it fetches full track metadata.
3. **Lyrics Search:** Searches Genius for the track with fuzzy matching.
4. **AI Selection:** Mistral AI selects the most accurate lyrics URL from the Genius search results.
5. **Telegram Update:** Updates channel messages, channel name, and photo.
6. **State Management:** Saves the current state to avoid duplicate updates.

---

## 🛠️ Troubleshooting

### 🎵 Spotify

**Authorization error / invalid credentials**

- Make sure `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in `.env` match the values from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
- Verify that `SPOTIFY_REDIRECT_URI` in `.env` exactly matches the Redirect URI configured in your Dashboard app settings (default is `http://127.0.0.1:47474/callback`).

**Spotify token expired / unable to get playback data**

- Delete the `session/spotify/.cache` file and restart the application — authorization will be performed again in the browser.
- Check that the application in the Spotify Developer Dashboard has not been deleted and is still active.
- If you have a free Spotify account, the playback status API may not be available — Premium is required.

### 📢 Telegram

**Authorization error / invalid credentials**

- Verify that `TG_API_ID` and `TG_API_HASH` in `.env` match the values from [my.telegram.org](https://my.telegram.org).
- Make sure `TG_BOT_TOKEN` is valid — you can regenerate the token via [@BotFather](https://t.me/BotFather).

**Channel not updating**

- Make sure the bot is added to the channel as an **administrator** with full permissions.
- Verify that the messages with IDs `TG_MEDIA_MESSAGE_ID` and `TG_TEXT_MESSAGE_ID` actually exist in the channel and have not been deleted.
- Make sure `TG_CHANNEL_ID` is correct (numeric channel ID, e.g. `1234567890`).

**Session file errors**

- If the application cannot connect to Telegram or throws session errors, delete `session/telegram/user.session*` and `session/telegram/bot.session*`, then restart the application to re-authorize.
- Make sure another instance of the application is not running simultaneously — two processes cannot use the same session.
- If the application is already running, a second launch shows a message and waits for Enter so the console stays open.

**The app exits on a second device**

- Only one device can run at a time. Stop the app on the device named in the message and press Enter to close the console.
- After a crash, the protection is removed after about 90 seconds. You can start the app on the same device immediately.
- At least one of the two fields — About or Description — must be free for this protection to work.

### 📝 Genius

**Lyrics search not working**

- Check that `GENIUS_ACCESS_TOKEN` in `.env` is valid — you can regenerate the token at [Genius API Clients](https://genius.com/api-clients).
- Make sure `ENABLE_GENIUS=True` in `.env`.

### 🤖 Mistral AI

**Mistral AI not responding / request errors**

- Verify that `MISTRAL_API_KEY` in `.env` is correct and the key is active in [Mistral AI Console](https://console.mistral.ai).
- Make sure `ENABLE_MISTRAL=True` in `.env`.
- If the model returns an error, make sure it is available — models are periodically updated and may be renamed or removed. You can check the current model list via the bot (⚙️ Genius settings → 🤖 Mistral AI models).
- If timeouts occur frequently, try switching to a lighter model (e.g. `ministral-3b-latest`).

---

## 📞 Support

If you encounter any issues or have questions, please open an [issue](https://github.com/xsfeelyou/SpotiPlayGram/issues).
