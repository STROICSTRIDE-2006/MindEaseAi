import os
import re
import json
import time
import html
import hashlib
import inspect
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, quote
from zoneinfo import ZoneInfo

import gradio as gr
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "7860"))

TINYFISH_ENDPOINT = "https://agent.tinyfish.ai/v1/automation/run-sse"

# The app still opens without a key and shows a clear message instead of crashing.
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
MISSING_GEMINI = "GEMINI_API_KEY is missing. Add it to your .env file and restart."

# One shared HTTP session = faster repeated TinyFish calls.
http = requests.Session()


# =========================================================
# 2. COMPATIBILITY HELPERS (Gradio 4 / 5 / 6)
# =========================================================

def supported_kwargs(func, **kwargs):
    """Keep only the keyword arguments that `func` really accepts."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    return {k: v for k, v in kwargs.items() if k in params}


# Gradio 6 takes theme/css/head in launch(); Gradio 5 and older in Blocks().
LAUNCH_ACCEPTS_UI = "css" in inspect.signature(gr.Blocks.launch).parameters


# =========================================================
# 3. CACHE
# =========================================================

class TTLCache:

    def __init__(self, ttl=3600, max_items=50):
        self.ttl = ttl
        self.max_items = max_items
        self.data = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            item = self.data.get(key)

            if not item:
                return None

            value, created = item

            if time.time() - created > self.ttl:
                del self.data[key]
                return None

            return value

    def set(self, key, value):
        with self.lock:
            if len(self.data) >= self.max_items:
                oldest = min(self.data, key=lambda k: self.data[k][1])
                del self.data[oldest]

            self.data[key] = (value, time.time())


research_cache = TTLCache(ttl=3600)
roadmap_cache = TTLCache(ttl=3600)


def make_key(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


# =========================================================
# 4. GEMINI STREAMING (Interactions API + safe fallback)
# =========================================================

STREAM_INTERVAL = 0.08  # seconds between UI refreshes


def _stream_interactions(prompt, max_tokens):
    interactions = getattr(gemini_client, "interactions")

    stream = interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        generation_config={
            "thinking_level": "low",
            "max_output_tokens": max_tokens,
        },
        stream=True,
    )

    for event in stream:
        event_type = (
            getattr(event, "event_type", None)
            or getattr(event, "type", None)
        )

        if event_type == "interaction.failed":
            raise RuntimeError(
                str(getattr(event, "error", "Gemini interaction failed."))
            )

        if event_type == "step.delta":
            delta = getattr(event, "delta", None)

            if delta and getattr(delta, "type", None) == "text":
                piece = getattr(delta, "text", "")

                if piece:
                    yield piece


def _stream_generate_content(prompt, max_tokens):
    config = types.GenerateContentConfig(max_output_tokens=max_tokens)

    for chunk in gemini_client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    ):
        piece = getattr(chunk, "text", None)

        if piece:
            yield piece


def gemini_stream(prompt, max_tokens=1200):
    """Yield the growing Gemini answer (UI updates are throttled)."""

    if gemini_client is None:
        raise ValueError(MISSING_GEMINI)

    collected = ""
    last_push = 0.0

    def run(source):
        nonlocal collected, last_push

        for piece in source:
            collected += piece
            now = time.monotonic()

            if now - last_push >= STREAM_INTERVAL:
                last_push = now
                yield collected

    if hasattr(gemini_client, "interactions"):
        try:
            yield from run(_stream_interactions(prompt, max_tokens))
        except Exception as error:
            if collected:
                raise
            print("Interactions API failed, using fallback:", repr(error))
            yield from run(_stream_generate_content(prompt, max_tokens))
    else:
        yield from run(_stream_generate_content(prompt, max_tokens))

    if not collected.strip():
        raise RuntimeError("Gemini returned an empty response.")

    yield collected


# =========================================================
# 5. MIND ORGANIZER
# =========================================================

def organize_thoughts(thoughts):

    if not thoughts or not thoughts.strip():
        yield "Please write down your thoughts first."
        return

    prompt = f"""
You are MindEase AI, a friendly personal life organizer.

Organize the user's thoughts into simple priorities.

USER'S THOUGHTS:
{thoughts}

Use simple and supportive language.

Follow this exact format:

## DO NOW
List the most urgent and important tasks.

## DO NEXT
List tasks that should be done after urgent tasks.

## CAN WAIT
List tasks that are not urgent.

## YOUR NEXT SMALL STEP
Give exactly one simple action the user can complete in 10 minutes.

## REMINDER
Give one short, encouraging reminder.

IMPORTANT:
- Do not assume something can wait without a reason.
- Respect deadlines and urgency mentioned by the user.
- Do not invent deadlines.
- Do not give medical or mental-health diagnoses.
- Keep the answer practical and easy to understand.
"""

    yield "Understanding your thoughts..."

    try:
        for answer in gemini_stream(prompt, max_tokens=1200):
            yield answer

    except Exception as e:
        yield f"Gemini error: {str(e)}"


# =========================================================
# 6. JSON EXTRACTION + TASK PLANNER
# =========================================================

def extract_json(text):

    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Could not find JSON in Gemini's response.")

    return json.loads(text[start:end + 1])


def build_task_plan(thoughts):

    if not thoughts or not thoughts.strip():
        yield [], "Please enter your thoughts first."
        return

    prompt = f"""
You are MindEase AI's task planning assistant.

Convert the user's thoughts into an actionable task plan.

USER'S THOUGHTS:
{thoughts}

Return ONLY valid JSON:

{{
  "tasks": [
    {{
      "task": "Task description",
      "priority": "High",
      "estimated_time": "30 minutes",
      "deadline": "Not specified",
      "reason": "Short reason for this priority"
    }}
  ]
}}

RULES:
1. Priority must be High, Medium, or Low.
2. Estimate realistic completion time.
3. Never invent a real deadline.
4. If no deadline is given, write "Not specified".
5. Suggested target dates must be labelled Suggested.
6. Put urgent and deadline-based tasks first.
7. Break large tasks into manageable actions.
8. Do not add unrelated tasks.
9. Include up to 10 tasks.
10. Do not include completed tasks unless mentioned as pending.
"""

    try:
        yield [], "Creating your actionable task plan..."

        full_response = ""

        for partial in gemini_stream(prompt, max_tokens=1800):
            full_response = partial

        data = extract_json(full_response)
        tasks = data.get("tasks", [])

        rows = []

        for task in tasks:
            if not isinstance(task, dict):
                continue

            rows.append([
                str(task.get("task", "")),
                str(task.get("priority", "Medium")),
                str(task.get("estimated_time", "Not estimated")),
                str(task.get("deadline", "Not specified")),
                str(task.get("reason", "")),
                False,
            ])

        if not rows:
            yield [], "No tasks were generated. Try adding more details."
            return

        yield rows, f"Your task plan is ready! {len(rows)} tasks created."

    except Exception as e:
        yield [], f"Task planner error: {str(e)}"


# =========================================================
# 7. TINYFISH HELPERS
# =========================================================

def clean_url(url):

    if not isinstance(url, str):
        return ""

    url = html.unescape(url.strip())
    url = url.replace(" ", "%20")

    if not url.startswith(("http://", "https://")):
        return ""

    try:
        parsed = urlparse(url)

        if not parsed.hostname:
            return ""

        return quote(url, safe=":/?#[]@!$&'()*+,;=%-._~")

    except Exception:
        return ""


def escape_markdown(value):
    value = str(value or "")
    return value.replace("[", "\\[").replace("]", "\\]")


def extract_resources(data):

    found = []
    seen = set()

    def walk(obj):

        if isinstance(obj, dict):

            name = obj.get("title") or obj.get("name") or obj.get("label")
            url = obj.get("url") or obj.get("link") or obj.get("href")
            description = (
                obj.get("description")
                or obj.get("snippet")
                or obj.get("summary")
                or ""
            )

            if isinstance(url, str):
                valid_url = clean_url(url)

                if valid_url and valid_url not in seen:
                    seen.add(valid_url)

                    found.append({
                        "name": str(name or "Useful resource"),
                        "url": valid_url,
                        "description": str(description),
                    })

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

        elif isinstance(obj, str):
            text = obj.strip()

            if text.startswith(("{", "[")):
                try:
                    walk(json.loads(text))
                except Exception:
                    pass

    walk(data)

    return found[:15]


def format_research_result(resources, query):

    if not resources:
        return (
            f"## Research results for: {query}\n\n"
            "TinyFish completed the search, but no structured links "
            "were found. Try a more specific search."
        )

    output = f"## Research results for: {query}\n\n"

    for index, item in enumerate(resources, start=1):
        name = escape_markdown(item["name"])
        description = escape_markdown(item["description"])

        output += f"### {index}. [{name}]({item['url']})\n"

        if description:
            output += f"{description}\n"

        output += "\n"

    return output


# =========================================================
# 8. TINYFISH ONLINE RESEARCH
# =========================================================

def research_online(query):

    if not query or not query.strip():
        yield (
            "Please enter something to research.",
            None,
            gr.update(interactive=False),
        )
        return

    if not TINYFISH_API_KEY:
        yield (
            "TINYFISH_API_KEY is missing from your .env file.",
            None,
            gr.update(interactive=False),
        )
        return

    query = query.strip()
    cache_key = query.lower()

    cached = research_cache.get(cache_key)

    if cached:
        resources = extract_resources(cached)

        yield (
            format_research_result(resources, query),
            cached,
            gr.update(interactive=bool(resources)),
        )
        return

    headers = {
        "X-API-Key": TINYFISH_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "url": "https://www.google.com",
        "goal": (
            f"Search the web for: {query}. "
            "Find useful, relevant and trustworthy resources. "
            "Return the resource title, URL and a short description."
        ),
    }

    result_data = None
    last_message = ""

    try:
        yield (
            "TinyFish is starting browser automation...\n\n"
            "Please wait while it searches the web.",
            None,
            gr.update(interactive=False),
        )

        with http.post(
            TINYFISH_ENDPOINT,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(15, 240),
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):

                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")

                if not line.startswith("data:"):
                    continue

                raw_event = line[5:].strip()

                if not raw_event or raw_event == "[DONE]":
                    continue

                try:
                    event = json.loads(raw_event)
                except Exception:
                    continue

                if not isinstance(event, dict):
                    continue

                event_type = str(
                    event.get("type") or event.get("event") or ""
                ).upper()

                if event_type in ("STARTED", "PROGRESS"):
                    message = str(
                        event.get("purpose")
                        or event.get("message")
                        or event.get("status")
                        or "TinyFish is working..."
                    )

                    if message != last_message:
                        last_message = message

                        yield (
                            f"TinyFish: {message}",
                            None,
                            gr.update(interactive=False),
                        )

                elif event_type == "COMPLETE":
                    result_data = (
                        event.get("resultJson")
                        or event.get("result")
                        or event.get("data")
                        or event
                    )
                    break

                elif event_type in ("FAILED", "ERROR"):
                    error_message = (
                        event.get("message")
                        or event.get("error")
                        or "TinyFish automation failed."
                    )

                    yield (
                        f"TinyFish error: {error_message}",
                        None,
                        gr.update(interactive=False),
                    )
                    return

        if result_data is None:
            raise RuntimeError(
                "TinyFish did not return a COMPLETE event. Please try again."
            )

        resources = extract_resources(result_data)

        if resources:
            research_cache.set(cache_key, result_data)

        yield (
            format_research_result(resources, query),
            result_data,
            gr.update(interactive=bool(resources)),
        )

    except requests.exceptions.Timeout:
        yield (
            "TinyFish timed out. Please try again.",
            None,
            gr.update(interactive=False),
        )

    except Exception as e:
        yield (
            f"Research error: {str(e)}",
            None,
            gr.update(interactive=False),
        )


# =========================================================
# 9. 7-DAY LEARNING ROADMAP
# =========================================================

def create_roadmap(research_data):

    if not research_data:
        yield "Please complete TinyFish research first."
        return

    resources = extract_resources(research_data)

    if not resources:
        yield "No usable resources were found. Run another research query."
        return

    resource_text = "\n".join(
        f"- {item['name']}: {item['url']} — {item['description']}"
        for item in resources
    )

    cache_key = make_key(resource_text)
    cached = roadmap_cache.get(cache_key)

    if cached:
        yield cached
        return

    prompt = f"""
You are MindEase AI, a friendly learning-roadmap creator.

Create a practical 7-day beginner learning roadmap using these
resources discovered through TinyFish browser automation.

RESOURCES:
{resource_text}

Requirements:
- Make a 7-day plan.
- Keep each day's workload realistic.
- Mention which resource to use each day.
- Include a small practical exercise.
- Include one achievable daily goal.
- Start from beginner level.
- Use clear headings and simple language.
- End with a small final project idea.
- Do not invent resource URLs.
"""

    yield "Creating your personalized 7-day roadmap..."

    try:
        final_answer = ""

        for answer in gemini_stream(prompt, max_tokens=1800):
            final_answer = answer
            yield answer

        if final_answer.strip():
            roadmap_cache.set(cache_key, final_answer)

    except Exception as e:
        yield f"Roadmap error: {str(e)}"


# =========================================================
# 10. CALM SPACE
# =========================================================

def calm_space():
    return """
## Welcome to Calm Space

Take a short break. You don't need to solve everything right now.

### Try this simple breathing exercise

1. Sit comfortably.
2. Breathe in slowly for **4 seconds**.
3. Breathe out gently for **6 seconds**.
4. Repeat this cycle 5 times.

### A small reminder

You are allowed to pause.

Progress does not mean doing everything at once.
Sometimes, taking a break is also a productive step.

**Less thinking. More living.**
"""


# =========================================================
# 11. CALENDAR & REMINDER BACKEND
# =========================================================

DB_PATH = Path(__file__).with_name("mindease.db")
CREDENTIALS_PATH = Path(__file__).with_name("credentials.json")
TOKEN_PATH = Path(__file__).with_name("token.json")

IST = ZoneInfo("Asia/Kolkata")
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


@contextmanager
def db():
    """Open the database, commit on success and always close it."""
    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                due_at TEXT NOT NULL,
                notify_at TEXT NOT NULL,
                notes TEXT,
                calendar_event_id TEXT,
                created_at TEXT NOT NULL
            )
        """)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_reminder_rows():

    with db() as conn:
        rows = conn.execute("""
            SELECT title, due_at, notify_at, notes, calendar_event_id
            FROM reminders
            ORDER BY due_at ASC
        """).fetchall()

    result = []

    for title, due, notify, notes, event_id in rows:
        due_dt = datetime.fromisoformat(due)
        notify_dt = datetime.fromisoformat(notify)

        result.append([
            title,
            due_dt.strftime("%d-%m-%Y %I:%M %p"),
            notify_dt.strftime("%d-%m-%Y %I:%M %p"),
            notes or "",
            "Synced" if event_id else "Local only",
        ])

    return result


def render_reminder_bridge():
    """Hidden HTML element the browser script reads to schedule alarms."""

    now = datetime.now(IST).isoformat()

    with db() as conn:
        rows = conn.execute("""
            SELECT id, title, due_at, notify_at
            FROM reminders
            WHERE notify_at > ?
            ORDER BY notify_at ASC
        """, (now,)).fetchall()

    reminders = [
        {
            "id": row[0],
            "title": row[1],
            "due_at": row[2],
            "notify_at": row[3],
        }
        for row in rows
    ]

    payload = html.escape(json.dumps(reminders), quote=True)

    return (
        f'<div class="mindease-reminder-bridge" '
        f'data-reminders="{payload}"></div>'
    )


def refresh_reminders():
    return render_reminder_bridge(), get_reminder_rows()


def get_google_calendar_service():
    # Imported here so the app works even if these packages are not installed.
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    credentials = None

    if TOKEN_PATH.exists():
        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_PATH), GOOGLE_SCOPES
        )

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    if not credentials or not credentials.valid:

        if not CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                "credentials.json is missing. "
                "Download your Google OAuth Desktop credentials first."
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_PATH), GOOGLE_SCOPES
        )

        credentials = flow.run_local_server(port=0)

    TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")

    return build("calendar", "v3", credentials=credentials)


def sync_google_calendar(title, due_at, notes, lead_minutes):

    service = get_google_calendar_service()

    start_time = datetime.fromisoformat(due_at)
    end_time = start_time + timedelta(minutes=30)

    event = {
        "summary": title,
        "description": notes or "Created using MindEase AI.",
        "start": {
            "dateTime": start_time.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": int(lead_minutes)}
            ],
        },
    }

    created_event = service.events().insert(
        calendarId="primary",
        body=event,
    ).execute()

    return created_event.get("id")


def create_reminder(title, date, time_value, lead_minutes, notes,
                    sync_google=False):

    try:
        title = (title or "").strip()
        date = (date or "").strip()
        time_value = (time_value or "").strip()

        if not title:
            raise ValueError("Please enter a reminder title.")

        try:
            due_at = datetime.strptime(
                f"{date} {time_value}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=IST)
        except ValueError:
            raise ValueError(
                "Use the date format YYYY-MM-DD and the time format HH:MM "
                "(24-hour), for example 2026-11-05 and 18:30."
            )

        lead_minutes = int(lead_minutes)
        notify_at = due_at - timedelta(minutes=lead_minutes)

        now = datetime.now(IST)

        if due_at <= now:
            raise ValueError("Please select a future date and time.")

        if notify_at <= now:
            raise ValueError(
                "The reminder notification time has already passed. "
                "Choose a later time or a shorter reminder interval."
            )

        event_id = None
        calendar_message = ""

        if sync_google:
            try:
                event_id = sync_google_calendar(
                    title, due_at.isoformat(), notes, lead_minutes
                )
                calendar_message = (
                    "\n\nGoogle Calendar event created successfully."
                )
            except Exception as calendar_error:
                calendar_message = (
                    "\n\nReminder saved locally, but Google Calendar sync "
                    f"failed: {calendar_error}"
                )

        with db() as conn:
            conn.execute("""
                INSERT INTO reminders
                (title, due_at, notify_at, notes, calendar_event_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                title,
                due_at.isoformat(),
                notify_at.isoformat(),
                notes or "",
                event_id,
                datetime.now(IST).isoformat(),
            ))

        status = (
            f"Reminder saved: **{title}**\n\n"
            f"Due: {due_at.strftime('%d %B %Y, %I:%M %p')}\n\n"
            f"Notification: {notify_at.strftime('%d %B %Y, %I:%M %p')}"
            f"{calendar_message}"
        )

        return status, render_reminder_bridge(), get_reminder_rows()

    except Exception as error:
        return (
            f"**Reminder error:** {error}",
            render_reminder_bridge(),
            get_reminder_rows(),
        )


def save_local(title, date, time_value, lead, notes):
    return create_reminder(title, date, time_value, lead, notes, False)


def save_and_sync(title, date, time_value, lead, notes):
    return create_reminder(title, date, time_value, lead, notes, True)


# =========================================================
# 12. FRONTEND DESIGN
# =========================================================

CUSTOM_CSS = """

/* ===== GLOBAL ===== */

html {
    scroll-behavior: smooth;
    scroll-padding-top: 85px;
}

html, body {
    width: 100%;
    margin: 0;
    overflow-x: hidden;
}

body {
    background: #ffffff !important;
    color: #292929 !important;
}

.gradio-container,
.gradio-container > .main,
.gradio-container .main,
.app {
    max-width: 100% !important;
    width: 100% !important;
}

.gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    background: #ffffff !important;
    font-family: 'Inter', Arial, sans-serif !important;
}

/* Gives Gradio inputs the same side spacing as the HTML sections */
.content-pad {
    padding: 0 6% 40px 6% !important;
    box-sizing: border-box;
}

/* ===== UTILITY BAR ===== */

.utility-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
    padding: 9px 5%;
    background: #f4f4f3;
    border-bottom: 1px solid #e6e6e4;
    font-size: 12px;
    color: #555555;
}

.utility-links { display: flex; flex-wrap: wrap; gap: 20px; }
.utility-bar a { color: #555555; text-decoration: none; }
.utility-bar a:hover { text-decoration: underline; }

/* ===== HEADER ===== */

.site-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 25px;
    padding: 20px 5%;
    background: #ffffff;
    border-bottom: 1px solid #e8e8e6;
}

.brand { display: flex; align-items: center; gap: 13px; min-width: 230px; }

.brand-mark {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #d8d8d5;
    border-radius: 50%;
    font-size: 23px;
    color: #222222;
    background: linear-gradient(145deg, #ffffff, #eeeeec);
}

.brand-name { font-size: 23px; font-weight: 800; letter-spacing: -0.7px; color: #222222; }
.brand-tagline { font-size: 11px; color: #777777; margin-top: 3px; }

.main-nav {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 22px;
}

.main-nav a {
    color: #333333;
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    white-space: nowrap;
}

.main-nav a:hover { color: #777777; text-decoration: underline; }

/* ===== HERO ===== */

.hero-section {
    position: relative;
    padding: clamp(45px, 8vw, 100px) 8%;
    background: linear-gradient(110deg, #ffffff 0%, #f8f8f7 50%, #eeeeec 100%);
    border-bottom: 1px solid #e8e8e6;
    overflow: hidden;
}

.hero-content { max-width: 850px; position: relative; z-index: 2; }

.hero-eyebrow {
    display: inline-block;
    padding: 7px 13px;
    border: 1px solid #dededb;
    border-radius: 30px;
    background: #ffffff;
    color: #555555;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.hero-section h1 {
    margin: 22px 0 15px;
    font-size: clamp(36px, 5.5vw, 67px);
    line-height: 1.08;
    letter-spacing: -2.4px;
    font-weight: 800;
    color: #222222;
}

.hero-section p {
    max-width: 620px;
    color: #666666;
    font-size: clamp(15px, 1.6vw, 18px);
    line-height: 1.8;
}

.hero-note { margin-top: 20px; color: #777777; font-size: 13px; }

/* ===== SECTION HEADINGS ===== */

.home-section {
    padding: 48px 6%;
    border-bottom: 1px solid #eeeeec;
    scroll-margin-top: 80px;
}

.home-section.tight { padding-bottom: 12px; border-bottom: none; }

.section-heading { margin-bottom: 25px; }

.section-heading .eyebrow {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.7px;
    font-weight: 700;
    color: #888888;
}

.section-heading h2 {
    margin: 8px 0;
    font-size: clamp(24px, 3vw, 34px);
    letter-spacing: -0.8px;
    font-weight: 750;
    color: #242424;
}

.section-heading p { max-width: 700px; color: #777777; font-size: 14px; line-height: 1.7; }

/* ===== ANNOUNCEMENTS ===== */

.announcement-panel {
    background: linear-gradient(135deg, #ffffff, #f5f5f4);
    border: 1px solid #e5e5e2;
    border-radius: 15px;
    padding: 22px;
}

.announcement-item {
    display: flex;
    align-items: flex-start;
    gap: 15px;
    padding: 15px 0;
    border-bottom: 1px solid #e8e8e6;
}

.announcement-item:last-child { border-bottom: none; }

.announcement-number {
    min-width: 35px;
    height: 35px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #ddddda;
    border-radius: 50%;
    background: #ffffff;
    font-size: 12px;
    font-weight: 700;
}

.announcement-item strong { display: block; font-size: 14px; color: #333333; }
.announcement-item span { display: block; margin-top: 5px; font-size: 12px; line-height: 1.6; color: #777777; }

/* ===== INFO CARDS ===== */

.info-card {
    padding: 23px;
    border: 1px solid #e5e5e3;
    border-radius: 15px;
    background: linear-gradient(145deg, #ffffff 0%, #fdfdfc 55%, #f4f4f2 100%);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.info-card:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(0, 0, 0, 0.055); }
.info-card h3 { margin: 0 0 9px; font-size: 17px; color: #292929; }
.info-card p { margin: 0; font-size: 13px; line-height: 1.75; color: #777777; }

/* ===== INPUTS AND BUTTONS ===== */

.gradio-group, .gradio-box, .panel {
    border-color: #e5e5e3 !important;
    border-radius: 14px !important;
}

textarea, input, select {
    background: #ffffff !important;
    color: #333333 !important;
    border: 1px solid #dededc !important;
    border-radius: 10px !important;
}

textarea:focus, input:focus {
    border-color: #999999 !important;
    box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.045) !important;
}

button { border-radius: 9px !important; }

button.primary {
    background: linear-gradient(135deg, #ffffff, #e9e9e7) !important;
    border: 1px solid #d5d5d2 !important;
    color: #252525 !important;
    font-weight: 700 !important;
    min-height: 46px !important;
    box-shadow: 0 3px 9px rgba(0, 0, 0, 0.035);
}

button.primary:hover { background: linear-gradient(135deg, #f2f2f1, #e1e1df) !important; }
button.primary[disabled] { opacity: 0.5 !important; cursor: not-allowed !important; }

button.secondary {
    background: #ffffff !important;
    color: #444444 !important;
    border: 1px solid #dededc !important;
}

button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 3px solid #555555 !important;
    outline-offset: 2px !important;
}

.wrap-row { flex-wrap: wrap !important; gap: 10px !important; }
.wrap-row > * { flex: 1 1 200px !important; min-width: 0 !important; }

#enable-notifications {
    padding: 12px 20px;
    background: linear-gradient(135deg, #ffffff, #e9e9e7);
    color: #222222;
    border: 1px solid #cccccc;
    border-radius: 9px;
    font-weight: 700;
    cursor: pointer;
}

/* ===== TABLES ===== */

#task-table, #reminder-table,
#task-table .table-wrap, #reminder-table .table-wrap {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}

#task-table th, #reminder-table th { background: #eeeeec !important; color: #333333 !important; }
#task-table td, #reminder-table td { background: #ffffff !important; }

/* ===== MARKDOWN ===== */

.prose, .markdown {
    max-width: none !important;
    color: #333333 !important;
    line-height: 1.75 !important;
    overflow-wrap: anywhere;
}

.prose h1, .prose h2, .prose h3,
.markdown h1, .markdown h2, .markdown h3 { color: #292929 !important; }

/* ===== FOOTER ===== */

.site-footer {
    padding: 35px 6%;
    background: linear-gradient(135deg, #f5f5f4, #ffffff);
    border-top: 1px solid #e5e5e3;
}

.footer-grid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 30px; }
.site-footer h3 { font-size: 17px; color: #292929; }
.site-footer p, .site-footer a { font-size: 12px; line-height: 1.8; color: #777777; text-decoration: none; }
.site-footer a:hover { text-decoration: underline; }

.footer-bottom {
    margin-top: 25px;
    padding-top: 15px;
    border-top: 1px solid #e4e4e2;
    text-align: center;
    font-size: 11px;
    color: #888888;
}

footer.svelte-footer, .gradio-container > footer { display: none !important; }

/* ===== RESPONSIVE ===== */

@media (max-width: 900px) {
    .site-header { flex-direction: column; align-items: flex-start; }
    .main-nav { justify-content: flex-start; gap: 15px; }
    .footer-grid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 600px) {
    .utility-bar { padding: 9px 4%; flex-direction: column; align-items: flex-start; }
    .site-header { padding: 15px 4%; }
    .brand { min-width: 0; }
    .brand-name { font-size: 20px; }
    .main-nav { gap: 12px; }
    .main-nav a { font-size: 11px; }
    .hero-section { padding: 48px 6%; }
    .hero-section h1 { letter-spacing: -1.4px; }
    .home-section { padding: 35px 5%; }
    .content-pad { padding: 0 5% 32px 5% !important; }
    .footer-grid { grid-template-columns: 1fr; }
    .content-pad button, #enable-notifications { width: 100% !important; }
    textarea, input { font-size: 16px !important; }
}

@media (prefers-reduced-motion: reduce) {
    * { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}

.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .markdown p,
.gradio-container .markdown li,
.gradio-container .prose p,
.gradio-container .prose li {
    color: #1f2937 !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container strong {
    color: #111827 !important;
}

"""


# =========================================================
# 13. HEAD (viewport, fonts, reminder alarm script)
# =========================================================

HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#ffffff">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script>
(() => {
    const scheduled = new Set();
    let audioContext = null;

    function playAlarm() {
        try {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }

            audioContext.resume();

            const oscillator = audioContext.createOscillator();
            const gain = audioContext.createGain();

            oscillator.connect(gain);
            gain.connect(audioContext.destination);

            oscillator.frequency.value = 880;
            gain.gain.value = 0.12;

            oscillator.start();

            setTimeout(() => { oscillator.stop(); }, 700);

        } catch (error) {
            console.log("Alarm sound unavailable:", error);
        }
    }

    function scheduleReminders() {
        document.querySelectorAll(".mindease-reminder-bridge").forEach(element => {

            let reminders = [];

            try {
                reminders = JSON.parse(element.dataset.reminders || "[]");
            } catch (error) {
                return;
            }

            reminders.forEach(reminder => {

                const id = String(reminder.id);

                if (scheduled.has(id)) return;

                const delay = new Date(reminder.notify_at).getTime() - Date.now();

                if (delay <= 0 || delay > 2147483647) return;

                scheduled.add(id);

                setTimeout(() => {

                    if ("Notification" in window && Notification.permission === "granted") {
                        new Notification("MindEase AI Reminder", { body: reminder.title });
                    }

                    playAlarm();

                }, delay);
            });
        });
    }

    document.addEventListener("click", async event => {

        if (event.target.closest("#enable-notifications")) {

            if (!("Notification" in window)) {
                alert("This browser does not support notifications.");
                return;
            }

            const permission = await Notification.requestPermission();

            if (permission === "granted") {
                alert("Browser notifications enabled!");
            } else {
                alert("Please allow notifications in your browser settings.");
            }

            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.resume();
            } catch (error) {
                console.log(error);
            }
        }
    });

    const observer = new MutationObserver(scheduleReminders);

    function start() {
        observer.observe(document.documentElement, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ["data-reminders"]
        });

        scheduleReminders();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
</script>
"""

THEME = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="gray",
    neutral_hue="slate",
)

UI_ARGS = {"theme": THEME, "css": CUSTOM_CSS, "head": HEAD}

blocks_kwargs = {
    "title": "MindEase AI | Less thinking. More living.",
    "fill_width": True,
}

if not LAUNCH_ACCEPTS_UI:
    blocks_kwargs.update(UI_ARGS)

blocks_kwargs = supported_kwargs(gr.Blocks.__init__, **blocks_kwargs)


def dataframe_kwargs(headers, datatype, label, elem_id, value=None,
                     interactive=True, fixed_columns=False):
    """Builds Dataframe arguments that work on every Gradio version."""

    params = inspect.signature(gr.Dataframe.__init__).parameters

    kwargs = {
        "headers": headers,
        "datatype": datatype,
        "value": value if value is not None else [],
        "interactive": interactive,
        "wrap": True,
        "label": label,
        "elem_id": elem_id,
    }

    if fixed_columns:
        kwargs["row_count"] = (0, "dynamic")

        if "column_count" in params:
            kwargs["column_count"] = (len(headers), "fixed")
        elif "col_count" in params:
            kwargs["col_count"] = (len(headers), "fixed")

    return supported_kwargs(gr.Dataframe.__init__, **kwargs)


# =========================================================
# 14. GRADIO APPLICATION
# =========================================================

with gr.Blocks(**blocks_kwargs) as demo:

    research_state = gr.State(None)

    # ---------------------------------------------
    # UTILITY BAR
    # ---------------------------------------------

    gr.HTML("""
    <div class="utility-bar">
        <div>MindEase AI | Your Personal Life Organizer</div>

        <div class="utility-links">
            <a href="#home">Home</a>
            <a href="#tasks">My Tasks</a>
            <a href="#research">Explore</a>
            <a href="#calendar">Calendar</a>
            <a href="#calm">Calm Space</a>
        </div>
    </div>
    """)

    # ---------------------------------------------
    # MAIN HEADER
    # ---------------------------------------------

    gr.HTML("""
    <header class="site-header">
        <div class="brand">
            <div class="brand-mark">M</div>

            <div>
                <div class="brand-name">MindEase AI</div>
                <div class="brand-tagline">Less thinking. More living.</div>
            </div>
        </div>

        <nav class="main-nav">
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#organizer">Organizer</a>
            <a href="#tasks">My Tasks</a>
            <a href="#research">Discover</a>
            <a href="#calendar">Calendar</a>
            <a href="#calm">Calm Space</a>
        </nav>
    </header>
    """)

    # ---------------------------------------------
    # HERO SECTION
    # ---------------------------------------------

    gr.HTML("""
    <section class="hero-section" id="home">
        <div class="hero-content">

            <span class="hero-eyebrow">YOUR PERSONAL LIFE ORGANIZER</span>

            <h1>Make space for<br>what matters.</h1>

            <p>
                Organize your thoughts, discover useful opportunities,
                manage your daily tasks and make time for yourself.
                Everything you need, in one calm space.
            </p>

            <div class="hero-note">Less thinking. More living.</div>

        </div>
    </section>
    """)

    # ---------------------------------------------
    # IMPORTANT UPDATES
    # ---------------------------------------------

    with gr.Column(elem_id="announcements"):

        gr.HTML("""
        <section class="home-section">
            <div class="section-heading">
                <div class="eyebrow">STAY ORGANIZED</div>
                <h2>Important Updates</h2>
                <p>
                    Keep your priorities, upcoming activities and
                    personal goals in one place.
                </p>
            </div>

            <div class="announcement-panel">

                <div class="announcement-item">
                    <div class="announcement-number">01</div>
                    <div>
                        <strong>Your daily priorities</strong>
                        <span>
                            Organize important tasks and decide
                            what needs your attention first.
                        </span>
                    </div>
                </div>

                <div class="announcement-item">
                    <div class="announcement-number">02</div>
                    <div>
                        <strong>Discover new opportunities</strong>
                        <span>
                            Explore learning resources, courses,
                            events and useful websites with TinyFish.
                        </span>
                    </div>
                </div>

                <div class="announcement-item">
                    <div class="announcement-number">03</div>
                    <div>
                        <strong>Make time for yourself</strong>
                        <span>
                            Take a pause, breathe and return
                            to your tasks with a clearer mind.
                        </span>
                    </div>
                </div>

            </div>
        </section>
        """)

    # ---------------------------------------------
    # ABOUT SECTION
    # ---------------------------------------------

    gr.HTML("""
    <section class="home-section" id="about">

        <div class="section-heading">
            <div class="eyebrow">ABOUT MINDEASE AI</div>
            <h2>A simpler way to organize your life</h2>
            <p>
                MindEase AI combines artificial intelligence
                and browser automation to help turn scattered
                thoughts into meaningful actions.
            </p>
        </div>

        <div class="info-card">
            <h3>One space. Multiple possibilities.</h3>
            <p>
                Use Gemini to organize thoughts and create
                actionable plans. Use TinyFish to explore the web
                and discover useful resources. Use your task planner
                to keep track of what matters.
            </p>
        </div>

    </section>
    """)

    # ---------------------------------------------
    # MIND ORGANIZER
    # ---------------------------------------------

    with gr.Column(elem_id="organizer"):

        gr.HTML("""
        <section class="home-section tight">
            <div class="section-heading">
                <div class="eyebrow">YOUR PERSONAL SPACE</div>
                <h2>Mind Organizer</h2>
                <p>
                    Empty your mind. Write everything down.
                    Let AI help you organize your thoughts.
                </p>
            </div>
        </section>
        """)

        with gr.Column(elem_classes=["content-pad"]):

            thoughts_input = gr.Textbox(
                label="What's on your mind?",
                placeholder=(
                    "Example: I have an exam next week, "
                    "two assignments, and I also want to learn Python..."
                ),
                lines=6,
                max_lines=16,
            )

            with gr.Row(elem_classes=["wrap-row"]):

                organize_btn = gr.Button(
                    "Organize My Thoughts",
                    variant="primary",
                    scale=3,
                )

                clear_btn = gr.Button("Clear", scale=1, min_width=90)

            organized_output = gr.Markdown()

    # ---------------------------------------------
    # TASK PLANNER
    # ---------------------------------------------

    with gr.Column(elem_id="tasks"):

        gr.HTML("""
        <section class="home-section tight">
            <div class="section-heading">
                <div class="eyebrow">PLAN YOUR DAY</div>
                <h2>Actionable Task Planner</h2>
                <p>
                    Convert your thoughts into manageable tasks.
                    Understand priorities and track your progress.
                </p>
            </div>
        </section>
        """)

        with gr.Column(elem_classes=["content-pad"]):

            planner_btn = gr.Button("Build My Task Plan", variant="primary")

            planner_status = gr.Markdown()

            task_table = gr.Dataframe(
                **dataframe_kwargs(
                    headers=[
                        "Task",
                        "Priority",
                        "Estimated Time",
                        "Deadline",
                        "Why this priority?",
                        "Completed",
                    ],
                    datatype=["str", "str", "str", "str", "str", "bool"],
                    label="Your task tracker",
                    elem_id="task-table",
                    interactive=True,
                    fixed_columns=True,
                )
            )

            gr.Markdown(
                "*Tip: tick the Completed checkbox when you finish a task. "
                "You can also edit task details.*"
            )

    # ---------------------------------------------
    # DISCOVER OPPORTUNITIES
    # ---------------------------------------------

    with gr.Column(elem_id="research"):

        gr.HTML("""
        <section class="home-section tight">
            <div class="section-heading">
                <div class="eyebrow">EXPLORE THE WEB</div>
                <h2>Discover Opportunities</h2>
                <p>
                    Find learning resources, courses, tutorials
                    and useful opportunities through TinyFish
                    browser automation.
                </p>
            </div>
        </section>
        """)

        with gr.Column(elem_classes=["content-pad"]):

            research_query = gr.Textbox(
                label="What do you want to research?",
                placeholder="Example: Best free Python courses for beginners",
                lines=2,
            )

            research_btn = gr.Button(
                "Start TinyFish Research",
                variant="primary",
            )

            research_output = gr.Markdown()

            gr.Markdown("### Convert research into a 7-day roadmap")

            roadmap_btn = gr.Button(
                "Generate My 7-Day Roadmap",
                variant="primary",
                interactive=False,
            )

            roadmap_output = gr.Markdown()

    # ---------------------------------------------
    # CALENDAR AND REMINDERS
    # ---------------------------------------------

    with gr.Column(elem_id="calendar"):

        gr.HTML("""
        <section class="home-section tight">
            <div class="section-heading">
                <div class="eyebrow">PLAN AHEAD</div>
                <h2>Calendar &amp; Reminders</h2>
                <p>
                    Schedule your tasks, set notifications and
                    keep important deadlines in sight.
                </p>
            </div>

            <div class="info-card">
                <h3>Your personal reminder centre</h3>
                <p>
                    Save reminders locally or create events
                    directly in your Google Calendar.
                </p>
            </div>
        </section>
        """)

        with gr.Column(elem_classes=["content-pad"]):

            gr.HTML("""
            <div style="padding: 12px 0;">
                <button id="enable-notifications" type="button">
                    Enable Browser Notifications &amp; Alarm
                </button>
                <p style="font-size: 12px; color: #777; margin-top: 8px;">
                    Allow browser notifications when prompted.
                    Keep this tab open so the alarm can ring.
                </p>
            </div>
            """)

            gr.Markdown("### Create a new reminder")

            reminder_title = gr.Textbox(
                label="Reminder title",
                placeholder="Example: Complete Digital Electronics assignment",
            )

            with gr.Row(elem_classes=["wrap-row"]):

                reminder_date = gr.Textbox(
                    label="Date (YYYY-MM-DD)",
                    placeholder="2026-11-05",
                )

                reminder_time = gr.Textbox(
                    label="Time (24-hour HH:MM)",
                    placeholder="18:30",
                )

            reminder_lead = gr.Dropdown(
                choices=["0", "5", "15", "30"],
                value="15",
                label="Notify me before (minutes)",
            )

            reminder_notes = gr.Textbox(
                label="Notes (optional)",
                placeholder="Add extra details...",
                lines=2,
            )

            with gr.Row(elem_classes=["wrap-row"]):

                save_reminder_btn = gr.Button("Save Reminder", variant="primary")

                google_sync_btn = gr.Button(
                    "Save & Sync Google Calendar",
                    variant="secondary",
                )

            reminder_status = gr.Markdown()

            reminder_bridge = gr.HTML(value="")

            gr.Markdown("### Your saved reminders")

            reminder_table = gr.Dataframe(
                **dataframe_kwargs(
                    headers=[
                        "Reminder",
                        "Due date and time",
                        "Notification time",
                        "Notes",
                        "Calendar status",
                    ],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Reminder list",
                    elem_id="reminder-table",
                    interactive=False,
                )
            )

    # ---------------------------------------------
    # CALM SPACE
    # ---------------------------------------------

    with gr.Column(elem_id="calm"):

        gr.HTML("""
        <section class="home-section tight">
            <div class="section-heading">
                <div class="eyebrow">TAKE A MOMENT</div>
                <h2>Calm Space</h2>
                <p>
                    Sometimes the best next step is taking a break.
                    Give yourself a moment to pause and reset.
                </p>
            </div>
        </section>
        """)

        with gr.Column(elem_classes=["content-pad"]):

            calm_btn = gr.Button("Give Me a Calm Break", variant="primary")

            calm_output = gr.Markdown()

    # ---------------------------------------------
    # FOOTER
    # ---------------------------------------------

    gr.HTML("""
    <footer class="site-footer">

        <div class="footer-grid">

            <div>
                <h3>MindEase AI</h3>
                <p>
                    Your personal life organizer.
                    Organize thoughts, discover opportunities
                    and make space for what matters.
                </p>
                <p><strong>Less thinking. More living.</strong></p>
            </div>

            <div>
                <h3>Explore</h3>
                <p><a href="#organizer">Mind Organizer</a></p>
                <p><a href="#tasks">Task Planner</a></p>
                <p><a href="#research">Discover</a></p>
            </div>

            <div>
                <h3>Your Space</h3>
                <p><a href="#calendar">Calendar</a></p>
                <p><a href="#calm">Calm Space</a></p>
                <p><a href="#home">Back to top</a></p>
            </div>

        </div>

        <div class="footer-bottom">
            MindEase AI | Less thinking. More living.
        </div>

    </footer>
    """)

    # =============================================
    # EVENT WIRING (all inside the same Blocks)
    # =============================================

    organize_btn.click(
        fn=organize_thoughts,
        inputs=thoughts_input,
        outputs=organized_output,
    )

    thoughts_input.submit(
        fn=organize_thoughts,
        inputs=thoughts_input,
        outputs=organized_output,
    )

    clear_btn.click(
        fn=lambda: ("", "", [], ""),
        inputs=None,
        outputs=[
            thoughts_input,
            organized_output,
            task_table,
            planner_status,
        ],
        queue=False,
    )

    planner_btn.click(
        fn=build_task_plan,
        inputs=thoughts_input,
        outputs=[task_table, planner_status],
    )

    research_btn.click(
        fn=research_online,
        inputs=research_query,
        outputs=[research_output, research_state, roadmap_btn],
    )

    roadmap_btn.click(
        fn=create_roadmap,
        inputs=research_state,
        outputs=roadmap_output,
    )

    reminder_inputs = [
        reminder_title,
        reminder_date,
        reminder_time,
        reminder_lead,
        reminder_notes,
    ]

    reminder_outputs = [reminder_status, reminder_bridge, reminder_table]

    save_reminder_btn.click(
        fn=save_local,
        inputs=reminder_inputs,
        outputs=reminder_outputs,
    )

    google_sync_btn.click(
        fn=save_and_sync,
        inputs=reminder_inputs,
        outputs=reminder_outputs,
    )

    calm_btn.click(
        fn=calm_space,
        inputs=None,
        outputs=calm_output,
        queue=False,
    )

    # Load reminders fresh every time the page opens or is refreshed.
    demo.load(
        fn=refresh_reminders,
        inputs=None,
        outputs=[reminder_bridge, reminder_table],
        queue=False,
    )


# =========================================================
# 15. LAUNCH (outside Blocks, once)
# =========================================================

if __name__ == "__main__":

    print("----------------------------------")
    print("Starting MindEase AI")
    print("Gradio  :", gr.__version__)
    print("Gemini  :", "Loaded" if GEMINI_API_KEY else "MISSING")
    print("TinyFish:", "Loaded" if TINYFISH_API_KEY else "MISSING")
    print("----------------------------------")

    demo.queue(max_size=20, default_concurrency_limit=4)

    launch_kwargs = {
        "server_name": HOST,
        "server_port": PORT,
        "show_error": True,
    }

    if LAUNCH_ACCEPTS_UI:
        launch_kwargs.update(UI_ARGS)

    demo.launch(**supported_kwargs(demo.launch, **launch_kwargs))