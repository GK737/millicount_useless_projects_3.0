# MILLICOUNT 🐛

AI-powered millipede segment and leg counter, built with Streamlit. Supports
three AI vision providers, switchable from the sidebar: **Google Gemini**,
**OpenAI**, and **Anthropic Claude**.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (You only strictly need the SDK for the provider(s) you plan to use —
   `google-genai`, `openai`, and/or `anthropic` — but installing all three
   lets you switch freely in the app.)

2. Provide an API key for at least one provider, using **any one** of these:
   - Rename `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in the key(s) you have.
   - Set an environment variable before launching, e.g. `export OPENAI_API_KEY=your-key-here`
   - Just paste it into the sidebar once the app is running (kept only for your session).

3. Run the app:
   ```bash
   streamlit run app.py
   ```

Pick a provider in the sidebar, upload an image or use your webcam, then click **Analyze**.

## Providers

| Provider | Default model | Key source | Get a key |
|---|---|---|---|
| Google Gemini | `gemini-3.7-flash` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | https://aistudio.google.com/apikey |
| OpenAI | `gpt-4o` | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic Claude | `claude-sonnet-5` | `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |

The model field in the sidebar is editable — swap in any newer vision-capable
model your key has access to without touching the code.

## Adding another provider

Provider logic lives in `providers.py`, isolated from the Streamlit UI in `app.py`:

1. Write a function `analyze_yourprovider(image_bytes, api_key, model) -> MillipedeAnalysis`
   that calls the provider's API and returns a `MillipedeAnalysis`.
2. Register it in the `PROVIDERS` dict at the bottom of `providers.py` with a
   label, default model, and the secrets/env var key to look for.

The app picks it up automatically — no changes needed in `app.py`.

## What was fixed/added from the original draft

- Extracted all provider-specific logic into `providers.py` so adding a new
  AI backend doesn't touch the UI code.
- Generalized API key resolution (session → `secrets.toml` → env vars) to
  work per-provider instead of being hardcoded to Gemini.
- `st.secrets.get(...)` is wrapped safely — it raises if no `secrets.toml`
  exists at all, which used to crash the app before the sidebar even rendered.
- OpenAI uses the SDK's built-in structured-output parsing
  (`beta.chat.completions.parse` with a Pydantic model).
- Claude uses a forced tool call (`tool_choice`) to get reliable structured
  JSON back, since the Messages API doesn't have a native `response_schema`.
- Added `requirements.txt` and a `secrets.toml.example` template covering
  all three providers.
