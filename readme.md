# Text Summarization with LangChain and Groq

This project summarizes text with LangChain and LLMs hosted on [Groq](https://console.groq.com). It has two parts:

- **`app.py`**: a Streamlit web app that summarizes a **YouTube video** (from its transcript) or a **website** from its URL.
- **`code.ipynb`**: a notebook that walks through the main LangChain summarization techniques on a speech and on a PDF (`apjspeech.pdf`).

## Features

- Summarize any public web page or YouTube video (`youtube.com` and `youtu.be` links).
- Picks the summarization strategy automatically:
  - **stuff**: short content is sent to the model in a single request.
  - **map-reduce**: long content is split into chunks. Each chunk is summarized, then the chunk summaries are combined into one.
- Handles Groq's free-tier rate limits by retrying automatically and trimming very long pages.
- Uses the app's own Groq API key from `.env` or Streamlit secrets, kept on the server so visitors can't see it. Visitors can also paste their own key into the sidebar.

## What the notebook covers

| Technique | Description |
|---|---|
| Basic prompt | Summarize a speech with system and human messages |
| Prompt templates | Summarize, then translate the summary into another language (French, Punjabi, Tamil, …) |
| `stuff` chain | Summarize a whole PDF in one request |
| `map_reduce` chain | Summarize chunks one by one, then combine them into a final summary with a title and bullet points |
| `refine` chain | Build the summary step by step, chunk by chunk |

## Tech stack

- [LangChain](https://python.langchain.com) (`langchain-core`, `langchain-classic`, `langchain-community`, `langchain-groq`)
- [Groq](https://groq.com) LLM API (model: `openai/gpt-oss-20b`)
- [Streamlit](https://streamlit.io) for the UI
- `youtube-transcript-api` and `unstructured` for loading content

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/aditsharma14/Text-Summarization.git
cd Text-Summarization
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Get a free key from the [Groq console](https://console.groq.com/keys). Then create a `.env` file in the project root:

```env
groq_api=your_groq_api_key_here
```

`.env` is listed in `.gitignore`, so your key is never committed.

## Usage

### Streamlit app

```bash
streamlit run app.py
```

1. Open the URL Streamlit prints (usually http://localhost:8501).
2. If `.env` contains a key, leave the sidebar box blank. Otherwise, paste your key there.
3. Paste a YouTube or website URL and click **Summarize the Content from YT or Website**.

### Notebook

Install the Jupyter kernel first (`pip install ipykernel`). Then open `code.ipynb` in Jupyter or VS Code, select the `venv` kernel, and run the cells from top to bottom.

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**. Choose this repository, the `main` branch, and `app.py` as the main file.
3. Under **Advanced settings**, choose Python 3.13 and paste this into **Secrets**:

   ```toml
   groq_api = "your_groq_api_key_here"
   ```

4. Click **Deploy**.

The key stays on the server and is never shown in the app. If you leave the secret out, each visitor has to enter their own Groq key.

## Project structure

```
Text-Summarization/
├── app.py             # Streamlit app: summarize YouTube videos and websites
├── code.ipynb         # Notebook: stuff / map-reduce / refine summarization
├── apjspeech.pdf      # Sample PDF used in the notebook
├── requirements.txt   # Python dependencies
├── .env               # Your API keys (not committed)
└── readme.md
```

## Notes and limitations

- **YouTube**: the video must have captions or a transcript available.
- **Rate limits**: Groq's free tier allows about 8,000 tokens per minute. For long pages, only the first 4 sections (about 24,000 characters) are summarized, and the app says so when this happens. To summarize more, change `MAX_CHUNKS` in `app.py`.
- **Model**: Groq retired `gemma-7b-it`, so the app uses `openai/gpt-oss-20b`. You can change `GROQ_MODEL` in `app.py` to any model your key can access.
- Some websites block scrapers or load their content with JavaScript. Those pages may return little or no text.
