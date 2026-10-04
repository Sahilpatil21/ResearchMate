# 🚀 ResearchMate — Deployment Guide

This guide covers deployment instructions for **Render**, **Railway**, **Docker**, and **Streamlit Community Cloud**.

---

## 📋 Required Environment Variables

Configure the following secrets / environment variables in your deployment dashboard (e.g. Render Environment, Railway Variables, or `.env`):

| Variable | Description | Example / Required Value |
| :--- | :--- | :--- |
| `APP_ENV` | Application environment mode | `production` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIzaSy...` (Required) |
| `GEMINI_MODEL` | Gemini LLM model | `gemini-2.5-flash` |
| `MONGODB_URI` | MongoDB Atlas cluster connection string | `mongodb+srv://<user>:<password>@cluster0.mongodb.net/ResearchMate?retryWrites=true&w=majority` |
| `MONGODB_DB_NAME` | MongoDB database name | `ResearchMate` |
| `CLOUDINARY_CLOUD_NAME` | Cloudinary Cloud Name | `<your_cloud_name>` |
| `CLOUDINARY_API_KEY` | Cloudinary API Key | `<your_api_key>` |
| `CLOUDINARY_API_SECRET` | Cloudinary API Secret | `<your_api_secret>` |
| `EMBEDDING_MODEL_NAME` | Embedding transformer model | `all-MiniLM-L6-v2` |
| `RERANKER_MODEL_NAME` | Cross-encoder reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

> [!IMPORTANT]
> **MongoDB Atlas Network Access**: Remember to go to **MongoDB Atlas $\rightarrow$ Network Access $\rightarrow$ Add IP Address** and select **Allow Access from Anywhere (`0.0.0.0/0`)** so your cloud provider can connect.

---

## Option 1: Render / Railway / Cloud Web Service (FastAPI Full-Stack) — Recommended

ResearchMate includes a FastAPI server that directly serves the web frontend and all REST endpoints.

1. Connect your GitHub repository to **Render** or **Railway**.
2. Set service type to **Web Service** with **Python 3.11**.
3. **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start Command (CHANGED from Streamlit to FastAPI)**:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port $PORT
   ```
   *(or `python server.py`)*
5. In the **Environment** tab, add the environment variables listed in the table above.
6. Deploy!

---

## Option 2: Containerized Deployment (Docker)

To build and run as a Docker container on any cloud (Render Docker, AWS ECS, GCP Cloud Run, DigitalOcean, etc.):

```bash
# 1. Build the Docker image
docker build -t researchmate:latest .

# 2. Run the container (maps to FastAPI port 8000)
docker run -d --name researchmate -p 8000:8000 --env-file .env researchmate:latest
```

---

## Option 3: Streamlit Community Cloud (Legacy Streamlit Interface)

If you specifically want to run the standalone Streamlit interface (app.py):

1. Push your repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a **New app**.
3. Point the main file path to `app.py`.
4. Add secrets in **Advanced settings $\rightarrow$ Secrets** in TOML format:
   ```toml
   APP_ENV = "production"
   GEMINI_API_KEY = "your_key"
   GEMINI_MODEL = "gemini-2.5-flash"
   MONGODB_URI = "mongodb+srv://..."
   MONGODB_DB_NAME = "ResearchMate"
   CLOUDINARY_CLOUD_NAME = "your_cloud_name"
   CLOUDINARY_API_KEY = "your_api_key"
   CLOUDINARY_API_SECRET = "your_api_secret"
   ```
