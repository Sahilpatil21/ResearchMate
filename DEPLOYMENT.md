# 🚀 ResearchMate — Deployment Guide

This guide covers deployment instructions for **Streamlit Community Cloud**, **Render**, **Hugging Face Spaces**, and **Docker**.

---

## 📋 Required Environment Variables

Configure the following secrets / environment variables in your deployment platform's settings:

| Variable | Description | Example / Required Value |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API key for Grounded RAG & Intelligence | `AIzaSy...` (Required) |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `MONGODB_URI` | MongoDB Atlas cluster connection string | `mongodb+srv://<user>:<password>@cluster0.mongodb.net/ResearchMate?retryWrites=true&w=majority` |
| `MONGODB_DB_NAME` | MongoDB database name | `ResearchMate` |
| `CLOUDINARY_CLOUD_NAME` | Cloudinary Cloud Name (from Cloudinary dashboard) | `<your_cloud_name>` |
| `CLOUDINARY_API_KEY` | Cloudinary API Key | `<your_api_key>` |
| `CLOUDINARY_API_SECRET` | Cloudinary API Secret | `<your_api_secret>` |
| `APP_ENV` | Application environment | `production` |

---

## Option 1: Streamlit Community Cloud (Recommended — Free & 1-Click)

1. Push your repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **"New app"**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Advanced settings** $\rightarrow$ **Secrets**, and paste your configuration:
```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-2.5-flash"
MONGODB_URI = "mongodb+srv://..."
MONGODB_DB_NAME = "ResearchMate"
CLOUDINARY_CLOUD_NAME = "your_cloud_name"
CLOUDINARY_API_KEY = "your_api_key"
CLOUDINARY_API_SECRET = "your_api_secret"
APP_ENV = "production"
```
5. Click **Deploy!**

---

## Option 2: Render / Railway (Web Service)

1. Connect your GitHub repository on [Render](https://render.com) or [Railway](https://railway.app).
2. Choose **Web Service** with **Python Environment**.
3. Set **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
4. Set **Start Command**:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Add the environment variables from the table above in the **Environment** tab.

---

## Option 3: Containerized Deployment (Docker)

To build and run locally or on AWS / GCP / Azure:

```bash
# 1. Build the Docker image
docker build -t researchmate:latest .

# 2. Run the container with your .env file
docker run -d --name researchmate -p 8501:8501 --env-file .env researchmate:latest
```

---

## Option 4: Hugging Face Spaces (Docker or Streamlit SDK)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Streamlit** SDK.
3. Push your repository files.
4. Add your secrets in **Settings** $\rightarrow$ **Variables and secrets**.
