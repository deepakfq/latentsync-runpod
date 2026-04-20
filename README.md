# LatentSync RunPod Serverless

GitHub Actions builds and pushes a Docker image to Docker Hub whenever this repo is updated.

## Files
- `Dockerfile` — LatentSync 1.6 + all deps + model (~5GB)
- `handler.py` — RunPod serverless handler
- `.github/workflows/build.yml` — auto-builds on push
- `SETUP_GUIDE.md` — RunPod endpoint config

## Setup (one-time)

### 1. Add GitHub secrets
Go to: `Settings → Secrets and variables → Actions → New repository secret`

Add two secrets:
- `DOCKERHUB_USERNAME` = `freaquer`
- `DOCKERHUB_TOKEN` = your Docker Hub access token (from hub.docker.com/settings/security)

### 2. Trigger build
- Push to `main` branch → auto-builds
- Or: Actions tab → "Build and Push LatentSync Docker Image" → Run workflow

### 3. Create RunPod endpoint
After build succeeds, use image: `freaquer/latentsync-runpod:latest`

See `SETUP_GUIDE.md` for RunPod endpoint configuration.

## Usage

```python
import requests, base64

r = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "video_url": "https://api.lawyerdigest.in/anchor/anchor_full.mp4",
            "audio_url": "https://api.lawyerdigest.in/bulletins/latest_hi.mp3",
            "inference_steps": 20,
            "guidance_scale": 1.5
        }
    },
    timeout=600
)
out = r.json()["output"]
with open("synced.mp4", "wb") as f:
    f.write(base64.b64decode(out["video_b64"]))
```
