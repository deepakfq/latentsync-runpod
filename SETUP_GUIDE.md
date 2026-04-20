# RunPod Serverless Setup — Step by Step

## Option A: Build Docker Locally (M1 Max)
**Warning: Docker image is ~15GB. Needs 20GB disk + 1 hour build time + Docker Desktop installed.**

```bash
cd /tmp/latentsync-runpod

# Create Docker Hub account if needed: https://hub.docker.com/signup
# Login
docker login

# Build for amd64 (RunPod GPUs are x86, not ARM)
docker buildx build --platform linux/amd64 \
  -t YOUR_DOCKERHUB_USER/latentsync-runpod:latest \
  --push .
```

## Option B: Build on RunPod directly (easier!)
Use RunPod's "Dev Pod" to build:
1. Launch a RunPod pod with any GPU
2. SSH in: `ssh root@<pod-ip>`
3. Clone your code
4. `docker build -t username/latentsync-runpod . && docker push`

## Option C: Use pre-built public image (fastest)
We'll publish to Docker Hub as `freaquer/latentsync-runpod:latest` so you can skip build entirely.

---

## RunPod Serverless Endpoint Creation

Go to: **https://console.runpod.io/serverless/new-endpoint?flow=custom**

Fill in form:

### Basic
- **Endpoint Name:** `latentsync-lipsync`
- **Template:** "Custom" (already selected)

### Container
- **Container Image:** `YOUR_USER/latentsync-runpod:latest`
- **Container Registry Credentials:** (Docker Hub — none needed if public)
- **Docker Command:** (leave empty, uses CMD from Dockerfile)
- **Container Disk:** 25 GB

### GPU
Select in priority order:
- ✅ L40S (48GB) — recommended, ~$0.00076/sec
- ✅ A100 80GB — backup
- ✅ RTX 4090 (24GB) — budget option
- ✅ RTX 6000 Ada (48GB)

### Workers
- **Max Workers:** 1
- **Idle Timeout:** 5 sec
- **GPUs per Worker:** 1
- **Flash Boot:** ✅ ON
- **Network Storage:** (optional, helps model loading)

### Advanced
- **Request Timeout:** 900 sec (15 min)

### Environment Variables (optional)
None needed unless you want R2 credentials for direct upload.

---

## Test the Endpoint

After creation you get:
- Endpoint ID: `xyz123abc` 
- Your API Key: from https://console.runpod.io/user/settings → API Keys

```bash
# Test with sample anchor + audio
curl -X POST https://api.runpod.ai/v2/ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "https://api.lawyerdigest.in/anchor/anchor_full.mp4",
      "audio_url": "https://api.lawyerdigest.in/bulletins/latest_hi.mp3",
      "inference_steps": 20,
      "guidance_scale": 1.5
    }
  }' > result.json

# Extract video
python3 -c "
import json, base64
d = json.load(open('result.json'))
with open('synced.mp4', 'wb') as f:
    f.write(base64.b64decode(d['output']['video_b64']))
print(f'Got {d[\"output\"][\"size_kb\"]}KB video, {d[\"output\"][\"processing_time\"]:.1f}s processing')
"
```

---

## Cost Estimate
- **Per bulletin (3 min audio):**
  - Cold start: ~30-60s
  - Inference: ~60-120s on L40S
  - Total GPU time: ~90-180s
  - Cost: ~$0.07-0.14 per bulletin
  
- **With warm worker:** ~$0.05 per bulletin
- **Monthly (hourly bulletins = 720/month):** ~$50-100/month

Much cheaper than always-on GPU pod ($400/mo).
