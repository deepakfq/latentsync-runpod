"""
RunPod Serverless Handler — LatentSync 1.6

Input:
  {
    "video_url": "https://...mp4" OR "video_b64": "...",
    "audio_url": "https://...mp3" OR "audio_b64": "...",
    "inference_steps": 20,        // 20-50, higher = better quality
    "guidance_scale": 1.5,         // 1.0-3.0, higher = better sync
    "seed": 1247
  }

Output:
  {
    "video_b64": "base64 encoded mp4",
    "duration": 194.5,
    "size_kb": 18234,
    "processing_time": 45.2
  }
"""
import runpod
import os, sys, subprocess, base64, requests, uuid, time, traceback

LATENTSYNC_DIR = "/opt/LatentSync"
CACHE = "/tmp/latentsync_cache"
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, LATENTSYNC_DIR)


def download_or_decode(src, ext):
    """Download URL or decode base64 to local file."""
    path = f"{CACHE}/{uuid.uuid4().hex[:8]}.{ext}"
    if isinstance(src, str) and src.startswith("http"):
        r = requests.get(src, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    else:
        with open(path, "wb") as f:
            f.write(base64.b64decode(src))
    return path


def get_duration(path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", path
        ]).decode().strip()
        return float(out)
    except:
        return 0


def loop_video(video_path, target_duration):
    """Loop video to match target duration (preserving original quality)."""
    video_dur = get_duration(video_path)
    if abs(video_dur - target_duration) < 0.5:
        return video_path

    out = f"{CACHE}/looped_{uuid.uuid4().hex[:8]}.mp4"
    if video_dur >= target_duration:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-t", str(target_duration),
            "-c", "copy", out
        ], capture_output=True)
        return out

    loops = int(target_duration / video_dur) + 1
    concat_file = f"{CACHE}/concat_{uuid.uuid4().hex[:8]}.txt"
    with open(concat_file, "w") as f:
        for _ in range(loops):
            f.write(f"file '{video_path}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-t", str(target_duration), "-c", "copy", out
    ], capture_output=True)

    try: os.remove(concat_file)
    except: pass
    return out


def run_latentsync(video_path, audio_path, inference_steps=20, guidance_scale=1.5, seed=1247):
    """Run LatentSync inference."""
    output = f"{CACHE}/output_{uuid.uuid4().hex[:8]}.mp4"

    # LatentSync inference.py signature
    cmd = [
        "python3", "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path", f"{LATENTSYNC_DIR}/checkpoints/latentsync_unet.pt",
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output,
        "--seed", str(seed),
    ]

    print(f"Running LatentSync: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=LATENTSYNC_DIR)

    if result.returncode != 0:
        err = result.stderr[-1000:] if result.stderr else "no stderr"
        out = result.stdout[-500:] if result.stdout else "no stdout"
        raise RuntimeError(f"LatentSync failed (code={result.returncode}):\nSTDERR: {err}\nSTDOUT: {out}")

    if not os.path.exists(output):
        raise RuntimeError(f"LatentSync no output. stdout: {result.stdout[-500:]}")

    return output


def merge_clean_audio(video_path, audio_path):
    """Remux original audio with synced video."""
    output = f"{CACHE}/final_{uuid.uuid4().hex[:8]}.mp4"
    r = subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
        output
    ], capture_output=True)
    return output if os.path.exists(output) else video_path


def handler(event):
    job_input = event.get("input", {})
    job_id = event.get("id", "unknown")
    t0 = time.time()

    print(f"[{job_id}] Starting LatentSync...", flush=True)

    try:
        # 1. Download/decode inputs
        video_src = job_input.get("video_url") or job_input.get("video_b64")
        audio_src = job_input.get("audio_url") or job_input.get("audio_b64")

        if not video_src: return {"error": "Missing video_url or video_b64"}
        if not audio_src: return {"error": "Missing audio_url or audio_b64"}

        print(f"[{job_id}] Downloading inputs...", flush=True)
        video_path = download_or_decode(video_src, "mp4")
        audio_path = download_or_decode(audio_src, "mp3")

        audio_dur = get_duration(audio_path)
        video_dur = get_duration(video_path)
        print(f"[{job_id}] Audio: {audio_dur:.1f}s, Video: {video_dur:.1f}s", flush=True)

        # 2. Loop video to match audio
        if abs(video_dur - audio_dur) > 1:
            print(f"[{job_id}] Looping video to match audio...", flush=True)
            video_path = loop_video(video_path, audio_dur)

        # 3. LatentSync params
        inference_steps = int(job_input.get("inference_steps", 20))
        guidance_scale = float(job_input.get("guidance_scale", 1.5))
        seed = int(job_input.get("seed", 1247))

        # 4. Run LatentSync
        print(f"[{job_id}] Running LatentSync (steps={inference_steps}, guidance={guidance_scale})...", flush=True)
        t1 = time.time()
        synced = run_latentsync(video_path, audio_path, inference_steps, guidance_scale, seed)
        print(f"[{job_id}] LatentSync done in {time.time()-t1:.1f}s", flush=True)

        # 5. Merge clean audio
        final = merge_clean_audio(synced, audio_path)

        # 6. Return
        with open(final, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        size_kb = os.path.getsize(final) // 1024
        dur = get_duration(final)
        total = time.time() - t0

        print(f"[{job_id}] Done: {size_kb}KB, {dur:.1f}s, total {total:.1f}s", flush=True)

        return {
            "video_b64": video_b64,
            "duration": dur,
            "size_kb": size_kb,
            "processing_time": total,
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{job_id}] ERROR: {e}\n{tb}", flush=True)
        return {"error": str(e), "traceback": tb}


runpod.serverless.start({"handler": handler})
