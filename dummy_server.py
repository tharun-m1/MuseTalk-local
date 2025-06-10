import asyncio
import os
import traceback
import uuid
from aiohttp import web
import aiohttp_cors
import json
import time
import io
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from pytz import timezone
import base64
from concurrent.futures import ThreadPoolExecutor

# Global constants and state
AVATAR_ID = "pradeep"
BBOX_SHIFT = 12
BATCH_SIZE = 9
FPS = 12

pcs = set()
gpu_executor = ThreadPoolExecutor(max_workers=3)
generate_sema = asyncio.Semaphore(3)

# Job Queue
MAX_ACTIVE_JOBS = 2
job_queue = asyncio.Queue()
active_jobs = 0
total_requests_received = 0

# WebSocket connections
active_websockets = []

# Frame caching
CACHE_DIR = "cached_frames"
MAX_CHUNKS_TO_CACHE = 20  # Set to 17 since chunk 4 is skipped
current_chunk_count = 0
cached_chunks = {}  # {chunk_id: [frames]}
cached_audio_data = {}  # {chunk_id: audio_bytes}
current_playback_index = 1  # Start from chunk_001

def extract_uuid_from_audio(audio_bytes):
    """Extract UUID from first 16 bytes of the audio data."""
    if len(audio_bytes) < 16:
        raise ValueError("Audio data too short to contain UUID")
    uuid_bytes = audio_bytes[:16]
    hex_digits = "0123456789abcdef"
    uuid_chars = []
    for i in range(16):
        uuid_chars.append(hex_digits[uuid_bytes[i] >> 4])
        uuid_chars.append(hex_digits[uuid_bytes[i] & 0x0f])
    uuid_str = ''.join(uuid_chars)
    uuid_str = f"{uuid_str[:8]}-{uuid_str[8:12]}-{uuid_str[12:16]}-{uuid_str[16:20]}-{uuid_str[20:]}"
    return uuid_str

def package_single_frame_for_webrtc(frame, uuid_str, chunk_id, is_final, status="success", error_message=""):
    payload = bytearray()
    uuid_bytes = bytes.fromhex(uuid_str.replace('-', ''))
    payload.extend(uuid_bytes)
    status_bytes = status.encode('utf-8')[:10].ljust(10, b'\0')
    payload.extend(status_bytes)
    error_bytes = error_message.encode('utf-8')[:100].ljust(100, b'\0')
    payload.extend(error_bytes)
    payload.extend(chunk_id.to_bytes(4, byteorder='big'))
    payload.extend((1 if is_final else 0).to_bytes(1, byteorder='big'))
    target_size = (640, 360)
    resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_bytes = buf.tobytes()
    payload.extend(len(frame_bytes).to_bytes(4, byteorder='big'))
    payload.extend(frame_bytes)
    return bytes(payload)

async def send_frame_via_webrtc_async(channel, frame, uuid_str, chunk_id, is_final, status="success", error_message=""):
    try:
        if channel.readyState != "open":
            print(f"⚠️ Cannot send frame {chunk_id} - channel state: {channel.readyState}")
            return False
        response = package_single_frame_for_webrtc(frame, uuid_str, chunk_id, is_final, status, error_message)
        channel.send(response)
        print(f"✅ Sent frame {chunk_id}/{uuid_str[:8]} (final: {is_final}) - {len(response)} bytes")
        return True
    except Exception as e:
        print(f"❌ Error sending frame {chunk_id}: {e}")
        return False

def load_chunk_frames_and_audio(chunk_id):
    chunk_dir = os.path.join(CACHE_DIR, f"chunk_{chunk_id:03d}")
    if not os.path.exists(chunk_dir):
        return None, None
    frame_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith("frame_") and f.endswith(".jpg")])
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(chunk_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
    audio_path = os.path.join(chunk_dir, "audio.wav")
    audio_data = None
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            audio_data = f.read()
    return frames if frames else None, audio_data

async def stream_cached_chunk_response(channel, uuid_str):
    global cached_chunks, cached_audio_data, current_playback_index
    if len(cached_chunks) < MAX_CHUNKS_TO_CACHE:
        print("❌ Not enough cached chunks to serve.")
        return False
    chunk_id = current_playback_index
    # Skip chunk_id 4 if it is not present in cached_chunks
    while chunk_id == 4 or chunk_id not in cached_chunks:
        chunk_id += 1
        if chunk_id > MAX_CHUNKS_TO_CACHE:
            chunk_id = 1
        # Prevent infinite loop if no valid chunks
        if chunk_id == current_playback_index:
            print("❌ No valid cached chunks to serve.")
            return False
    frames = cached_chunks[chunk_id]
    audio_data = cached_audio_data.get(chunk_id)
    print(f"🚀 Using cached chunk {chunk_id} as response ({len(frames)} frames) - sequential playback")
    for idx, frame in enumerate(frames):
        is_final = (idx == len(frames) - 1)
        success = await send_frame_via_webrtc_async(channel, frame, uuid_str, idx, is_final)
        if not success:
            print(f"⚠️ Failed to send cached frame {idx}, continuing...")
        if idx < len(frames) - 1:
            await asyncio.sleep(0.01)
    print(f"✅ Successfully streamed {len(frames)} cached frames via WebRTC")
    if active_websockets and audio_data:
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        await send_audio_via_websocket(audio_b64, uuid_str)
        print(f"✅ Sent cached audio for chunk {chunk_id} via WebSocket")
    # Only increment within the valid chunk range (1 to MAX_CHUNKS_TO_CACHE)
    current_playback_index = chunk_id + 1
    if current_playback_index > MAX_CHUNKS_TO_CACHE:
        current_playback_index = 1
    print(f"📍 Next playback will use chunk {current_playback_index}")
    return True

async def process_audio_and_respond(msg, channel):
    global active_jobs
    try:
        ist = timezone('Asia/Kolkata')
        now = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"[{now}] Processing audio request with streaming (cache-only mode)...")
        audio_bytes = bytes(msg)
        if len(audio_bytes) < 16:
            raise ValueError(f"Audio data too small: {len(audio_bytes)} bytes, expected at least 16 bytes for UUID")
        try:
            uuid_str = extract_uuid_from_audio(audio_bytes)
            print(f"Extracted UUID: {uuid_str}")
        except Exception as e:
            print(f"Error extracting UUID: {e}")
            uuid_str = str(uuid.uuid4())
        # Only serve from cache
        if len(cached_chunks) >= MAX_CHUNKS_TO_CACHE:
            print(f"🚀 Using cached response - ignoring incoming audio, no voice AI needed")
            await stream_cached_chunk_response(channel, uuid_str)
            return
        # Try to load cache from disk if not already loaded
        if not cached_chunks:
            print("🔄 Loading existing cached chunks from disk...")
            await load_all_cached_chunks()
            if len(cached_chunks) >= MAX_CHUNKS_TO_CACHE:
                await stream_cached_chunk_response(channel, uuid_str)
                return
        # If not enough cache, return error
        print("❌ Not enough cached chunks to serve. Please pre-populate the cache.")
        if channel and channel.readyState == "open":
            error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            await send_frame_via_webrtc_async(
                channel, error_frame, uuid_str, 0, True,
                status="failed", error_message="No cached data available"
            )
    except Exception as e:
        print(f"❌ Error in streaming processing job: {e}")
        traceback.print_exc()
        if channel and channel.readyState == "open":
            error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            error_uuid = uuid_str if 'uuid_str' in locals() else str(uuid.uuid4())
            await send_frame_via_webrtc_async(
                channel, error_frame, error_uuid, 0, True,
                status="failed", error_message=str(e)[:99]
            )
    finally:
        active_jobs -= 1
        await maybe_start_next_job()

async def load_all_cached_chunks():
    global cached_chunks, cached_audio_data
    print("🔄 Loading cached chunks at startup...")
    cached_chunks.clear()
    cached_audio_data.clear()
    for i in range(1, MAX_CHUNKS_TO_CACHE + 2):  # Start from chunk_001
        if i == 4:
            continue
        existing_frames, existing_audio = load_chunk_frames_and_audio(i)
        if existing_frames and existing_audio:
            cached_chunks[i] = existing_frames
            cached_audio_data[i] = existing_audio
    print(f"✅ Cached chunks loaded: {sorted(cached_chunks.keys())}")
    print(f"✅ Total cached chunks: {len(cached_chunks)}")

async def init_models(app):
    """Stub for model initialization (no-op in cache-only mode)."""
    print("Cache-only mode: No models loaded.")
    await load_all_cached_chunks()

async def offer(request):
    try:
        params = await request.json()
        print(f"Received WebRTC offer from client")
        from aiortc import RTCPeerConnection, RTCSessionDescription
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        pcs.add(pc)
        @pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label != "lipsyncdatachannel":
                print(f"Unexpected data channel: {channel.label}")
                return
            @channel.on("message")
            def handle(msg):
                print(f"Received WebRTC message of size {len(bytes(msg))} bytes")
                asyncio.create_task(queue_job(msg, channel))
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("WebRTC connection established, sending answer to client")
        return web.Response(content_type="application/json", text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }))
    except Exception as e:
        print(f"Error handling WebRTC offer: {e}")
        traceback.print_exc()
        return web.Response(status=500, text=f"Server error: {str(e)}")

async def queue_job(msg, channel):
    global total_requests_received
    total_requests_received += 1
    print(f"📥 Request #{total_requests_received} received. Queue size: {job_queue.qsize()}")
    await job_queue.put((msg, channel))
    await maybe_start_next_job()

async def maybe_start_next_job():
    global active_jobs
    if active_jobs >= MAX_ACTIVE_JOBS or job_queue.empty():
        return
    msg, channel = await job_queue.get()
    active_jobs += 1
    asyncio.create_task(process_audio_and_respond(msg, channel))

async def send_audio_via_websocket(audio_b64, uuid_str):
    ws_sent = False
    for ws in active_websockets:
        try:
            audio_msg = {
                "event": "playAudio",
                "media": {
                    "track": "inbound",
                    "payload": audio_b64
                },
                "streamId": uuid_str
            }
            await ws.send_json(audio_msg)
            ws_sent = True
            print(f"Sent audio with UUID {uuid_str} via WebSocket")
        except Exception as e:
            print(f"Error sending to WebSocket: {e}")
    if not ws_sent:
        print("⚠️ No active WebSockets to send audio")

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    client_id = request.query.get('agent', 'unknown')
    print(f"New WebSocket connection from client: {client_id}")
    active_websockets.append(ws)
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    print(f"Received WebSocket message: {data.get('event', 'unknown')}")
                    if data.get('event') == 'media':
                        pass
                    elif data.get('event') == 'ping':
                        await ws.send_json({"event": "pong"})
                except json.JSONDecodeError:
                    print(f"Received invalid JSON: {msg.data[:100]}...")
            elif msg.type == web.WSMsgType.ERROR:
                print(f"WebSocket error: {ws.exception()}")
    finally:
        if ws in active_websockets:
            active_websockets.remove(ws)
        print(f"WebSocket connection closed, client: {client_id}")
    return ws

async def ping(request):
    return web.Response(text="pong")

def setup_application():
    app = web.Application()
    app.on_startup.append(init_models)
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["POST", "GET", "OPTIONS", "DELETE", "PUT"]
        )
    })
    app.router.add_route('GET', '/ws/', websocket_handler)
    resource = cors.add(app.router.add_resource("/offer"))
    cors.add(resource.add_route("POST", offer))
    ping_resource = cors.add(app.router.add_resource("/ping"))
    cors.add(ping_resource.add_route("GET", ping))
    return app

def cleanup():
    for pc in pcs:
        pc.close()
    for file in Path("tmp").glob("audio_*.wav"):
        try:
            file.unlink()
        except:
            pass

if __name__ == "__main__":
    try:
        app = setup_application()
        print(f"🚀 Starting Dummy Server (cache-only mode) - will serve only cached chunks")
        web.run_app(app, host="0.0.0.0", port=8080)
    finally:
        cleanup()