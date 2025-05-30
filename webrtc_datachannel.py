import asyncio
import os
import traceback
import uuid
from aiohttp import web
import aiohttp_cors

from aiortc import RTCPeerConnection, RTCSessionDescription
import json
import time
import io
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from pytz import timezone
from pydub import AudioSegment
# import pynvml
import base64
from concurrent.futures import ThreadPoolExecutor

from musetalk.utils.blending import get_image_blending
from musetalk.utils.utils import load_all_model, datagen
from scripts.realtime_inference import Avatar
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

# Global constants and state
AVATAR_ID = "shreyan"
BBOX_SHIFT = 12
BATCH_SIZE = 15
FPS = 15
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

audio_processor = vae = unet = pe = timesteps = avatar_model = whisper = None
pcs = set()

# GPU setup
# pynvml.nvmlInit()
# gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_executor = ThreadPoolExecutor(max_workers=3)
generate_sema = asyncio.Semaphore(3)

# Job Queue
MAX_ACTIVE_JOBS = 1
job_queue = asyncio.Queue()
active_jobs = 0
total_requests_received = 0

# WebSocket connections
active_websockets = []

# def log_gpu(prefix=""):
#     """Log GPU utilization and memory usage."""
#     util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
#     mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
#     print(f"{prefix} | GPU util {util.gpu}% | mem used {mem.used//1024**2}MiB / {mem.total//1024**2}MiB")

def extract_uuid_from_audio(audio_bytes):
    """Extract UUID from first 16 bytes of the audio data."""
    if len(audio_bytes) < 16:
        raise ValueError("Audio data too short to contain UUID")
    
    uuid_bytes = audio_bytes[:16]
    
    # Convert UUID bytes to string representation
    hex_digits = "0123456789abcdef"
    uuid_chars = []
    
    for i in range(16):
        uuid_chars.append(hex_digits[uuid_bytes[i] >> 4])
        uuid_chars.append(hex_digits[uuid_bytes[i] & 0x0f])
    
    # Format UUID with hyphens (8-4-4-4-12 format)
    uuid_str = ''.join(uuid_chars)
    uuid_str = f"{uuid_str[:8]}-{uuid_str[8:12]}-{uuid_str[12:16]}-{uuid_str[16:20]}-{uuid_str[20:]}"
    
    return uuid_str

def log_message_details(msg, prefix="Received"):
    """
    Print detailed information about a binary message for debugging.
    
    Args:
        msg: Binary message data
        prefix: Prefix for log messages (default: "Received")
    """
    bytes_data = bytes(msg)
    size = len(bytes_data)
    
    print(f"{prefix} message: {size} bytes total")
    
    # Show first 16 bytes (potential UUID)
    if size >= 16:
        uuid_bytes = bytes_data[:16]
        uuid_hex = ''.join(f'{b:02x}' for b in uuid_bytes)
        uuid_str = f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:]}"
        print(f"{prefix} UUID: {uuid_str}")
    
    # Show first few bytes as hex
    max_bytes_to_show = min(64, size)
    hex_preview = ' '.join(f'{bytes_data[i]:02x}' for i in range(max_bytes_to_show))
    print(f"{prefix} data (first {max_bytes_to_show} bytes): {hex_preview}")

def debug_response_structure(response):
    """
    Verify the structure of a response to ensure it matches the expected format.
    
    Args:
        response: Binary response data
    
    Returns:
        bool: True if the structure appears valid, False otherwise
    """
    try:
        if len(response) < 130:
            print(f"❌ Response too small: {len(response)} bytes, expected at least 130 bytes")
            return False
        
        # Extract UUID (first 16 bytes)
        uuid_bytes = response[:16]
        uuid_hex = ''.join(f'{b:02x}' for b in uuid_bytes)
        uuid_str = f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:]}"
        
        # Extract status (next 10 bytes)
        status_bytes = response[16:26]
        status = status_bytes.split(b'\0')[0].decode('utf-8', errors='ignore')
        
        # Extract error message (next 100 bytes)
        error_bytes = response[26:126]
        error_msg = error_bytes.split(b'\0')[0].decode('utf-8', errors='ignore')
        
        # Extract frame count (next 4 bytes)
        frame_count_bytes = response[126:130]
        frame_count = int.from_bytes(frame_count_bytes, byteorder='big')
        
        # Log structure details
        print(f"Response structure check:")
        print(f"- UUID: {uuid_str}")
        print(f"- Status: '{status}'")
        print(f"- Error: '{error_msg}' (empty is normal for success)")
        print(f"- Frame count: {frame_count}")
        print(f"- Data size after header: {len(response) - 130} bytes")
        
        # Check if data size seems reasonable for frame count
        if frame_count > 0:
            avg_frame_size = (len(response) - 130) / frame_count
            print(f"- Average bytes per frame: {avg_frame_size:.2f}")
            
            if avg_frame_size < 100 or avg_frame_size > 1000000:
                print(f"⚠️ Unusual average frame size: {avg_frame_size:.2f} bytes")
        
        return True
    except Exception as e:
        print(f"❌ Error checking response structure: {e}")
        traceback.print_exc()
        return False

def package_frames_for_frontend(frames, uuid_str, status="success", error_message=""):
    """
    Package frames in the format expected by the frontend:
    - 16 bytes: UUID
    - 10 bytes: Status string (padded with null bytes)
    - 100 bytes: Error message (padded with null bytes)
    - 4 bytes: Frame count
    - Frame data (variable length)
    """
    payload = bytearray()
    
    # 1. Add UUID (16 bytes)
    uuid_bytes = bytes.fromhex(uuid_str.replace('-', ''))
    payload.extend(uuid_bytes)
    
    # 2. Add status (10 bytes, padded with nulls)
    status_bytes = status.encode('utf-8')[:10].ljust(10, b'\0')
    payload.extend(status_bytes)
    
    # 3. Add error message (100 bytes, padded with nulls)
    error_bytes = error_message.encode('utf-8')[:100].ljust(100, b'\0')
    payload.extend(error_bytes)
    
    # 4. Add frame count (4 bytes)
    payload.extend(len(frames).to_bytes(4, byteorder='big'))
    
    # 5. Add frame data (no size prefix needed)
    for frame in frames:
        # Resize frame to reduce data size
        target_size = (640, 360)  # (width, height) - adjust as needed
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        # Convert frame to JPEG
        _, buf = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        frame_bytes = buf.tobytes()
        # payload.extend(frame_bytes)
        # Add frame size (4 bytes)
        payload.extend(len(frame_bytes).to_bytes(4, byteorder='big'))
    
        # Add frame data
        payload.extend(frame_bytes)
    
    return bytes(payload)

async def init_models(app):
    """Initialize ML models for lip-sync generation."""
    global audio_processor, vae, unet, pe, timesteps, avatar_model, whisper
    print("Loading models...")
    vae, unet, pe = load_all_model()
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    pe = pe.half().to(MODEL_DEVICE)
    vae.vae = vae.vae.half().to(MODEL_DEVICE)
    unet.model = unet.model.half().to(MODEL_DEVICE)
    timesteps = torch.tensor([0], device=MODEL_DEVICE)
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=MODEL_DEVICE, dtype=unet.model.dtype).eval()
    whisper.requires_grad_(False)
    avatar_model = Avatar(
        avatar_id=AVATAR_ID,
        video_path="./data/video/dummy.mp4",
        bbox_shift=BBOX_SHIFT,
        batch_size=BATCH_SIZE,
        preparation=True,
        vae=vae
    )
    print("Models loaded successfully!")

async def offer(request):
    """Handle WebRTC offer from client."""
    try:
        params = await request.json()
        print(f"Received WebRTC offer from client")
        
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
                log_message_details(msg, prefix="WebRTC received")
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
    """Queue audio processing job."""
    global total_requests_received
    total_requests_received += 1
    print(f"📥 Request #{total_requests_received} received. Queue size: {job_queue.qsize()}")
    await job_queue.put((msg, channel))
    await maybe_start_next_job()

async def maybe_start_next_job():
    """Start processing the next job if resources are available."""
    global active_jobs
    if active_jobs >= MAX_ACTIVE_JOBS or job_queue.empty():
        return
    msg, channel = await job_queue.get()
    active_jobs += 1
    asyncio.create_task(process_audio_and_respond(msg, channel))

async def process_audio_and_respond(msg, channel):
    """
    Process audio data from WebRTC and generate lip-sync frames.
    This is the core of the audio-to-frame pipeline.
    
    Args:
        msg: Binary data received from the WebRTC data channel
        channel: WebRTC data channel for sending the response
    """
    global active_jobs
    try:
        # Get timestamp for logging
        ist = timezone('Asia/Kolkata')
        now = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"[{now}] Processing audio request...")

        # Extract audio bytes
        audio_bytes = bytes(msg)
        log_message_details(audio_bytes, prefix="Processing")
        
        # Log data size for debugging
        print(f"Received audio data size: {len(audio_bytes)} bytes")
        
        # Check if data contains UUID (should be at least 16 bytes)
        if len(audio_bytes) < 16:
            raise ValueError(f"Audio data too small: {len(audio_bytes)} bytes, expected at least 16 bytes for UUID")
        
        # Extract UUID (first 16 bytes) from audio data
        try:
            uuid_str = extract_uuid_from_audio(audio_bytes)
            print(f"Extracted UUID: {uuid_str}")
            # Remove UUID prefix from audio for processing
            audio_data = audio_bytes[16:]
        except Exception as e:
            print(f"Error extracting UUID: {e}")
            uuid_str = str(uuid.uuid4())  # Generate a new one if extraction fails
            audio_data = audio_bytes  # Use all data
        
        print(f"Audio data size after UUID extraction: {len(audio_data)} bytes")
        
        # Create buffer and convert to proper format for processing
        try:
            mp3_buf = io.BytesIO(audio_data)
            audio_seg = AudioSegment.from_file(mp3_buf, format="mp3", parameters=["-err_detect", "ignore_err"])
            
            # Get audio info for logging
            print(f"Audio properties: {audio_seg.channels} channels, {audio_seg.frame_rate}Hz, {audio_seg.duration_seconds:.2f}s")
            
            # Convert to WAV for processing
            wav_buf = io.BytesIO()
            audio_seg.export(wav_buf, format="wav", parameters=["-ar", "44100", "-ac", "1"])
            wav_bytes = wav_buf.getvalue()
        except Exception as audio_error:
            print(f"Error processing audio data: {audio_error}")
            raise ValueError(f"Failed to process audio data: {audio_error}")

        # Save to temporary file for processing
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        audio_path = f"{tmp_dir}/audio_{time.time()}.wav"
        
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)
        print(f"Saved audio to temporary file: {audio_path}")

        # Monitor GPU usage before processing
        # log_gpu("BEFORE generate_frames")
        
        # Generate lip-sync frames from audio
        async with generate_sema:
            print(f"Starting frame generation...")
            frames = await asyncio.get_running_loop().run_in_executor(
                gpu_executor, generate_frames, audio_path
            )
        
        # Monitor GPU usage after processing
        # log_gpu("AFTER generate_frames")
        
        # Log successful frame generation
        print(f"✅ Successfully generated {len(frames)} frames")
        
        try:
            # Clean up temporary file
            os.remove(audio_path)
        except Exception as cleanup_error:
            print(f"Warning: Failed to remove temporary file {audio_path}: {cleanup_error}")

        # Format response with the expected structure
        response = package_frames_for_frontend(frames, uuid_str)
        
        # Debug response structure before sending
        debug_response_structure(response)
        
        # Send response through WebRTC data channel
        if channel.readyState == "open":
            print(f"Sending response via WebRTC data channel...")
            channel.send(response)
            print(f"✅ Successfully sent {len(frames)} frames through WebRTC")
        else:
            print(f"⚠️ Cannot send response - channel state: {channel.readyState}")
            raise RuntimeError(f"WebRTC data channel not open: {channel.readyState}")
         
        # Send audio separately through WebSocket if needed
        # This is optional - only if your frontend expects audio through WebSocket
        if active_websockets:
            # Convert audio to base64 for WebSocket transmission
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            await send_audio_via_websocket(audio_b64, uuid_str)
    
    except Exception as e:
        print(f"❌ Error in processing job: {e}")
        traceback.print_exc()
        
        # Send error response if possible
        if channel and channel.readyState == "open":
            try:
                # Create error response with same format but status="failed"
                error_payload = bytearray()
                
                # 1. Add UUID (16 bytes) - use the original or zeros if unavailable
                if 'uuid_str' in locals():
                    try:
                        uuid_bytes = bytes.fromhex(uuid_str.replace('-', ''))
                        error_payload.extend(uuid_bytes)
                    except:
                        error_payload.extend(b'\0' * 16)
                else:
                    error_payload.extend(b'\0' * 16)
                
                # 2. Add status (10 bytes) - "failed" padded with null bytes
                error_payload.extend(b'failed'.ljust(10, b'\0'))
                
                # 3. Add error message (100 bytes) - truncated and padded
                error_msg = str(e).encode('utf-8')[:100].ljust(100, b'\0')
                error_payload.extend(error_msg)
                
                # 4. Add frame count (4 bytes) - zero frames
                error_payload.extend((0).to_bytes(4, byteorder='big'))
                
                # 5. No frames to add
                
                # Send error response
                channel.send(bytes(error_payload))
                print(f"Sent error response through WebRTC")
            except Exception as send_err:
                print(f"Failed to send error response: {send_err}")
    
    finally:
        # Decrement active jobs counter and check queue
        active_jobs -= 1
        await maybe_start_next_job()

async def send_audio_via_websocket(audio_b64, uuid_str):
    """Send audio data via WebSocket with matching UUID."""
    # Find the appropriate WebSocket connection
    ws_sent = False
    
    for ws in active_websockets:
        try:
            # Prepare audio message
            audio_msg = {
                "event": "playAudio",
                "media": {
                    "track": "inbound",
                    "payload": audio_b64
                },
                "streamId": uuid_str  # Use UUID as stream ID for matching
            }
            
            # Send message
            await ws.send_json(audio_msg)
            ws_sent = True
            print(f"Sent audio with UUID {uuid_str} via WebSocket")
        except Exception as e:
            print(f"Error sending to WebSocket: {e}")
    
    if not ws_sent:
        print("⚠️ No active Websockets to send audio")

# WebSocket handler function
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    client_id = request.query.get('agent', 'unknown')
    print(f"New WebSocket connection from client: {client_id}")
    
    # Add to active connections
    active_websockets.append(ws)
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    print(f"Received WebSocket message: {data.get('event', 'unknown')}")
                    
                    # Process messages according to your application logic
                    if data.get('event') == 'media':
                        # Handle audio data
                        pass
                    elif data.get('event') == 'ping':
                        # Respond to ping
                        await ws.send_json({"event": "pong"})
                    
                except json.JSONDecodeError:
                    print(f"Received invalid JSON: {msg.data[:100]}...")
            
            elif msg.type == web.WSMsgType.ERROR:
                print(f"WebSocket error: {ws.exception()}")
    
    finally:
        # Remove from active connections when closed
        if ws in active_websockets:
            active_websockets.remove(ws)
        print(f"WebSocket connection closed, client: {client_id}")
    
    return ws

def generate_frames(audio_path):
    """
    Generate lip-sync frames from audio file.
    This is where ML models convert audio to synchronized frames.
    """
    t0 = time.time()
    print(f"[Timing] Start processing audio chunk from: {audio_path}")

    # Use the correct AudioProcessor API for Whisper feature extraction
    whisper_feature, librosa_length = audio_processor.get_audio_feature(audio_path)
    t1 = time.time()
    print(f"[Timing] get_audio_feature took {t1 - t0:.3f} seconds")

    # Pass the loaded whisper model to get_whisper_chunk
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_feature,
        MODEL_DEVICE,
        unet.model.dtype,
        whisper,
        librosa_length,
        fps=FPS,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    )
    t2 = time.time()
    print(f"[Timing] feature2chunks took {t2 - t1:.3f} seconds")

    # Get batch data for processing
    gen = datagen(whisper_chunks, avatar_model.input_latent_list_cycle, BATCH_SIZE)
    whisper_batch, latent_batch = next(gen)
    t3 = time.time()
    print(f"[Timing] datagen processing took {t3 - t2:.3f} seconds")

    # Remove torch.from_numpy, since whisper_batch is already a Tensor
    audio_feat = whisper_batch.to(MODEL_DEVICE, dtype=unet.model.dtype, non_blocking=True)
    audio_feat = pe(audio_feat)
    latent_batch = latent_batch.to(MODEL_DEVICE, dtype=unet.model.dtype, non_blocking=True)

    # Forward pass through unet (with autocast)
    with torch.cuda.amp.autocast():
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feat).sample
    t4 = time.time()
    print(f"[Timing] Inference (unet forward pass) took {t4 - t3:.3f} seconds")

    # Decode with vae
    with torch.no_grad():
        recon = vae.decode_latents(pred_latents)
    t5 = time.time()
    print(f"[Timing] Decoding (vae) took {t5 - t4:.3f} seconds")

    # Process each frame
    full_frames = []
    for idx, patch in enumerate(recon):
        frame_start = time.time()
        np_patch = patch.cpu().numpy() if isinstance(patch, torch.Tensor) else patch
        if np_patch.ndim == 3 and np_patch.shape[0] == 3:
            np_patch = np_patch.transpose(1, 2, 0)
        np_patch = np_patch.astype(np.uint8)

        # Blend generated lip patch with original frame
        ori = avatar_model.frame_list_cycle[idx]
        x1, y1, x2, y2 = avatar_model.coord_list_cycle[idx]
        mask = avatar_model.mask_list_cycle[idx]
        mask_box = avatar_model.mask_coords_list_cycle[idx]
        resized_patch = cv2.resize(np_patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
        blended = get_image_blending(ori, resized_patch, [x1, y1, x2, y2], mask, mask_box)
        print(f"Frame {idx} shape: {blended.shape}")
        full_frames.append(blended)
        frame_end = time.time()
        print(f"[Timing] Processing frame {idx} took {frame_end - frame_start:.3f} seconds")

    total_time = time.time() - t0
    print(f"[Timing] Total time for generating frames for audio chunk: {total_time:.3f} seconds")
    return full_frames

async def ping(request):
    """Simple ping endpoint for connection testing."""
    return web.Response(text="pong")

def setup_application():
    """Configure and set up the web application."""
    app = web.Application()
    app.on_startup.append(init_models)
    
    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["POST", "GET", "OPTIONS", "DELETE", "PUT"]
        )
    })
    
    # Add routes
    app.router.add_route('GET', '/ws/', websocket_handler)
    
    # Add WebRTC and ping routes with CORS
    resource = cors.add(app.router.add_resource("/offer"))
    cors.add(resource.add_route("POST", offer))
    
    ping_resource = cors.add(app.router.add_resource("/ping"))
    cors.add(ping_resource.add_route("GET", ping))
    
    return app

def cleanup():
    """Clean up resources on shutdown."""
    # Close all peer connections
    for pc in pcs:
        pc.close()
    
    # Clean up temporary files
    for file in Path("tmp").glob("audio_*.wav"):
        try:
            file.unlink()
        except:
            pass

if __name__ == "__main__":
    try:
        app = setup_application()
        web.run_app(app, host="0.0.0.0", port=8080)
    finally:
        cleanup()