# MuseTalk Concurrency Testing

This set of scripts helps you test the concurrency capabilities of your MuseTalk inference system to determine how many parallel inferences you can run simultaneously for 500ms audio chunks **while keeping each individual task under 350ms**.

## Key Goal
Find the maximum number of concurrent 500ms audio chunk lipsync generations where each task completes within 350ms (suitable for real-time applications).

## Files

1. **`concurrency_test.py`** - Main testing script with 350ms target focus
2. **`prepare_and_test.py`** - All-in-one script that prepares avatar and runs tests
3. **`test_concurrency.sh`** - Bash wrapper script 
4. **`requirements_concurrency.txt`** - Additional Python dependencies
5. **`CONCURRENCY_TESTING_README.md`** - This file

## Quick Start (Recommended)

The easiest way to run the concurrency test:

### 1. Install Dependencies
```bash
pip install -r requirements_concurrency.txt
```

### 2. Run Everything At Once
```bash
python3 prepare_and_test.py --config configs/inference/realtime.yaml --version v1.5 --mode realtime
```

This script will:
- Check if your avatar is prepared
- Automatically prepare it if needed (sets `preparation: True` temporarily)
- Run the concurrency test focused on your 350ms target
- Restore your original config

## Manual Setup (Alternative)

If you prefer to do things step by step:

### 1. Prepare Your Avatar First
Edit your YAML config to set `preparation: True`:
```yaml
avator_1:
  preparation: True  # Change this to True
  bbox_shift: 12
  video_path: "data/video/dummy.mp4"
  audio_clips:
    audio_0: "data/audio/500ms.mp3"
```

Run normal inference once:
```bash
./inference.sh v1.5 realtime
```

### 2. Set preparation back to False
```yaml
avator_1:
  preparation: False  # Change back to False
```

### 3. Run Concurrency Test
```bash
python3 concurrency_test.py --inference_config configs/inference/realtime.yaml
```

## What the Test Does

1. **Model Loading**: Loads all MuseTalk models once (shared across all workers)
2. **Avatar Data Loading**: Loads prepared avatar data (coordinates, latents, frames, masks)
3. **Concurrency Testing**: Tests different numbers of parallel workers with **finer increments** (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24)
4. **350ms Target Focus**: For each concurrency level:
   - Runs 3 test iterations
   - Measures individual task completion time
   - **Automatically stops when tasks exceed 350ms**
   - Monitors system resources (CPU, RAM, GPU)
   - Calculates throughput and success rates
5. **Smart Stopping**: Stops testing when:
   - Individual task time exceeds 350ms, OR
   - Success rate drops below 80%, OR
   - Tasks timeout (likely GPU memory exhaustion)

## Output

### Console Output
Real-time results showing:
```
============================================================
Testing 2 concurrent workers
============================================================
Test run 1/3
  Overall time: 0.36s
  Successful: 2/2
  Success rate: 100.0%
  Avg task time: 320ms ✅ (target: ≤350ms)
  Throughput: 5.55 inferences/sec
  Frames/sec: 33.3

🛑 Stopping tests - average task time (380ms) exceeds 350ms limit
   Maximum recommended concurrency: 2 workers
```

### Final Summary
```
CONCURRENCY TEST RESULTS - 350ms Target Analysis
===============================================================================
Workers  Success% Task Time   Meets 350ms  Throughput   FPS    
---------------------------------------------------------------------------
1        100.0    220ms      ✅           4.53         27.2   
2        100.0    320ms      ✅           5.55         33.3   
3        100.0    380ms      ❌           4.89         29.3   

===============================================================================
RECOMMENDATION FOR 500ms AUDIO CHUNKS
===============================================================================
🎯 Maximum recommended concurrency: 2 workers
   ✅ Task time: 320ms (under 350ms limit)
   ✅ Success rate: 100.0%
   📈 Total throughput: 5.55 inferences/sec
   🎬 Frame rate: 33.3 fps

💡 Real-world performance:
   • Can process 5.5 × 500ms audio chunks per second
   • Equivalent to 2.8 seconds of audio per second
   • Processing speed: 280% of real-time
```

### Results File
Detailed JSON file saved as `concurrency_test_results_[timestamp].json` containing:
- Summary statistics for each concurrency level
- Individual task performance data
- System resource usage
- 350ms target compliance analysis

## Key Metrics

- **Success Rate**: Percentage of inferences that complete successfully
- **Throughput**: Number of inferences completed per second
- **Average Time**: Mean time per inference across all workers
- **Max Time**: Longest single inference time

## Optimization Tips

1. **GPU Memory**: The main bottleneck is usually GPU memory. Monitor GPU usage in the results.

2. **Batch Size**: Try different `--batch_size` values (10, 15, 20, 25) to find the sweet spot.

3. **Model Precision**: The script uses half precision (`.half()`) to save GPU memory.

4. **System Resources**: Monitor CPU and RAM usage to identify other bottlenecks.

## Customization

### Test Different Audio Lengths
Modify your YAML config to test different audio clip lengths:
```yaml
avator_1:
  audio_clips:
    audio_0: "data/audio/500ms.mp3"   # 500ms
    audio_1: "data/audio/1000ms.mp3"  # 1 second
    audio_2: "data/audio/2000ms.mp3"  # 2 seconds
```

### Adjust Concurrency Levels
Edit the `concurrency_levels` list in `concurrency_test.py`:
```python
concurrency_levels = [1, 2, 4, 6, 8, 10, 12, 16, 20]  # Customize as needed
```

### Change Test Parameters
Common parameters to adjust:
- `--batch_size`: Processing batch size (affects GPU memory usage)
- `--fps`: Video frame rate (affects processing time)
- `--gpu_id`: Which GPU to use for testing

## Troubleshooting

### Common Issues

1. **"avator_X does not exist, you should set preparation to True"**
   - This means the avatar hasn't been prepared yet
   - **Solution**: Use `prepare_and_test.py` (recommended) or manually prepare the avatar first
   - The avatar preparation only needs to be done once

2. **"Test timed out after 60 seconds - likely GPU memory issue"**
   - GPU memory exhausted at this concurrency level
   - **Solution**: The previous concurrency level is your maximum
   - Try reducing `--batch_size` and test again

3. **Tests stop with "average task time exceeds 350ms limit"**
   - **This is normal!** The test found your optimal concurrency level
   - The previous level is your maximum for real-time applications
   - To test higher levels anyway, increase the limit in the code

4. **Very low concurrency limits (only 1-2 workers)**
   - Try reducing batch size: `--batch_size 10` or `--batch_size 5`
   - Check GPU memory with `nvidia-smi` during testing
   - Consider using a more powerful GPU

5. **Import Errors**
   - Ensure you're running from the MuseTalk root directory
   - Check that all MuseTalk dependencies are installed

6. **Audio File Not Found**
   - Verify the audio path in your YAML config
   - Ensure the audio file exists and is accessible

### Performance Expectations

- **Good performance**: 2+ workers meeting 350ms target
- **Excellent performance**: 4+ workers meeting 350ms target  
- **Outstanding performance**: 6+ workers meeting 350ms target

If you're only getting 1 worker, this is still useful for single-user real-time applications!

### Performance Notes

- The test uses `skip_save_images=True` to focus on inference speed rather than I/O
- Each test run includes a 2-second pause to allow GPU memory cleanup
- Results may vary based on GPU temperature and system load

## Expected Results

For a typical setup:
- **RTX 3090/4090**: 8-12 concurrent inferences for 500ms audio
- **RTX 3080/4080**: 6-10 concurrent inferences
- **RTX 3070/4070**: 4-8 concurrent inferences

Your results will depend on:
- GPU memory size
- Model version (v1.0 vs v1.5)
- Audio length
- System configuration