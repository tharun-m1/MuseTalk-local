#!/usr/bin/env python3
"""
Concurrency Testing Script for MuseTalk Real-time Inference

This script tests how many parallel inferences can be run simultaneously
for 500ms audio chunks without significant performance degradation.

Usage:
    python concurrency_test.py --config configs/inference/realtime.yaml
"""

import argparse
import os
import sys
import time
import threading
import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import json
from omegaconf import OmegaConf
from transformers import WhisperModel
import subprocess
import statistics
import psutil
import GPUtil
import copy
import cv2
import pickle
import glob
from tqdm import tqdm

# Import MuseTalk modules
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending


class ConcurrencyTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.avatar_data = None
        self.setup_models()
        self.setup_avatar_data()
        
    def setup_models(self):
        """Initialize all models once for shared use"""
        print("Loading models...")
        start_time = time.time()
        
        # Load core models
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.args.unet_model_path,
            vae_type=self.args.vae_type,
            unet_config=self.args.unet_config,
            device=self.device
        )
        
        # Set up models with half precision
        self.timesteps = torch.tensor([0], device=self.device)
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        
        # Initialize audio processor and Whisper
        self.audio_processor = AudioProcessor(feature_extractor_path=self.args.whisper_dir)
        self.weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(self.args.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        
        load_time = time.time() - start_time
        print(f"Models loaded in {load_time:.2f} seconds")
        
    def setup_avatar_data(self):
        """Load avatar data once for shared use"""
        self.inference_config = OmegaConf.load(self.args.inference_config)
        
        # Get the first avatar for testing
        self.avatar_id = list(self.inference_config.keys())[0]
        self.avatar_config = self.inference_config[self.avatar_id]
        
        # Check if avatar exists
        avatar_path = f"./results/v15/avatars/{self.avatar_id}"
        if not os.path.exists(avatar_path):
            print(f"\nERROR: Avatar '{self.avatar_id}' not found at {avatar_path}")
            print("\nTo fix this, you need to prepare the avatar first:")
            print("1. Set 'preparation: True' in your YAML config")
            print("2. Run normal inference once:")
            print(f"   ./inference.sh v1.5 realtime")
            print("3. Then run the concurrency test again")
            sys.exit(1)
            
        # Load avatar data
        print(f"Loading avatar data for '{self.avatar_id}'...")
        
        try:
            # Load precomputed data
            coords_path = f"{avatar_path}/coords.pkl"
            latents_path = f"{avatar_path}/latents.pt"
            mask_coords_path = f"{avatar_path}/mask_coords.pkl"
            
            with open(coords_path, 'rb') as f:
                coord_list_cycle = pickle.load(f)
            
            input_latent_list_cycle = torch.load(latents_path)
            
            with open(mask_coords_path, 'rb') as f:
                mask_coords_list_cycle = pickle.load(f)
            
            # Load images
            full_imgs_path = f"{avatar_path}/full_imgs"
            input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            frame_list_cycle = read_imgs(input_img_list)
            
            mask_out_path = f"{avatar_path}/mask"
            input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
            input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            mask_list_cycle = read_imgs(input_mask_list)
            
            self.avatar_data = {
                'coord_list_cycle': coord_list_cycle,
                'input_latent_list_cycle': input_latent_list_cycle,
                'mask_coords_list_cycle': mask_coords_list_cycle,
                'frame_list_cycle': frame_list_cycle,
                'mask_list_cycle': mask_list_cycle
            }
            
            print(f"✓ Avatar data loaded successfully")
            print(f"  - {len(coord_list_cycle)} coordinate frames")
            print(f"  - {len(input_latent_list_cycle)} latent frames")
            print(f"  - {len(frame_list_cycle)} image frames")
            
        except Exception as e:
            print(f"Error loading avatar data: {e}")
            print("Make sure the avatar was properly prepared with 'preparation: True'")
            sys.exit(1)
        
    def get_system_stats(self):
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        gpu_stats = {}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[self.args.gpu_id] if self.args.gpu_id < len(gpus) else gpus[0]
                gpu_stats = {
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'load': gpu.load * 100
                }
        except:
            gpu_stats = {'error': 'Unable to get GPU stats'}
            
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'gpu': gpu_stats
        }
    
    def process_single_frame(self, frame_idx, res_frame, avatar_data):
        """Process a single result frame"""
        bbox = avatar_data['coord_list_cycle'][frame_idx % len(avatar_data['coord_list_cycle'])]
        ori_frame = copy.deepcopy(avatar_data['frame_list_cycle'][frame_idx % len(avatar_data['frame_list_cycle'])])
        x1, y1, x2, y2 = bbox
        
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except:
            return None
            
        mask = avatar_data['mask_list_cycle'][frame_idx % len(avatar_data['mask_list_cycle'])]
        mask_crop_box = avatar_data['mask_coords_list_cycle'][frame_idx % len(avatar_data['mask_coords_list_cycle'])]
        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        
        return combine_frame
    
    def single_inference_task(self, task_id, audio_path, barrier=None):
        """Run a single inference task using shared avatar data"""
        try:
            # Wait for all threads to be ready if barrier is provided
            if barrier:
                barrier.wait()
                
            start_time = time.time()
            
            # Extract audio features
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                audio_path, weight_dtype=self.weight_dtype)
            
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=self.args.fps,
                audio_padding_length_left=2,  # Default padding
                audio_padding_length_right=2,
            )
            
            # Run inference batch by batch
            video_num = len(whisper_chunks)
            res_frame_list = []
            
            gen = datagen(whisper_chunks,
                         self.avatar_data['input_latent_list_cycle'],
                         self.args.batch_size)
            
            for i, (whisper_batch, latent_batch) in enumerate(gen):
                audio_feature_batch = self.pe(whisper_batch.to(self.device))
                latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

                pred_latents = self.unet.model(latent_batch,
                                            self.timesteps,
                                            encoder_hidden_states=audio_feature_batch).sample
                pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                recon = self.vae.decode_latents(pred_latents)
                
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            
            # Process frames (but don't save them for speed)
            processed_frames = 0
            for idx, res_frame in enumerate(res_frame_list):
                combined_frame = self.process_single_frame(idx, res_frame, self.avatar_data)
                if combined_frame is not None:
                    processed_frames += 1
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            return {
                'task_id': task_id,
                'success': True,
                'inference_time': inference_time,
                'start_time': start_time,
                'end_time': end_time,
                'frames_processed': processed_frames,
                'total_frames': video_num
            }
            
        except Exception as e:
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'inference_time': None
            }
    
    def test_concurrency_level(self, num_workers, num_tests=1):
        """Test a specific concurrency level"""
        print(f"\n{'='*60}")
        print(f"Testing {num_workers} concurrent workers")
        print(f"{'='*60}")
        
        audio_path = list(self.avatar_config["audio_clips"].values())[0]
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file {audio_path} not found")
            return None
            
        test_results = []
        
        for test_run in range(num_tests):
            print(f"\nTest run {test_run + 1}/{num_tests}")
            
            # Record system stats before test
            pre_stats = self.get_system_stats()
            
            # Create barrier for synchronized start
            barrier = threading.Barrier(num_workers)
            
            # Run concurrent tasks with timeout protection
            overall_start = time.time()
            
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(self.single_inference_task, i, audio_path, barrier)
                        for i in range(num_workers)
                    ]
                    
                    # Wait for all tasks to complete with timeout
                    results = []
                    for future in concurrent.futures.as_completed(futures, timeout=60):  # 60s timeout
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            results.append({
                                'task_id': -1,
                                'success': False,
                                'error': f"Task exception: {e}",
                                'inference_time': None
                            })
            except concurrent.futures.TimeoutError:
                print(f"  ⚠️  Test timed out after 60 seconds - likely GPU memory issue")
                # Create failure results for timed out tasks
                results = [{
                    'task_id': i,
                    'success': False,
                    'error': 'Timeout - likely GPU memory exhaustion',
                    'inference_time': None
                } for i in range(num_workers)]
            
            overall_end = time.time()
            overall_time = overall_end - overall_start
            
            # Record system stats after test
            post_stats = self.get_system_stats()
            
            # Analyze results
            successful_tasks = [r for r in results if r['success']]
            failed_tasks = [r for r in results if not r['success']]
            
            if successful_tasks:
                inference_times = [r['inference_time'] for r in successful_tasks]
                avg_inference_time = statistics.mean(inference_times)
                max_inference_time = max(inference_times)
                min_inference_time = min(inference_times)
                std_inference_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
                total_frames = sum(r['frames_processed'] for r in successful_tasks)
            else:
                avg_inference_time = max_inference_time = min_inference_time = std_inference_time = total_frames = None
            
            test_result = {
                'test_run': test_run + 1,
                'num_workers': num_workers,
                'overall_time': overall_time,
                'successful_tasks': len(successful_tasks),
                'failed_tasks': len(failed_tasks),
                'success_rate': len(successful_tasks) / num_workers * 100,
                'avg_inference_time': avg_inference_time,
                'max_inference_time': max_inference_time,
                'min_inference_time': min_inference_time,
                'std_inference_time': std_inference_time,
                'throughput': len(successful_tasks) / overall_time if overall_time > 0 else 0,
                'frames_per_second': total_frames / overall_time if overall_time > 0 and total_frames else 0,
                'pre_stats': pre_stats,
                'post_stats': post_stats,
                'failed_errors': [r.get('error', 'Unknown') for r in failed_tasks]
            }
            
            test_results.append(test_result)
            
            # Print immediate results with 450ms target focus
            print(f"  Overall time: {overall_time:.2f}s")
            print(f"  Successful: {len(successful_tasks)}/{num_workers}")
            print(f"  Success rate: {test_result['success_rate']:.1f}%")
            if avg_inference_time:
                task_time_ms = avg_inference_time * 1000
                meets_target = "✅" if task_time_ms <= 450 else "❌"
                print(f"  Avg task time: {task_time_ms:.0f}ms {meets_target} (target: ≤450ms)")
                print(f"  Throughput: {test_result['throughput']:.2f} inferences/sec")
                if total_frames:
                    print(f"  Frames/sec: {test_result['frames_per_second']:.1f}")
            else:
                print(f"  No successful tasks completed")
            
            # Wait between test runs
            if test_run < num_tests - 1:
                time.sleep(2)
                
        return test_results
    
    def run_preflight_check(self):
        """Run a single inference to verify everything works before concurrency testing"""
        print("\nRunning preflight check...")
        
        audio_path = list(self.avatar_config["audio_clips"].values())[0]
        if not os.path.exists(audio_path):
            print(f"ERROR: Audio file not found: {audio_path}")
            return False
            
        try:
            # Try a single inference
            result = self.single_inference_task(0, audio_path)
            if result['success']:
                print(f"✓ Preflight check passed (took {result['inference_time']:.2f}s)")
                print(f"  Processed {result['frames_processed']}/{result['total_frames']} frames")
                return True
            else:
                print(f"✗ Preflight check failed: {result.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"✗ Preflight check failed with exception: {e}")
            return False
    
    def run_concurrency_tests(self):
        """Run tests with different concurrency levels"""
        print("Starting concurrency tests...")
        print(f"Device: {self.device}")
        print(f"Audio file: {list(self.avatar_config['audio_clips'].values())[0]}")
        
        # Run preflight check first
        if not self.run_preflight_check():
            print("\nPreflight check failed. Please fix the issues above before running concurrency tests.")
            sys.exit(1)
        
        # Test different concurrency levels with finer increments
        concurrency_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24]
        
        all_results = []
        
        for num_workers in concurrency_levels:
            try:
                level_results = self.test_concurrency_level(num_workers, num_tests=1)
                if level_results:
                    all_results.extend(level_results)
                    
                    # Check if we should stop based on your performance criteria
                    recent_success_rate = statistics.mean([r['success_rate'] for r in level_results])
                    valid_times = [r['avg_inference_time'] for r in level_results if r['avg_inference_time'] is not None]
                    avg_task_time = statistics.mean(valid_times) if valid_times else None
                    
                    # Stop if success rate drops below 80% OR average task time exceeds 450ms
                    if recent_success_rate < 80:
                        print(f"\n🛑 Stopping tests - success rate dropped to {recent_success_rate:.1f}%")
                        break
                    elif avg_task_time and avg_task_time > 0.45:  # 450ms limit
                        print(f"\n🛑 Stopping tests - average task time ({avg_task_time*1000:.0f}ms) exceeds 450ms limit")
                        print(f"   Maximum recommended concurrency: {num_workers-1} workers" if num_workers > 1 else "")
                        break
                        
            except KeyboardInterrupt:
                print("\nTests interrupted by user")
                break
            except Exception as e:
                print(f"Error testing {num_workers} workers: {e}")
                continue
        
        self.results = all_results
        self.analyze_and_report_results()
    
    def analyze_and_report_results(self):
        """Analyze results and generate report"""
        if not self.results:
            print("No results to analyze")
            return
            
        print(f"\n{'='*80}")
        print("CONCURRENCY TEST RESULTS - 450ms Target Analysis")
        print(f"{'='*80}")
        
        # Group results by concurrency level
        by_concurrency = {}
        for result in self.results:
            level = result['num_workers']
            if level not in by_concurrency:
                by_concurrency[level] = []
            by_concurrency[level].append(result)
        
        # Analyze each concurrency level
        summary_data = []
        
        print(f"{'Workers':<8} {'Success%':<8} {'Task Time':<11} {'Meets 450ms':<12} {'Throughput':<12} {'FPS':<8}")
        print("-" * 75)
        
        max_valid_workers = 0
        
        for level in sorted(by_concurrency.keys()):
            results = by_concurrency[level]
            
            avg_success_rate = statistics.mean([r['success_rate'] for r in results])
            avg_throughput = statistics.mean([r['throughput'] for r in results])
            avg_fps = statistics.mean([r['frames_per_second'] for r in results if r['frames_per_second']])
            
            valid_times = [r['avg_inference_time'] for r in results if r['avg_inference_time'] is not None]
            avg_time = statistics.mean(valid_times) if valid_times else None
            
            # Check if meets 450ms target
            meets_target = avg_time is not None and avg_time <= 0.45 and avg_success_rate >= 90
            target_symbol = "✅" if meets_target else "❌"
            
            if meets_target:
                max_valid_workers = level
            
            summary_data.append({
                'workers': level,
                'success_rate': avg_success_rate,
                'avg_time': avg_time,
                'meets_target': meets_target,
                'throughput': avg_throughput,
                'fps': avg_fps
            })
            
            time_str = f"{avg_time*1000:.0f}ms" if avg_time else "N/A"
            fps_str = f"{avg_fps:.1f}" if avg_fps else "N/A"
            
            print(f"{level:<8} {avg_success_rate:<8.1f} {time_str:<11} {target_symbol:<12} {avg_throughput:<12.2f} {fps_str:<8}")
        
        # Report optimal configuration
        print(f"\n{'='*80}")
        print("RECOMMENDATION FOR 500ms AUDIO CHUNKS")
        print(f"{'='*80}")
        
        if max_valid_workers > 0:
            optimal_data = next(r for r in summary_data if r['workers'] == max_valid_workers)
            print(f"🎯 Maximum recommended concurrency: {max_valid_workers} workers")
            print(f"   ✅ Task time: {optimal_data['avg_time']*1000:.0f}ms (under 450ms limit)")
            print(f"   ✅ Success rate: {optimal_data['success_rate']:.1f}%")
            print(f"   📈 Total throughput: {optimal_data['throughput']:.2f} inferences/sec")
            if optimal_data['fps']:
                print(f"   🎬 Frame rate: {optimal_data['fps']:.1f} fps")
            
            # Calculate real-world capacity
            print(f"\n💡 Real-world performance:")
            print(f"   • Can process {optimal_data['throughput']:.1f} × 500ms audio chunks per second")
            print(f"   • Equivalent to {optimal_data['throughput']*0.5:.1f} seconds of audio per second")
            print(f"   • Processing speed: {(optimal_data['throughput']*0.5)*100:.0f}% of real-time")
        else:
            print(f"⚠️  No concurrency level met the 450ms target!")
            print(f"   Consider reducing batch size or using a more powerful GPU")
        
        # Save detailed results
        output_file = f"concurrency_test_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary': summary_data,
                'detailed_results': self.results,
                'test_config': {
                    'device': str(self.device),
                    'version': self.args.version,
                    'batch_size': self.args.batch_size,
                    'fps': self.args.fps,
                    'avatar_id': self.avatar_id
                }
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test MuseTalk inference concurrency")
    
    # Model and path arguments
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--parsing_mode", default='jaw')
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    
    args = parser.parse_args()
    
    print("MuseTalk Concurrency Tester")
    print("=" * 50)
    print(f"Config: {args.inference_config}")
    print(f"Model: {args.unet_model_path}")
    print(f"Version: {args.version}")
    print()
    
    # Validate paths
    if not os.path.exists(args.inference_config):
        print(f"Error: Inference config file not found: {args.inference_config}")
        print("\nMake sure you're running from the MuseTalk root directory.")
        sys.exit(1)
        
    if not os.path.exists(args.unet_model_path):
        print(f"Error: UNet model not found: {args.unet_model_path}")
        sys.exit(1)
    
    # Run concurrency tests
    tester = ConcurrencyTester(args)
    tester.run_concurrency_tests()


if __name__ == "__main__":
    main()