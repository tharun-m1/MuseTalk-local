#!/usr/bin/env python3
"""
Prepare Avatar and Run Concurrency Test

This script ensures the avatar is prepared before running concurrency tests.
It will automatically prepare the avatar if it doesn't exist.

Usage:
    python prepare_and_test.py --config configs/inference/realtime.yaml
"""

import argparse
import os
import sys
import subprocess
from omegaconf import OmegaConf
import json

def check_avatar_exists(avatar_id):
    """Check if avatar exists and is properly prepared"""
    avatar_path = f"./results/v15/avatars/{avatar_id}"
    
    if not os.path.exists(avatar_path):
        return False, "Avatar directory doesn't exist"
    
    # Check required files
    required_files = [
        f"{avatar_path}/coords.pkl",
        f"{avatar_path}/latents.pt", 
        f"{avatar_path}/mask_coords.pkl",
        f"{avatar_path}/avator_info.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        return False, f"Missing files: {', '.join(missing_files)}"
    
    return True, "Avatar ready"

def prepare_avatar(config_path, version="v1.5", mode="realtime"):
    """Prepare avatar using the normal inference script"""
    print("Preparing avatar...")
    
    # Set config to preparation mode
    config = OmegaConf.load(config_path)
    
    # Backup original config
    backup_path = config_path + ".backup"
    with open(backup_path, 'w') as f:
        OmegaConf.save(config, f)
    
    # Set preparation to True for all avatars
    modified = False
    for avatar_id in config:
        if 'preparation' in config[avatar_id]:
            if not config[avatar_id]['preparation']:
                config[avatar_id]['preparation'] = True
                modified = True
                print(f"  Enabled preparation for {avatar_id}")
    
    if modified:
        # Save modified config
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)
        
        try:
            # Run preparation
            print("  Running avatar preparation...")
            if os.path.exists("./inference.sh"):
                result = subprocess.run(["./inference.sh", version, mode], 
                                      capture_output=True, text=True, timeout=300)
            else:
                # Fallback to direct python call
                if version == "v1.5":
                    model_path = "./models/musetalkV15/unet.pth"
                    config_file = "./models/musetalkV15/musetalk.json"
                    version_arg = "v15"
                else:
                    model_path = "./models/musetalk/pytorch_model.bin"
                    config_file = "./models/musetalk/musetalk.json"
                    version_arg = "v1"
                
                cmd = [
                    "python3", "-m", "scripts.realtime_inference",
                    "--inference_config", config_path,
                    "--unet_model_path", model_path,
                    "--unet_config", config_file,
                    "--version", version_arg,
                    "--fps", "12" if mode == "realtime" else "25"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"  Preparation failed: {result.stderr}")
                return False
            else:
                print("  ✓ Avatar preparation completed")
                
        except subprocess.TimeoutExpired:
            print("  Preparation timed out")
            return False
        except Exception as e:
            print(f"  Preparation failed: {e}")
            return False
        finally:
            # Restore original config
            os.rename(backup_path, config_path)
            
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare avatar and run concurrency test")
    parser.add_argument("--config", type=str, default="configs/inference/realtime.yaml",
                       help="Path to inference config")
    parser.add_argument("--version", type=str, default="v1.5", choices=["v1.0", "v1.5"])
    parser.add_argument("--mode", type=str, default="realtime", choices=["normal", "realtime"])
    parser.add_argument("--skip-preparation", action="store_true", 
                       help="Skip avatar preparation check")
    
    args = parser.parse_args()
    
    print("MuseTalk Avatar Preparation & Concurrency Test")
    print("=" * 50)
    
    # Load config to get avatar info
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = OmegaConf.load(args.config)
    avatar_ids = list(config.keys())
    
    print(f"Found {len(avatar_ids)} avatar(s): {', '.join(avatar_ids)}")
    
    # Check each avatar
    all_ready = True
    for avatar_id in avatar_ids:
        exists, status = check_avatar_exists(avatar_id)
        print(f"  {avatar_id}: {status}")
        if not exists:
            all_ready = False
    
    # Prepare avatars if needed
    if not all_ready and not args.skip_preparation:
        print("\nSome avatars need preparation...")
        if prepare_avatar(args.config, args.version, args.mode):
            print("✓ All avatars prepared successfully")
        else:
            print("✗ Avatar preparation failed")
            print("\nTry running preparation manually:")
            print(f"  ./inference.sh {args.version} {args.mode}")
            print("  (Make sure preparation: True in your config)")
            sys.exit(1)
    elif not all_ready:
        print("\nSkipping preparation (--skip-preparation flag used)")
        print("Make sure to prepare avatars manually before running concurrency test")
    
    # Run concurrency test
    print(f"\nStarting concurrency test...")
    
    try:
        # Determine model paths based on version
        if args.version == "v1.5":
            model_path = "./models/musetalkV15/unet.pth"
            config_file = "./models/musetalkV15/musetalk.json"
            version_arg = "v15"
        else:
            model_path = "./models/musetalk/pytorch_model.bin"
            config_file = "./models/musetalk/musetalk.json"
            version_arg = "v1"
        
        cmd = [
            "python3", "concurrency_test.py",
            "--inference_config", args.config,
            "--unet_model_path", model_path,
            "--unet_config", config_file,
            "--version", version_arg,
            "--fps", "12" if args.mode == "realtime" else "25"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nConcurrency test interrupted by user")
    except Exception as e:
        print(f"Error running concurrency test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()