"""
Script to run biomarker inference.
This script loads traced models and processes npz files to get biomarker scores.

Usage:
    python run_biomarker.py data/anonymized_positive_example.npz --eeg_model encoder.pt --classifier_model classifier.pt

Or as a module:
    from run_biomarker import run_inference
    result = run_inference('data/anonymized_positive_example.npz', 'encoder.pt', 'classifier.pt')
"""

import torch
import numpy as np
import argparse
import os
from scipy.ndimage import zoom
from scipy.ndimage.filters import minimum_filter1d


# ============================================================================
# Preprocessing functions (standalone implementations)
# ============================================================================

def signal_std(signal):
    """Calculate robust standard deviation."""
    if len(signal) < 10:
        return 1
    else:
        cut = int(len(signal) * 0.1)
        std = np.std(np.sort(signal)[cut:-cut])
    std = 1 if std == 0 else std
    return std


def signal_normalize(signal):
    """Normalize signal by subtracting mean and dividing by std."""
    signal = signal - np.mean(signal)
    return signal / signal_std(signal)


def signal_crop(signal, clip_limit=6):
    """Clip signal to specified range."""
    return np.clip(signal, -clip_limit, clip_limit)


def label_to_interval(label, val=0):
    """Convert label array to intervals where label equals val."""
    hit = (label == val).astype(int)
    a = np.concatenate([np.zeros((1,)), hit.flatten(), np.zeros((1,))], axis=0)
    a = np.diff(a, axis=0)
    left = np.where(a == 1)[0]
    right = np.where(a == -1)[0]
    return np.array([*zip(left, right)], dtype=np.int32)


def signal_crop_motion(signal, window=10, fs=10, threshold=5):
    """Crop signal to remove motion artifacts."""
    signal_norm = signal_normalize(signal)
    threshold = max(np.max(np.abs(signal_norm)) * 0.5, threshold)
    normal_part = np.abs(signal_norm) < threshold
    normal_part = minimum_filter1d(normal_part, int(window * fs))
    indices = np.where(normal_part == 1)[0]
    signal_crop = signal_norm[indices]
    return signal_crop, indices


def detect_motion_iterative(signal, fs=10, level=3):
    """Iteratively detect and remove motion artifacts."""
    signal = signal.copy()
    motion = np.ones(len(signal), dtype=int)
    right_most_ratio = 1
    if level == 0 or len(signal) < 30 * fs:
        std = signal_std(signal)
        signal = signal / std
        right_most_ratio = 1 / std
        motion *= 0
    else:
        signal_crop, indices = signal_crop_motion(
            signal, window=10, threshold=10, fs=fs
        )
        if level == 3 and len(signal_crop) == len(signal):
            signal_crop, indices = signal_crop_motion(
                signal, window=10, threshold=6, fs=fs
            )
        motion[indices] = 0
        stable_periods = label_to_interval(motion, 0)
        for i, (p0, p1) in enumerate(stable_periods):
            signal_norm, right_r, motion_seg = detect_motion_iterative(
                signal[p0:p1], level=level - 1
            )
            signal[p0:p1] = signal_norm
            motion[p0:p1] = motion_seg
            if i != len(stable_periods) - 1:
                signal[p1: stable_periods[i + 1][0]] *= right_r
            else:
                right_most_ratio = right_r
    signal = np.clip(signal, -8, 8)
    return signal, right_most_ratio, motion


def normalize_by_ci(x):
    """Normalize by confidence interval (median and IQR)."""
    return (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25) + 1e-10) * 1.35


def preprocess_signal(signal_data, fs, crop_len=512):
    """
    Preprocess raw signal data - recreates BrOnlyDataset preprocessing steps.
    
    Args:
        signal_data: Raw 1D signal data as numpy array [T]
        fs: Sampling frequency (Hz)
        crop_len: Crop length for model input
    
    Returns:
        br: Processed breathing spectrogram [3, 256, T] (numpy array)
        br_1d: Processed 1D breathing [1, T] (numpy array)
    """
    br_fs = 10
    win = 30
    clip_range = 4
    do_normalize = True
    
    # Limit signal length (matching original)
    max_len = int(120 * 4 * 60 * fs)
    if len(signal_data) > max_len:
        signal_data = signal_data[:max_len]
    
    # Resample to 10 Hz if needed
    if fs != br_fs:
        signal_data = zoom(signal_data, br_fs / fs)
    
    # Create dummy spectrogram (zeros) - matching BrOnlyDataset
    br_time_len = max(1, signal_data.shape[0] // int(win * br_fs))
    br = np.zeros((256, br_time_len), dtype=np.float32)
    
    # Create dummy stage (all 9s = awake)
    stage = np.ones(br_time_len, dtype=np.float32) * 9
    
    # Find sleep period (but in practice, we use all data)
    idx_sleep_onset = 0
    try:
        idx_last_in_sleep = np.where((stage > 0) & (stage <= 5))[0][-1]
    except:
        idx_last_in_sleep = len(stage) - 1
    
    # Crop to sleep period
    stage = stage[idx_sleep_onset:idx_last_in_sleep + 1]
    br = br[:, idx_sleep_onset:idx_last_in_sleep + 1]
    br_1d = signal_data[idx_sleep_onset * br_fs * 30:(idx_last_in_sleep + 1) * br_fs * 30]
    
    # Process br_1d: detect motion, normalize, crop
    br_1d, _, _ = detect_motion_iterative(br_1d, br_fs)
    br_1d = signal_normalize(br_1d)
    br_1d = signal_crop(br_1d)
    
    # Normalize br
    if do_normalize:
        br = normalize_by_ci(br)
    br = np.clip(br, -clip_range, clip_range)
    br = ((br + clip_range) / (2 * clip_range) * 255).astype(np.uint8)
    br = np.concatenate([br[:, :, None]] * 3, 2)  # [256, T, 3]
    
    # Pad if shorter than crop_len
    if br.shape[1] < crop_len:
        br_new = np.zeros((br.shape[0], crop_len, br.shape[2]), dtype=br.dtype)
        br_new[:, :br.shape[1]] = br
        br = br_new
        br_1d_new = np.zeros(crop_len * br_fs * 30, dtype=br_1d.dtype)
        br_1d_new[:br_1d.shape[0]] = br_1d
        br_1d = br_1d_new
    
    # Ensure br_1d has correct length
    if br.shape[1] * br_fs * 30 > br_1d.shape[0]:
        br_1d_new = np.zeros(br.shape[1] * br_fs * 30, dtype=br_1d.dtype)
        br_1d_new[:br_1d.shape[0]] = br_1d
        br_1d = br_1d_new
    
    # For inference mode, preserve full length (don't crop)
    # Transpose br to [3, 256, T]
    br = br.transpose([2, 0, 1])
    
    # Normalize br
    br = br.astype(np.float32) / 255.0
    br = (br - 0.5) / 0.5
    
    # Add channel dimension to br_1d
    br_1d = br_1d[None, :]  # [1, T]
    
    return br, br_1d


# ============================================================================
# Inference functions
# ============================================================================

def get_eeg_from_signal(signal_data, fs, traced_eeg_model_path, crop_len=512, device='cuda'):
    """
    Run inference using traced EEG model - only needs signal and fs.
    
    Args:
        signal_data: Raw signal data as numpy array [T]
        fs: Sampling frequency (Hz)
        traced_eeg_model_path: Path to traced get_eeg model
        crop_len: Crop length (default 512)
        device: Device to use
    
    Returns:
        eeg: Reconstructed EEG as numpy array [256, T]
        latent: Decoder latent representation as numpy array [F, T, D]
    """
    # Load traced model (no need to load original model)
    traced_model = torch.jit.load(traced_eeg_model_path, map_location=device)
    traced_model.eval()
    
    # Preprocess signal
    br, br_1d = preprocess_signal(signal_data, fs, crop_len)
    
    # Convert to tensors
    br = torch.Tensor(br).to(device).unsqueeze(0)  # [1, 3, 256, T]
    br_1d = torch.Tensor(br_1d).to(device).unsqueeze(0)  # [1, 1, T]
    
    # Process in clips (for longer signals)
    night_len = br.shape[-1]
    num_clips = night_len // crop_len + 1
    br_fs = 10
    
    # Initialize output tensors - get dimensions from first forward pass
    with torch.no_grad():
        # Test with first clip to get output shapes
        br_clip_test = br[..., :crop_len]
        br_1d_clip_test = br_1d[..., :crop_len*br_fs*30]
        recon_eeg_test, decoder_latent_test = traced_model(br_clip_test, br_1d_clip_test)
        
        # Get output shapes
        eeg_channels = recon_eeg_test.shape[1]
        eeg_height = recon_eeg_test.shape[2]
        if decoder_latent_test is not None:
            latent_freq = decoder_latent_test.shape[1]
            latent_emb_dim = decoder_latent_test.shape[3]
            latent_time_per_clip = decoder_latent_test.shape[2]
            # Calculate patch size from the ratio
            patch_size_time = crop_len // latent_time_per_clip
        else:
            latent_freq = 8
            latent_emb_dim = 768
            patch_size_time = 8  # Default patch size
    
    recon_eeg = torch.zeros(1, eeg_channels, eeg_height, night_len, device=device) - 1
    if decoder_latent_test is not None:
        latent_time_len = night_len // patch_size_time
        decoder_eeg_latent = torch.zeros(
            1, latent_freq, 
            latent_time_len,
            latent_emb_dim,
            device=device
        )
    else:
        decoder_eeg_latent = None
    
    # Process each clip
    with torch.no_grad():
        for clip_idx in range(num_clips):
            if clip_idx < num_clips - 1:
                start_idx = clip_idx * crop_len
                end_idx = (clip_idx + 1) * crop_len
            else:
                # tail (last clip)
                start_idx = night_len - crop_len
                end_idx = night_len
            
            br_clip = br[..., start_idx:end_idx]
            br_1d_clip = br_1d[..., start_idx*br_fs*30:end_idx*br_fs*30]
            
            # Run traced model (only needs br_clip and br_1d_clip)
            recon_eeg_clip, decoder_latent_clip = traced_model(br_clip, br_1d_clip)
            
            recon_eeg[..., start_idx:end_idx] = recon_eeg_clip
            
            if decoder_latent_clip is not None and decoder_eeg_latent is not None:
                # Latent time dimension uses patch_size_time
                latent_start = start_idx // patch_size_time
                latent_end = end_idx // patch_size_time
                if latent_end <= decoder_eeg_latent.shape[2]:
                    decoder_eeg_latent[:, :, latent_start:latent_end, :] = decoder_latent_clip
    
    # Convert to numpy
    eeg_output = recon_eeg.cpu().numpy()[0, 0]  # [256, T] - take first channel
    if decoder_eeg_latent is not None:
        latent_output = decoder_eeg_latent.cpu().numpy()[0]  # [F, T, D]
    else:
        latent_output = None
    
    return eeg_output, latent_output


def get_classification_from_signal(signal_data, fs, traced_eeg_model_path, traced_classifier_path, 
                                    modality='belt', device='cuda'):
    """
    Complete end-to-end function: signal -> classification output.
    
    Args:
        signal_data: Raw signal data as numpy array [T]
        fs: Sampling frequency (Hz)
        traced_eeg_model_path: Path to traced get_eeg model
        traced_classifier_path: Path to traced EndToEndFold model
        modality: Modality string ('belt', 'airflow', 'wireless')
        device: Device to use
    
    Returns:
        output: Classifier output tensor [1, 1]
    """
    # Step 1: Get latent from signal
    _, latent = get_eeg_from_signal(signal_data, fs, traced_eeg_model_path, device=device)
    
    # Step 2: Convert latent to tensor
    latent_tensor = torch.tensor(latent, dtype=torch.float32).to(device)  # [F, T, D]
    
    # Step 3: Load traced classifier
    traced_classifier = torch.jit.load(traced_classifier_path, map_location=device)
    traced_classifier.eval()
    
    # Step 4: Convert modality to tensor
    modality_map = {'belt': 0, 'airflow': 1, 'wireless': 2}
    modality_tensor = torch.tensor([modality_map[modality]], dtype=torch.int64, device=device)
    
    # Step 5: Run through classifier
    with torch.no_grad():
        output = traced_classifier(latent_tensor, modality_tensor)
    
    return output


def run_inference(npz_file_path, eeg_model_path, classifier_model_path, modality='belt', device='cuda'):
    """
    Main inference function - loads npz file and returns classification result.
    
    Args:
        npz_file_path: Path to input npz file (must contain 'data' and 'fs' keys)
        eeg_model_path: Path to traced get_eeg model (.pt file)
        classifier_model_path: Path to traced classifier model (.pt file)
        modality: Modality string ('belt', 'airflow', 'wireless')
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        result: Classification output as numpy array [1, 1]
    """
    # Load input data
    if not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"Input file not found: {npz_file_path}")
    
    data = np.load(npz_file_path)
    if 'data' not in data:
        raise ValueError("Input file must contain 'data' key")
    if 'fs' not in data:
        raise ValueError("Input file must contain 'fs' key")
    
    signal_data = data['data']
    fs = float(data['fs'])
    
    # Run end-to-end inference
    output = get_classification_from_signal(
        signal_data, fs, eeg_model_path, classifier_model_path,
        modality=modality, device=device
    )
    
    # Convert to numpy
    result = torch.sigmoid(output)
    result = result.cpu().numpy()
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standalone inference for MAGE model')
    parser.add_argument('input_file', type=str, help='Path to input npz file (must contain "data" and "fs" keys)')
    parser.add_argument('--eeg_model', type=str, default='get_eeg_traced.pt', 
                       help='Path to traced get_eeg model')
    parser.add_argument('--classifier_model', type=str, default='end_to_end_fold_traced.pt',
                       help='Path to traced classifier model')
    parser.add_argument('--modality', type=str, default='belt', choices=['belt', 'airflow', 'wireless'],
                       help='Modality type')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: save output to this file (numpy .npy format)')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists(args.eeg_model):
        raise FileNotFoundError(f"EEG model not found: {args.eeg_model}")
    if not os.path.exists(args.classifier_model):
        raise FileNotFoundError(f"Classifier model not found: {args.classifier_model}")
    
    # Run inference
    print('\n\n')
    print('_' * 50)
    print('Running biomarker inference')
    print('_' * 50)
    print('\n\n')
    print(f"Loading input from: {args.input_file}")
    print(f"Using Encoder: {args.eeg_model}")
    print(f"Using Classifier: {args.classifier_model}")
    print('\n')
    
    result = run_inference(
        args.input_file,
        args.eeg_model,
        args.classifier_model,
        modality=args.modality,
        device='cpu'
    )
    print(f"Classification result: {result[0, 0]:.4f}")
    print('\n')
    
