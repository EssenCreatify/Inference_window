#!/usr/bin/env python3
"""
Debug script to test face quality calculation
"""

import cv2
import numpy as np
import torch
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from window_selector import WindowScorer

def init_face_detector():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet18",
            pretrained=True,
            num_nb=10,
            num_lms=68,
            net_stride=32,
            input_size=256,
            meanface_type="300w",
            map_location=device,
            checkpoint=None,
        )
    )

def debug_face_quality():
    """Debug the face quality calculation"""
    
    # Initialize face detector
    init_face_detector()
    scorer = WindowScorer()
    
    # Load a frame from the video
    video_path = "/mnt/round-cake/home/essen/inference_window/data/Outdoor-talking_video-Laura_Martinez.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read video frame")
        return
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get face detection results
    landmarks, bb = torchlm.runtime.forward(frame_rgb)
    
    print(f"🔍 Debug Information:")
    print(f"   Frame shape: {frame.shape}")
    print(f"   Number of faces detected: {bb.shape[0]}")
    
    if bb.shape[0] == 0:
        print("❌ No faces detected")
        return
    
    # Face confidence
    face_confidence = float(bb[0][4])
    print(f"   Face confidence: {face_confidence:.4f}")
    
    # Face quality calculation
    face_landmarks = landmarks[0]
    print(f"   Landmarks shape: {face_landmarks.shape}")
    print(f"   Landmarks range: x=[{face_landmarks[:, 0].min():.1f}, {face_landmarks[:, 0].max():.1f}], y=[{face_landmarks[:, 1].min():.1f}, {face_landmarks[:, 1].max():.1f}]")
    
    # Check region visibility
    h, w = frame.shape[:2]
    facial_regions = {
        'left_eye': list(range(36, 42)),
        'right_eye': list(range(42, 48)),
        'nose': list(range(27, 36)),
        'mouth': list(range(48, 68)),
        'face_contour': list(range(0, 17)),
        'left_eyebrow': list(range(17, 22)),
        'right_eyebrow': list(range(22, 27))
    }
    
    print(f"\n📊 Region Analysis:")
    for region_name, landmark_indices in facial_regions.items():
        region_landmarks = face_landmarks[landmark_indices]
        valid_landmarks = 0
        total_landmarks = len(landmark_indices)
        
        for landmark in region_landmarks:
            x, y = landmark
            if 0 <= x < w and 0 <= y < h:
                valid_landmarks += 1
        
        visibility_score = valid_landmarks / total_landmarks
        print(f"   {region_name}: {valid_landmarks}/{total_landmarks} = {visibility_score:.3f}")
    
    # Calculate face quality manually
    region_weights = {
        'left_eye': 0.10,
        'right_eye': 0.10,
        'nose': 0.15,
        'mouth': 0.35,
        'face_contour': 0.15,
        'left_eyebrow': 0.075,
        'right_eyebrow': 0.075
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    for region_name, landmark_indices in facial_regions.items():
        region_landmarks = face_landmarks[landmark_indices]
        valid_landmarks = 0
        total_landmarks = len(landmark_indices)
        
        for landmark in region_landmarks:
            x, y = landmark
            if 0 <= x < w and 0 <= y < h:
                valid_landmarks += 1
        
        visibility_score = valid_landmarks / total_landmarks
        weight = region_weights[region_name]
        total_score += visibility_score * weight
        total_weight += weight
    
    base_quality = total_score / total_weight
    
    # Variance calculation
    landmark_variance = np.var(face_landmarks, axis=0)
    max_variance = (w * h) / 4
    variance_score = 1.0 - min(np.sum(landmark_variance) / max_variance, 1.0)
    
    final_quality = 0.8 * base_quality + 0.2 * variance_score
    
    print(f"\n📈 Quality Calculation:")
    print(f"   Base quality: {base_quality:.4f}")
    print(f"   Variance score: {variance_score:.4f}")
    print(f"   Final quality: {final_quality:.4f}")
    
    # Compare with scorer method
    scorer_quality = scorer.compute_face_quality(frame_rgb)
    print(f"   Scorer quality: {scorer_quality:.4f}")
    
    # Show mouth-specific calculations
    mouth_landmarks = face_landmarks[48:68]
    mouth_spread_x = np.std(mouth_landmarks[:, 0])
    mouth_spread_y = np.std(mouth_landmarks[:, 1])
    mouth_bbox = [
        np.min(mouth_landmarks[:, 0]), np.max(mouth_landmarks[:, 0]),
        np.min(mouth_landmarks[:, 1]), np.max(mouth_landmarks[:, 1])
    ]
    mouth_area = (mouth_bbox[1] - mouth_bbox[0]) * (mouth_bbox[3] - mouth_bbox[2])
    
    print(f"\n👄 Mouth Analysis:")
    print(f"   Mouth area: {mouth_area:.1f} pixels")
    print(f"   Mouth spread X: {mouth_spread_x:.1f}")
    print(f"   Mouth spread Y: {mouth_spread_y:.1f}")
    
    print(f"\n🔍 Issue Analysis:")
    if abs(face_confidence - scorer_quality) < 0.1:
        print("   ⚠️ Face quality and confidence are too similar!")
        print("   💡 This suggests the quality calculation is not detecting meaningful differences")
    else:
        print("   ✅ Face quality and confidence show meaningful differences")
        print(f"   📊 Difference: {abs(face_confidence - scorer_quality):.3f}")

if __name__ == "__main__":
    debug_face_quality() 