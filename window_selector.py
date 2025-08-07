import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sixdrepnet import SixDRepNet
from dataclasses import dataclass
from typing import List, Tuple, Dict

# torchlm imports for face cropping
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet

@dataclass
class WindowMetrics:
    pose_score: float
    motion_score: float
    face_confidence: float
    face_quality: float  # New metric for face occlusion detection
    lighting_score: float
    stability_score: float
    final_score: float
    start_time: float
    end_time: float

@dataclass
class FrameMetrics:
    time: float
    pose_score: float
    motion_score: float
    face_confidence: float
    face_quality: float  # New metric for face occlusion detection
    lighting_score: float
    stability_score: float
    final_score: float

class WindowScorer:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'pose': 0.25,
            'motion': 0.15,
            'face': 0.15,
            'face_quality': 0.15,  # New weight for face quality
            'lighting': 0.15,
            'stability': 0.15
        }
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    def compute_lighting_score(self, frame: np.ndarray) -> float:
        # Convert to grayscale and calculate histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate standard deviation of pixel intensities
        std_dev = np.std(gray)
        
        # Penalize both too dark and too bright images
        mean_intensity = np.mean(gray)
        optimal_intensity = 127
        intensity_score = 1 - abs(mean_intensity - optimal_intensity) / 127
        
        # Combine with contrast score from std_dev
        contrast_score = min(std_dev / 50, 1.0)  # Normalize std_dev
        
        return 0.5 * intensity_score + 0.5 * contrast_score

    def compute_pose_score(self, pose: Tuple[float, float, float]) -> float:
        # Inverse of average absolute angles, normalized by 90 degrees
        pitch, yaw, roll = pose
        return 1.0 - (abs(pitch) + abs(yaw) + abs(roll)) / (3 * 90.0)

    def compute_motion_score(self, motion: float) -> float:
        # Inverse of motion, normalized by a factor
        return 1.0 - min(motion / 10.0, 1.0)

    def compute_stability_score(self, poses: List[Tuple[float, float, float]]) -> float:
        if len(poses) < 2:
            return 0.0
            
        # Calculate the variance of pose angles
        poses_array = np.array(poses)
        angle_variances = np.var(poses_array, axis=0)
        
        # Penalize high variance in any angle
        max_acceptable_variance = 100  # Threshold for maximum acceptable variance
        stability_scores = np.exp(-angle_variances / max_acceptable_variance)
        
        return np.mean(stability_scores)

    def compute_face_confidence(self, frame: np.ndarray) -> float:
        landmarks, bb = torchlm.runtime.forward(frame)
        if bb.shape[0] == 0:
            return 0.0
        # Use the confidence score from the face detector
        return float(bb[0][4])  # Assuming the last value in bb is confidence

    def compute_face_quality(self, frame: np.ndarray) -> float:
        """
        Compute face quality score based on landmark confidence and face completeness.
        This helps detect when parts of the face are occluded or low quality.
        """
        landmarks, bb = torchlm.runtime.forward(frame)
        if bb.shape[0] == 0:
            return 0.0
        
        # Get landmarks for the first detected face
        face_landmarks = landmarks[0]  # Shape: (68, 2) for 68 landmarks
        
        # For cropped face images, we need a different approach since all landmarks are visible
        # Instead of checking visibility, we'll analyze landmark confidence and distribution
        
        # 1. Check landmark confidence (if available) - this indicates detection quality
        # Since we don't have per-landmark confidence, we'll use the overall face confidence
        face_confidence = float(bb[0][4])
        
        # 2. Analyze landmark distribution quality
        h, w = frame.shape[:2]
        
        # Check if landmarks are well-distributed (not clustered in one area)
        landmark_std_x = np.std(face_landmarks[:, 0])
        landmark_std_y = np.std(face_landmarks[:, 1])
        
        # Normalize by frame size
        normalized_std_x = landmark_std_x / w
        normalized_std_y = landmark_std_y / h
        
        # Good distribution should have reasonable spread (not too small, not too large)
        distribution_score = min(normalized_std_x, normalized_std_y) * 4  # Scale to [0,1]
        distribution_score = min(distribution_score, 1.0)
        
        # 3. Check for landmark clustering (occlusion often causes landmarks to cluster)
        # Calculate average distance between landmarks
        distances = []
        for i in range(len(face_landmarks)):
            for j in range(i+1, len(face_landmarks)):
                dist = np.linalg.norm(face_landmarks[i] - face_landmarks[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        max_possible_distance = np.sqrt(w*w + h*h)
        distance_score = min(avg_distance / max_possible_distance * 2, 1.0)  # Scale to [0,1]
        
        # 4. Check for unusual landmark patterns (e.g., landmarks too close together)
        # Calculate minimum distance between landmarks
        min_distance = np.min(distances) if distances else 0
        min_distance_score = min(min_distance / 50.0, 1.0)  # Penalize if landmarks are too close
        
        # 5. Mouth area specific check (most important for speech)
        mouth_landmarks = face_landmarks[48:68]  # Mouth landmarks
        
        # Calculate mouth area and spread
        mouth_spread_x = np.std(mouth_landmarks[:, 0])
        mouth_spread_y = np.std(mouth_landmarks[:, 1])
        
        # Calculate mouth area (approximate)
        mouth_bbox = [
            np.min(mouth_landmarks[:, 0]), np.max(mouth_landmarks[:, 0]),
            np.min(mouth_landmarks[:, 1]), np.max(mouth_landmarks[:, 1])
        ]
        mouth_area = (mouth_bbox[1] - mouth_bbox[0]) * (mouth_bbox[3] - mouth_bbox[2])
        
        # Check for mouth collapse (occlusion often reduces mouth area)
        expected_mouth_area = 2000  # Approximate expected mouth area in pixels
        mouth_area_score = min(mouth_area / expected_mouth_area, 1.0)
        
        # Check for mouth spread (occlusion often reduces spread)
        mouth_spread_score = min((mouth_spread_x + mouth_spread_y) / 60.0, 1.0)
        
        # Combined mouth quality (area and spread)
        mouth_quality = 0.6 * mouth_area_score + 0.4 * mouth_spread_score
        
        # Combine all scores with much higher focus on mouth area and occlusion detection
        final_quality = (
            0.1 * face_confidence +      # Reduced weight for overall detection confidence
            0.15 * distribution_score +  # Landmark distribution
            0.15 * distance_score +      # Average landmark distance
            0.1 * min_distance_score +   # Minimum landmark distance
            0.5 * mouth_quality          # Much higher weight for mouth area quality
        )
        
        return float(final_quality)

    def compute_window_metrics(self, 
                             poses: List[Tuple[float, float, float]],
                             motions: List[float],
                             face_confidences: List[float],
                             face_qualities: List[float],
                             lighting_scores: List[float],
                             times: List[float]) -> WindowMetrics:
        
        # Compute pose score (inverse of average absolute angles)
        pose_angles = np.abs(poses)
        pose_score = 1.0 - np.mean(pose_angles) / 90.0  # Normalize by 90 degrees
        
        # Motion score (inverse of average motion)
        # We normalize by a factor (e.g., 10.0) to scale raw motion into a [0,1] range.
        motion_score = 1.0 - min(np.mean(motions) / 10.0, 1.0)
        
        # Face detection confidence
        face_confidence = np.mean(face_confidences)
        
        # Face quality (occlusion detection)
        face_quality = np.mean(face_qualities)
        
        # Lighting conditions
        lighting_score = np.mean(lighting_scores)
        
        # Pose stability
        stability_score = self.compute_stability_score(poses)
        
        # Compute final weighted score
        final_score = (
            self.weights['pose'] * pose_score +
            self.weights['motion'] * motion_score +
            self.weights['face'] * face_confidence +
            self.weights['face_quality'] * face_quality +
            self.weights['lighting'] * lighting_score +
            self.weights['stability'] * stability_score
        )
        
        return WindowMetrics(
            pose_score=pose_score,
            motion_score=motion_score,
            face_confidence=face_confidence,
            face_quality=face_quality,
            lighting_score=lighting_score,
            stability_score=stability_score,
            final_score=final_score,
            start_time=times[0],
            end_time=times[-1]
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--output_plot_path', type=str, default='score_plot.png', help='Base name for plot outputs')
    parser.add_argument('--weights', type=str, help='JSON string of weights for different metrics')
    return parser.parse_args()

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

def crop_faces_from_video(video_path, frame_skip=2, avnet_width=224, avnet_height=224, expanded_ratio=0.6):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_indices = []
    index = 0

    # Get video FPS, default to 30 if not available
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("⚠️ Warning: Could not detect video FPS, using default value of 30")
        fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index % frame_skip == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_indices.append(index)
        index += 1
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    bboxes = []
    for i, frame in enumerate(frames):
        landmarks, bb = torchlm.runtime.forward(frame)
        if bb.shape[0] == 0:
            height, width = frame.shape[:2]
            bb = np.array([[0, 0, width, height, 0]])
        elif bb.shape[0] > 1 and i > 0:
            last_bb = bboxes[-1]
            bb = np.array([min(bb, key=lambda b: np.linalg.norm(b[:2] - last_bb[:2]))])
        bboxes.append(bb[0])

    cropped_frames = []
    for frame, bbox in zip(frames, bboxes):
        x1, y1, x2, y2 = bbox[:4]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop_w = int(w * (1 + expanded_ratio))
        crop_h = int(h * (1 + expanded_ratio))
        x1 = max(0, int(cx - crop_w / 2))
        y1 = max(0, int(cy - crop_h / 2))
        x2 = min(frame.shape[1], x1 + crop_w)
        y2 = min(frame.shape[0], y1 + crop_h)
        cropped = frame[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (avnet_width, avnet_height))
        cropped_frames.append(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    return cropped_frames, frame_indices, fps

def load_pose_model(gpu_id):
    return SixDRepNet(gpu_id=gpu_id)

def compute_motion(prev_gray, gray, use_gpu):
    if use_gpu:
        gpu_prev = cv2.cuda_GpuMat()
        gpu_curr = cv2.cuda_GpuMat()
        gpu_prev.upload(prev_gray)
        gpu_curr.upload(gray)
        flow = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
        flow_map = flow.calc(gpu_prev, gpu_curr, None)
        flow_xy = flow_map.download()
    else:
        flow_xy = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow_xy[..., 0], flow_xy[..., 1])
    return np.mean(mag)

def process_video(cropped_frames, frame_indices, video_fps, pose_model):
    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
    yaw_list, pitch_list, roll_list, motion_list = [], [], [], []
    face_confidences_list, face_qualities_list, lighting_scores_list = [], [], []
    prev_gray = None
    poses = []  # Store all poses for stability calculation
    scorer = WindowScorer()
    
    # For temporal consistency check
    face_quality_history = []
    # For temporal jitter check
    mouth_landmarks_history = []

    print("\n📊 Processing video frames...")
    for i, frame in enumerate(tqdm(cropped_frames, desc="Frame analysis")):
        resized = cv2.resize(frame, (224, 224))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Pose and motion
        pitch, yaw, roll = pose_model.predict(resized)
        motion = compute_motion(prev_gray, gray, use_gpu) if prev_gray is not None else 0
        prev_gray = gray

        poses.append((pitch, yaw, roll))
        yaw_list.append(yaw)
        pitch_list.append(pitch)
        roll_list.append(roll)
        motion_list.append(motion)

        # Other metrics
        face_confidences_list.append(scorer.compute_face_confidence(frame))
        current_face_quality = scorer.compute_face_quality(frame)
        face_qualities_list.append(current_face_quality)
        face_quality_history.append(current_face_quality)
        lighting_scores_list.append(scorer.compute_lighting_score(frame))

        # Save mouth landmarks for jitter analysis
        landmarks, bb = torchlm.runtime.forward(frame)
        if bb.shape[0] > 0:
            mouth_landmarks = landmarks[0][48:68]  # (20, 2)
            mouth_landmarks_history.append(mouth_landmarks)
        else:
            mouth_landmarks_history.append(None)

    times = [idx / video_fps for idx in frame_indices]
    
    # Create window scorer
    scorer = WindowScorer()
    
    # Calculate per-frame scores
    frame_metrics = []
    stability_window_size = int(video_fps)  # 1-second window for stability
    jitter_window = 5  # Number of frames for jitter analysis

    for i in range(len(times)):
        pose_score = scorer.compute_pose_score(poses[i])
        motion_score = scorer.compute_motion_score(motion_list[i])
        face_confidence = face_confidences_list[i]
        face_quality = face_qualities_list[i]
        lighting_score = lighting_scores_list[i]

        # Apply temporal jitter penalty to face quality
        # Compute variance of mouth landmark positions over a short window
        jitter_start = max(0, i - jitter_window // 2)
        jitter_end = min(len(mouth_landmarks_history), i + jitter_window // 2 + 1)
        window_mouth_landmarks = [ml for ml in mouth_landmarks_history[jitter_start:jitter_end] if ml is not None]
        if len(window_mouth_landmarks) > 1:
            # Stack to (window, 20, 2)
            stacked = np.stack(window_mouth_landmarks, axis=0)
            # Compute variance across window for each landmark
            jitter = np.mean(np.var(stacked, axis=0))  # Mean variance across all mouth landmarks
            # Heuristic: if jitter is high, penalize face quality
            if jitter > 10.0:  # Threshold may need tuning
                face_quality *= 0.5  # Penalize more if jitter is high

        # Apply temporal consistency adjustment to face quality
        if i > 0 and i < len(face_quality_history) - 1:
            prev_quality = face_quality_history[i-1]
            next_quality = face_quality_history[i+1] if i+1 < len(face_quality_history) else face_quality
            avg_surrounding_quality = (prev_quality + next_quality) / 2
            if face_quality < avg_surrounding_quality * 0.8:
                face_quality *= 0.7

        # For stability, use a sliding window over poses
        start = max(0, i - stability_window_size // 2)
        end = min(len(poses), i + stability_window_size // 2 + 1)
        stability_score = scorer.compute_stability_score(poses[start:end])

        final_score = (
            scorer.weights['pose'] * pose_score +
            scorer.weights['motion'] * motion_score +
            scorer.weights['face'] * face_confidence +
            scorer.weights['face_quality'] * face_quality +
            scorer.weights['lighting'] * lighting_score +
            scorer.weights['stability'] * stability_score
        )

        frame_metrics.append(FrameMetrics(
            time=times[i],
            pose_score=float(pose_score),
            motion_score=float(motion_score),
            face_confidence=float(face_confidence),
            face_quality=float(face_quality),
            lighting_score=float(lighting_score),
            stability_score=float(stability_score),
            final_score=float(final_score)
        ))

    # Process in sliding windows to find the best window
    window_size = int(30 * video_fps / 2)
    if window_size >= len(cropped_frames):
        print(f"\n⚠️ Warning: Video is shorter than 30 seconds, using entire video as window")
        window_size = len(cropped_frames)
    
    best_metrics = None
    best_start = -1
    
    print("\n🔍 Analyzing windows...")
    for i in tqdm(range(0, len(cropped_frames) - window_size + 1), desc="Window analysis"):
        # Calculate window average scores from per-frame scores
        window_frame_metrics = frame_metrics[i:i + window_size]
        
        avg_pose_score = np.mean([m.pose_score for m in window_frame_metrics])
        avg_motion_score = np.mean([m.motion_score for m in window_frame_metrics])
        avg_face_confidence = np.mean([m.face_confidence for m in window_frame_metrics])
        avg_face_quality = np.mean([m.face_quality for m in window_frame_metrics])
        avg_lighting_score = np.mean([m.lighting_score for m in window_frame_metrics])
        avg_stability_score = np.mean([m.stability_score for m in window_frame_metrics])

        final_score = (
            scorer.weights['pose'] * avg_pose_score +
            scorer.weights['motion'] * avg_motion_score +
            scorer.weights['face'] * avg_face_confidence +
            scorer.weights['face_quality'] * avg_face_quality +
            scorer.weights['lighting'] * avg_lighting_score +
            scorer.weights['stability'] * avg_stability_score
        )
        
        current_metrics = WindowMetrics(
            pose_score=avg_pose_score,
            motion_score=avg_motion_score,
            face_confidence=avg_face_confidence,
            face_quality=avg_face_quality,
            lighting_score=avg_lighting_score,
            stability_score=avg_stability_score,
            final_score=final_score,
            start_time=times[i],
            end_time=times[i + window_size - 1] if i + window_size <= len(times) else times[-1]
        )

        if best_metrics is None or current_metrics.final_score > best_metrics.final_score:
            best_metrics = current_metrics
            best_start = i

    if best_metrics is None:
        print("\n❌ Error: Could not find any valid windows in the video")
        return (yaw_list, pitch_list, roll_list, motion_list, times, 
                0, len(cropped_frames), frame_metrics)

    best_end = best_start + window_size
    print("\n✅ Best window metrics:")
    print(f"🕒 Time: {best_metrics.start_time:.2f}s to {best_metrics.end_time:.2f}s")
    print(f"📊 Scores:")
    print(f"  - Pose: {best_metrics.pose_score:.3f}")
    print(f"  - Motion: {best_metrics.motion_score:.3f}")
    print(f"  - Face Confidence: {best_metrics.face_confidence:.3f}")
    print(f"  - Face Quality: {best_metrics.face_quality:.3f}")
    print(f"  - Lighting: {best_metrics.lighting_score:.3f}")
    print(f"  - Stability: {best_metrics.stability_score:.3f}")
    print(f"  - Final Score: {best_metrics.final_score:.3f}")

    return (yaw_list, pitch_list, roll_list, motion_list, times, 
            best_start, best_end, frame_metrics)

def plot_all_metrics(times, yaws, pitches, rolls, motions, 
                     best_start_idx, best_end_idx, 
                     frame_metrics, output_path):
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    fig, axs = plt.subplots(11, 1, figsize=(16, 30), sharex=True)
    fig.suptitle('Comprehensive Inference Window Analysis', fontsize=16)

    best_window_start_time = times[best_start_idx]
    best_window_end_time = times[min(best_end_idx-1, len(times)-1)]
    full_time_range = (times[0], times[-1])

    def plot_frame_metric(ax, data, label, color):
        ax.plot(times, data, color=color, label=label, alpha=0.7)
        ax.axvspan(best_window_start_time, best_window_end_time, color='green', alpha=0.1, label='Best Window')
        ax.set_ylabel(label)
        ax.grid(True)
        ax.set_xlim(full_time_range)

    plot_frame_metric(axs[0], yaws, "Yaw (°)", 'orange')
    plot_frame_metric(axs[1], pitches, "Pitch (°)", 'green')
    plot_frame_metric(axs[2], rolls, "Roll (°)", 'red')
    plot_frame_metric(axs[3], motions, "Raw Motion", 'purple')

    def plot_window_metric(ax, data, label, color):
        if not times: return
        ax.plot(times, data, color=color, label=label, alpha=0.7)
        ax.axvspan(best_window_start_time, best_window_end_time, color='green', alpha=0.1)
        ax.set_ylabel(label)
        ax.grid(True)
        ax.set_xlim(full_time_range)
    
    if frame_metrics:
        plot_window_metric(axs[4], [m.pose_score for m in frame_metrics], "Pose Score", 'saddlebrown')
        plot_window_metric(axs[5], [m.stability_score for m in frame_metrics], "Stability Score", 'maroon')
        plot_window_metric(axs[6], [m.motion_score for m in frame_metrics], "Motion Score", 'darkviolet')
        plot_window_metric(axs[7], [m.face_confidence for m in frame_metrics], "Face Confidence", 'cyan')
        plot_window_metric(axs[8], [m.face_quality for m in frame_metrics], "Face Quality", 'orange')
        plot_window_metric(axs[9], [m.lighting_score for m in frame_metrics], "Lighting Score", 'magenta')
        plot_window_metric(axs[10], [m.final_score for m in frame_metrics], "Final Score", 'blue')

    axs[10].set_xlabel("Time (s)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path)
    print(f"\n📊 Combined plot saved to: {output_path}")

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected! Running on CPU (this may be slower).")
    else:
        print("✅ GPU detected! Using CUDA acceleration.")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Set up output plot path based on video name
    if not args.output_plot_path or args.output_plot_path == 'score_plot.png':
        video_filename = os.path.basename(args.video_path)
        video_name_without_ext = os.path.splitext(video_filename)[0]
        output_plot_path = os.path.join('plots', f"{video_name_without_ext}_metrics.png")
    else:
        output_plot_path = args.output_plot_path

    init_face_detector()
    cropped_frames, frame_indices, video_fps = crop_faces_from_video(args.video_path)

    pose_model = load_pose_model(gpu_id=args.gpu if torch.cuda.is_available() else -1)
    (yaws, pitches, rolls, motions, times, 
     best_start, best_end, 
     frame_metrics) = process_video(
        cropped_frames, frame_indices, video_fps, pose_model
    )

    # Log per-frame metrics
    log_path = os.path.join(os.path.dirname(output_plot_path), f"{os.path.splitext(os.path.basename(output_plot_path))[0]}_log.csv")
    with open(log_path, 'w') as f:
        f.write("time,yaw,pitch,roll,raw_motion,pose_score,motion_score,face_confidence,face_quality,lighting_score,stability_score,final_score\n")
        for i, metrics in enumerate(frame_metrics):
            f.write(f"{metrics.time:.4f},{float(yaws[i]):.4f},{float(pitches[i]):.4f},{float(rolls[i]):.4f},{float(motions[i]):.4f},"
                    f"{metrics.pose_score:.4f},{metrics.motion_score:.4f},{metrics.face_confidence:.4f},{metrics.face_quality:.4f},"
                    f"{metrics.lighting_score:.4f},{metrics.stability_score:.4f},{metrics.final_score:.4f}\n")
    print(f"📝 Per-frame metrics logged to: {log_path}")

    plot_all_metrics(times, yaws, pitches, rolls, motions, 
                     best_start, best_end, 
                     frame_metrics, output_plot_path)

if __name__ == "__main__":
    main()
