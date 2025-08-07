# Face Quality Detection Improvements

## Problem Statement

The original inference window selection system had 5 scoring metrics but struggled with scenarios where something suddenly covers part of the face for a brief moment. The existing `face_confidence` score only used the face detector's confidence, which might remain high even when parts of the face are occluded.

## Solution: Added Face Quality Score

Instead of replacing the existing face confidence score, we **added a new "Face Quality Score"** that specifically addresses occlusion detection and face completeness.

### New Scoring System (6 metrics total)

| Metric | Weight | Description |
|--------|--------|-------------|
| **Pose Score** | 25% | Head pose angles (pitch, yaw, roll) |
| **Motion Score** | 15% | Optical flow motion |
| **Face Confidence** | 15% | Face detection confidence |
| **Face Quality** | 15% | **NEW**: Occlusion detection & face completeness |
| **Lighting Score** | 15% | Image intensity and contrast |
| **Stability Score** | 15% | Pose variance over time |

## Face Quality Detection Features

The new `compute_face_quality()` method analyzes:

### 1. Facial Region Visibility
- **Eyes** (left & right): 10% each
- **Nose**: 15%
- **Mouth**: 35% (prioritized for speech analysis)
- **Face Contour**: 15%
- **Eyebrows** (left & right): 7.5% each

### 2. Landmark Completeness
- Checks if all 68 facial landmarks are within frame bounds
- Calculates visibility percentage for each facial region
- Weights regions by importance for face analysis

### 3. Occlusion Detection
- **Landmark Variance Analysis**: Detects unusual landmark distributions
- **Region-based Scoring**: Different facial regions weighted by importance (mouth prioritized)

### 4. Quality Score Components
```
Final Quality = 0.8 × Base Quality + 0.2 × Variance Score
```

Where:
- **Base Quality**: Weighted average of region visibility scores (mouth prioritized at 35%)
- **Variance Score**: Penalizes unusual landmark distributions

## Benefits

### ✅ Improved Occlusion Detection
- Detects when hands, objects, or other faces partially cover the subject
- **Prioritizes mouth area** to ensure speech analysis quality
- Provides more granular face assessment than simple confidence scores

### ✅ Better Window Selection
- Avoids selecting time windows with occluded faces
- Maintains high-quality face visibility throughout selected windows

### ✅ Robust Analysis
- Works alongside existing metrics without conflicts
- Provides additional context for face quality assessment

### ✅ Backward Compatibility
- Existing face confidence score is preserved
- New metric adds complementary information

## Usage

The face quality score is automatically computed and integrated into the scoring system. No changes to your existing workflow are required.

### Example Output
```
✅ Best window metrics:
🕒 Time: 5.20s to 35.20s
📊 Scores:
  - Pose: 0.823
  - Motion: 0.756
  - Face Confidence: 0.912
  - Face Quality: 0.847  ← NEW metric
  - Lighting: 0.734
  - Stability: 0.891
  - Final Score: 0.827
```

## Testing

Run the test script to see the face quality detection in action:

```bash
python test_face_quality.py
```

This will demonstrate how the face quality score responds to different occlusion scenarios.

## Implementation Details

### Key Changes Made:
1. **Added `face_quality` field** to `WindowMetrics` and `FrameMetrics` dataclasses
2. **Updated weights** to accommodate the new metric (reduced pose from 30% to 25%)
3. **Implemented `compute_face_quality()`** method with comprehensive occlusion detection
4. **Updated all scoring calculations** to include face quality
5. **Enhanced plotting** to show face quality over time
6. **Updated CSV logging** to include face quality data

### Files Modified:
- `window_selector.py` - Main implementation
- `test_face_quality.py` - Test script (new)
- `FACE_QUALITY_IMPROVEMENTS.md` - This documentation (new)

## Future Enhancements

Potential improvements for the face quality detection:

1. **Temporal Consistency**: Check face quality consistency over time windows
2. **Advanced Occlusion Patterns**: Detect specific types of occlusion (hands, objects, etc.)
3. **Adaptive Weights**: Adjust face quality weight based on video characteristics
4. **Machine Learning**: Train a model to better classify face quality scenarios

## Conclusion

The new face quality score significantly improves the system's ability to handle occlusion scenarios while maintaining all existing functionality. It provides a more comprehensive assessment of face visibility and quality, leading to better inference window selection. 