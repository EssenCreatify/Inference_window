#!/usr/bin/env python3
"""
Test script to demonstrate the new face quality detection feature.
This script shows how the face quality score can detect occlusion scenarios.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from window_selector import WindowScorer, init_face_detector

def create_test_scenarios():
    """Create synthetic test scenarios to demonstrate face quality detection."""
    
    # Initialize face detector
    init_face_detector()
    scorer = WindowScorer()
    
    # Create a simple test image (you would replace this with actual video frames)
    # This is just for demonstration - in practice you'd use real video frames
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    print("🧪 Testing Face Quality Detection")
    print("=" * 50)
    
    # Test 1: Normal face (should have high quality)
    print("\n1️⃣ Testing normal face scenario:")
    quality_score = scorer.compute_face_quality(test_image)
    confidence_score = scorer.compute_face_confidence(test_image)
    print(f"   Face Quality Score: {quality_score:.3f}")
    print(f"   Face Confidence Score: {confidence_score:.3f}")
    
    # Test 2: Simulate partial occlusion by creating a mask
    print("\n2️⃣ Testing partial occlusion scenario:")
    # Create a mask that covers part of the face
    occluded_image = test_image.copy()
    # Add a dark rectangle to simulate occlusion
    cv2.rectangle(occluded_image, (50, 50), (150, 150), (0, 0, 0), -1)
    
    quality_score_occluded = scorer.compute_face_quality(occluded_image)
    confidence_score_occluded = scorer.compute_face_confidence(occluded_image)
    print(f"   Face Quality Score (occluded): {quality_score_occluded:.3f}")
    print(f"   Face Confidence Score (occluded): {confidence_score_occluded:.3f}")
    
    print(f"\n📊 Quality Score Difference: {quality_score - quality_score_occluded:.3f}")
    print(f"📊 Confidence Score Difference: {confidence_score - confidence_score_occluded:.3f}")
    
    # Test 3: Show how the new scoring system would work
    print("\n3️⃣ New Scoring System Weights:")
    print(f"   Pose: {scorer.weights['pose']:.2f}")
    print(f"   Motion: {scorer.weights['motion']:.2f}")
    print(f"   Face Confidence: {scorer.weights['face']:.2f}")
    print(f"   Face Quality: {scorer.weights['face_quality']:.2f}")
    print(f"   Lighting: {scorer.weights['lighting']:.2f}")
    print(f"   Stability: {scorer.weights['stability']:.2f}")
    
    print("\n✅ Face Quality Detection Test Complete!")
    print("\n💡 Key Benefits:")
    print("   • Detects partial face occlusion")
    print("   • Prioritizes mouth area (35% weight) for speech analysis")
    print("   • Analyzes facial landmark completeness")
    print("   • Provides more robust face assessment")
    print("   • Helps avoid selecting windows with occluded faces")

if __name__ == "__main__":
    create_test_scenarios() 