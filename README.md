# Inference Window

A Python-based computer vision project for window selection and face quality analysis using pose estimation and face detection.

## Features

- **Window Selection**: Intelligent window detection and selection using computer vision
- **Face Quality Analysis**: Face quality assessment and improvement algorithms
- **Pose Estimation**: Integration with PoseNet for pose detection
- **6D Pose Estimation**: 6DRepNet integration for 6D pose estimation

## Project Structure

```
inference_window/
├── window_selector.py          # Main window selection logic
├── debug_face_quality.py       # Face quality debugging utilities
├── test_face_quality.py        # Face quality testing framework
├── FACE_QUALITY_IMPROVEMENTS.md # Documentation for face quality improvements
├── 6DRepNet/                   # 6D pose estimation models
├── posenet-python/             # PoseNet implementation
├── data/                       # Data files and datasets
├── plots/                      # Generated plots and visualizations
└── venv/                       # Python virtual environment
```

## Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd inference_window
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Window Selection
```python
from window_selector import WindowSelector

# Initialize the window selector
selector = WindowSelector()

# Process an image
result = selector.process_image(image_path)
```

### Face Quality Analysis
```python
from debug_face_quality import FaceQualityAnalyzer

# Initialize the face quality analyzer
analyzer = FaceQualityAnalyzer()

# Analyze face quality
quality_score = analyzer.analyze_face(image_path)
```

## Dependencies

- OpenCV
- NumPy
- PyTorch
- MediaPipe
- Other dependencies (see requirements.txt)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Contact

[Add your contact information here]
