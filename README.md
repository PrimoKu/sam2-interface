# SAM2 Video Annotation Tool

A powerful tool for semi-automated video segmentation and annotation built on Meta AI's Segment Anything Model 2 (SAM2).

<!-- <video src="./demo/sam2_demo.mp4" controls="controls" muted="muted" style="max-width:100%;">
</video> -->

## Features

- Interactive object segmentation with point-based annotation
- Automatic mask propagation through video frames 
- Object tracking with automatic re-identification
- COCO format export for machine learning datasets
- Support for loading and updating existing COCO annotations
- Frame-by-frame navigation and editing
- Multi-object support with custom categories

## Requirements

- Python 3.10
- CUDA-capable GPU (recommended)
- SAM2 checkpoint

## Installation

1. Clone this repository with submodules:
   ```bash
   git clone --recursive https://github.com/PrimoKu/sam2-interface.git
   cd sam2-interface
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate sam2
   ```

3. Set required environment variables:
   ```bash
   export QT_PLUGIN_PATH=/path/to/envs/sam2/lib/python3.10/site-packages/PyQt5/Qt5/plugins
   export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1/extras/CUPTI/lib64:/path/to/envs/sam2/lib/python3.10/site-packages/PyQt5/Qt5/lib:$LD_LIBRARY_PATH
   ```

4. Download the SAM2 checkpoint:
   ```bash
   mkdir -p external/sam2/checkpoints
   # Download sam2.1_hiera_large.pt to external/sam2/checkpoints/
   ```

## Usage

1. Run the application:
   ```bash
   cd ui
   python main.py
   ```

2. Load video frames:
   - Click "Load Video" and select a directory containing video frames
   - Frames should be named with sequential numbers (e.g., 1.jpg, 2.jpg, etc.)

3. Annotate objects:
   - Click "New Object" to create a new object
   - Left-click to add positive points (include in mask)
   - Right-click to add negative points (exclude from mask)

4. Propagate masks:
   - Click "Propagate Masks" to automatically track objects through frames
   - Use arrow keys or "Prev Frame"/"Next Frame" buttons to navigate

5. Export annotations:
   - Click "Start COCO Export" to set up COCO file
   - Navigate through frames to export individual frames
   - Or use "Propagate and Export All" for batch processing

## Directory Structure

```
/
├── data/                  # Input video frames (not tracked by git)
├── output/                # Output COCO files (not tracked by git)
├── external/              
│   └── sam2/              # SAM2 submodule
├── ui/                    # UI application code
└── environment.yml        # Conda environment specification
```

## UI Components

- **Left Panel**: Control buttons for workflow steps
- **Center Panel**: Main display with annotation interface
- **Right Panel**: Object table with categories and tracking toggles

## Advanced Features

### Object Tracking Control

Each object can have tracking enabled or disabled:
- **Enabled**: Object will be automatically tracked and segmented in new frames
- **Disabled**: Object mask will be copied without adaptation

### Loading Existing Annotations

The tool can load existing COCO format annotations:
- Use "Load COCO JSON" to load annotations for the entire video
- Use "Load Current Frame COCO" to load annotations for just the current frame

### Frame-Specific Operations

- "Save Current Frame COCO" exports only the current frame
- "Load Current Frame COCO" imports annotations for the current frame

## Troubleshooting

- If you encounter UI rendering issues, verify the QT_PLUGIN_PATH is set correctly
- For CUDA errors, check that the correct CUDA version is installed (12.1 recommended)
- If segmentation quality is poor, try adding more points or adjusting object tracking
