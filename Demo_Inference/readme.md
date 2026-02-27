# BrightRate: HDR-UGC Video Quality Assessment
### Overview
BrightRate is a machine learning system for HDR video quality assessment that combines multiple feature extractors with a regression model to predict quality scores. This repository contains a modular, class-based implementation for processing HDR videos, extracting quality-relevant features, and predicting quality scores using pretrained models.

### Features
- **Multiple Feature Extractors:**
    - CONTRIQUE: Content-based quality features with temporal difference analysis
    - HDR: HDR-specific image quality metrics
    - CLIP: Visual semantic features from OpenAI's CLIP model

- **Robust Processing Framework:**
    - Support for both MP4 and YUV formats
    - 10-bit HDR video support
    - Frame sampling for faster processing

- **Parallel Processing Capabilities:**
    - MPI support for multi-node processing
    - Multi-core processing within nodes
    - Flexible data partitioning

- **Quality Assessment:**
    - Automated feature normalization and standardization
    - Pretrained SVR models for quality prediction
    - Feature dimension compatibility handling

## Installation
### Requirements
- Python 3.10
- PyTorch 1.8+
- NumPy, Pandas, SciPy
- mpi4py (for distributed processing)
- FFmpeg/FFprobe (for video processing)

### Setup

1. Clone the repository:
```bash
    git clone https://github.com/brightvqa/BrightVQ.git
    cd BrightVQ/Demo_Inference/
```

2. Create a conda environment:
```bash
    conda env create -f hdrvqa_env.yml
```

***Ensure FFmpeg is installed***

## Inference

**Basic Usage**
```python
    python demo_inference.py  --model_path ./models/brightrate_brightvq.pt  --dataset_csv ./sample_videos.csv --video_path ./sample_videos/ --save_path ./demo-feats/ --parallel_level video --num_workers -1 --num_frames 30 --ffmpeg_path /folder_to_ffmpeg_ffprobe/
```

**Using YUV files**
```python
   python demo_inference.py  --model_path ./models/brightrate_brightvq.pt  --read_yuv --dataset_csv ./sample_videos.csv --video_path ./sample_videos/ --save_path ./demo-feats/ --parallel_level video --num_workers -1 --num_frames 30 --ffmpeg_path /folder_to_ffmpeg_ffprobe/
```

**Parallel Processing**
Run on multiple nodes with MPI:
```bash
    mpirun -n 4 python demo_inference.py --model_path ./models/brightrate_brightvq.pt --dataset_csv ./sample_videos.csv --video_path ./sample_videos/ --save_path ./demo-feats/ --parallel_level video --num_workers -1 --num_frames 30 --ffmpeg_path /folder_to_ffmpeg_ffprobe/
```

### Input Format
The input CSV file should contain video information with the following columns:

| Column | Description | Required for YUV |
|--------|-------------|------------------|
| Video | Filename without extension | Always required |
| width | Video width in pixels | Required |
| height | Video height in pixels | Required |
| name | Display name (optional) | Optional |

**Example CSV:**
```csv
    Video,width,height
    video1,1920,1080
    video2,3840,2160
```

