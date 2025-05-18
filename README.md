# Multi-Camera Multi-Object Tracking

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)  
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

## Project Overview
Our **Multi-Camera Multi-Object Tracking** is a comprehensive framework that extends single-camera tracking to a multi-camera setup. It leverages [ByteTrack](https://github.com/ifzhang/ByteTrack) for high-performance single-camera multi-object tracking, enhanced by our **Conflict-aware Cosine Tracking (CC)** algorithm. Once per-camera tracks are generated, the **Human Matching (HM) Algorithm** assigns global IDs by fusing 3D ground coordinates and appearance features across all views.

Key steps:
1. **Single-Camera Tracking:** ByteTrack + CC algorithm for robust ID assignment.
2. **3D Localization:** Compute ground-plane coordinates using intrinsic & extrinsic parameters.
3. **Global ID Association:** Fuse spatial and appearance cues with our HM Algorithm.

---

## Features
- **Enhanced Single-Camera Tracking**  
  - Integrates torchreid for appearance feature extraction.  
  - Implements CC algorithm to reduce ID switches.
- **3D Ground-Plane Estimation**  
  - Converts 2D detections to real-world coordinates.  
  - Supports arbitrary camera configurations (intrinsics + extrinsics).
- **Global Identity Management**  
  - HM Algorithm for consistent cross-view ID assignment.  
  - Balances spatial proximity and appearance similarity.

---

## Video demo
Watch my video demo here: https://www.youtube.com/watch?v=Uq-M_70Ip8Y

---

## Prerequisites
- **Python:** >= 3.8  
- **OpenCV:** 4.x  
- **NumPy**  
- **TensorRT** (optional, for accelerated inference)  
- **torchreid** (for feature embedding)

Install with pip:
```bash
pip install opencv-python numpy tensorRT torchreid
