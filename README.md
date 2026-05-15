# Hybrid Lid Orientation Estimation Pipeline
Hybrid computer vision pipeline for lid orientation estimation using segmentation, PCA, ellipse fitting, and keypoint-guided direction resolution. Zeon Assignment 1 submission by Jaywardhan Raghu (23B0737).

## Problem Statement

The objective of this assignment is to estimate the orientation angle of circular lids from RGB images.

Given an input image containing one or more lids, the system should:
- detect each lid
- estimate its center coordinates
- predict its orientation angle

The solution should generalize across varying lid positions, rotations, and image conditions.

## Methodology

### 1. Segmentation
### 2. Contour Extraction
### 3. Ellipse-Based Orientation
### 4. PCA-Based Orientation
### 5. Keypoint Detection
### 6. Direction Resolution
### 7. Hybrid Arbitration

Image
  ↓
Segmentation
  ↓
Contour Extraction
  ↓
Ellipse + PCA
  ↓
Keypoint Detection
  ↓
Direction Resolution
  ↓
Hybrid Selection
  ↓
Final Angle
