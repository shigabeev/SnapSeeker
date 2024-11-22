# Image Processing and Collection Pipeline

A comprehensive tool for automated image collection, processing, and quality control designed for model training datasets.

![SnapSeeker Logo](https://raw.githubusercontent.com/shigabeev/SnapSeeker/refs/heads/main/logo.webp)

## Overview

This project combines image processing tools to streamline the collection and curation of training data. It addresses the bottleneck in gathering high-quality images for model training by automating collection, filtering, and quality control processes.

## Features

### Image Collection
- Automated image downloading from:
  - Google Images
  - Pexels
  - Unsplash
- Query-based collection with smart suggestions
- Automatic S3 upload integration

### Quality Control
- [x] Resolution Check
- [x] Face Detection and Analysis
- [x] Face focus detection
#### Image Quality Metrics
- [x] Noise/grain detection
- [x] Color/grayscale classification
- [ ] Border line detection 
- [ ] JPEG artifact detection
- [x] Focus quality assessment

### Smart Processing
- [ ] LLM-based image tagging
- [ ] Demographic balancing
- [ ] Automated query enhancement suggestions
- [ ] Super-resolution for low-res images
- [ ] Watermark removal
- [ ] Duplicate detection

### User Interface
- [ ] Gallery-style image review interface
- [ ] Batch operations (select, tag, delete)
- [ ] Guided filtering suggestions
- [ ] Query optimization recommendations

## How to launch
1. Create an .env file with your API keys and S3 bucket name
2. Install the requirements `pip install -r requirements.txt`
3. Run `python app.py`

