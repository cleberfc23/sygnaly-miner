# Sygnały - Public Signals Mining 
![Status](https://img.shields.io/badge/status-active%20development-yellow?style=for-the-badge)

An exploration of how machine learning can discover meaningful signals in large collections of public text.

## The Problem

Large volumes of public text contain valuable information about events, trends, and emerging issues.  
However, extracting structure from thousands of documents is difficult without automated analysis.

## The Approach

This project explores how machine learning can transform raw text into structured signals by creating semantic representations and discovering recurring themes through unsupervised clustering.

## Engineering Focus

The goal is to build a clear and reproducible machine learning workflow that evolves from data exploration into a deployable system with testing, automation, and containerized environments.

## Impact

By uncovering patterns in large text collections, organizations can better understand emerging topics and hidden structures in public information streams.

## Dataset

The methodology is demonstrated using a corpus of Polish news articles containing:

- **248,123 documents**
- **4 fields:** `title`, `headline`, `content`, `link`
- full article text written in Polish

Dataset source:  
https://huggingface.co/datasets/WiktorS/polish-news

## Current Progress

The initial data exploration and preprocessing phase has been completed.

- Created a **3,000-sample development dataset** (~1.21% of total)
- Selected `content` as the primary field for NLP analysis
- Removed missing, empty, and duplicate entries
- Applied initial text cleaning to eliminate boilerplate patterns
- Filtered low-quality documents (<100 characters)
- Final dataset: **2,890 high-quality documents (~96.3% retained)**

This cleaned dataset establishes a reliable foundation for the next stages of the pipeline, including vectorization and clustering.
