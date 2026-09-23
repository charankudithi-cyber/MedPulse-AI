#!/bin/bash
# Directory and package structure scaffolding script for MedPulse-AI

mkdir -p src
mkdir -p data
mkdir -p research
mkdir -p templates
mkdir -p static

touch src/__init__.py
touch src/helper.py
touch src/prompt.py
touch research/trials.ipynb
touch store_index.py
touch app.py

echo "[+] MedPulse-AI project architecture scaffolded successfully."
