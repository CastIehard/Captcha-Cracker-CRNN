# End-to-End CAPTCHA Solver

This project implements an end-to-end system to solve synthetic alphanumeric CAPTCHAs using classical and deep learning-based approaches.

## Overview

The goal is to predict the character sequence from a distorted CAPTCHA image, handling variations such as:
- Rotation
- Color distortions
- Font style changes
- Background noise

Have fun experimenting with different models and techniques!

## Getting started
First step is to download the data from the [course materials](https://utn.instructure.com/courses/275/files/9592/download?download_frd=1).

## How to run the model?
Model training code: `main.py`

Model weights saved in: `crnn_captcha.pth`

Predictions for test images: `predictions_part2.json`