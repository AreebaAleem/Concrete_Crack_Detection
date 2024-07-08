# Concrete_Crack_Detection

A comprehensive Concrete Crack Detection system developed using various machine learning models. The system aims to automatically identify and classify concrete cracks in images, facilitating early detection and maintenance in infrastructure management.

## Dataset Description

The dataset used for this task consists of 10,000 images sourced from [this link](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip). It includes 5,000 images each of concrete with cracks and concrete without cracks.

## Purpose

The goal of this project is to leverage machine learning models to predict whether an input image contains a crack (positive class) or does not contain a crack (negative class). This predictive capability aids in proactive maintenance and safety assessments for concrete structures.

## Comparison of Machine Learning Algorithms

| #   | Machine Learning Algorithm | Accuracy (%) | Efficiency (Sec) | Comments                                 |
| --- | -------------------------- | ------------ | ---------------- | ----------------------------------------- |
| 1   | KNN                        | 81.21%       | 250.70 sec       | KNN demonstrates good accuracy, achieving 81.21%, but its efficiency is comparatively lower. |
| 2   | SVM                        | 92.87%       | 269.75 sec       | SVM achieves high accuracy, with an impressive 92.87%, matching the performance of more complex algorithms. |
| 3   | Deep Learning (CNN)        | 95.12%       | 480.23 sec       | CNN achieves superior accuracy, reaching 95.12%, with increased computational time due to its deeper architecture. |
| 4   | Transfer Learning (AlexNet)| 93.45%       | 315.60 sec       | Transfer learning with AlexNet achieves competitive accuracy at 93.45%, leveraging pre-trained weights for faster convergence. |

