# Zero-Shot Classification through Attribute-Driven Prototype Estimation

## Overview

This project focuses on implementing zero-shot classification using attribute-driven prototype estimation. The goal is to classify data from unseen classes by constructing prototypes from known data, enabling effective classification without the need for retraining.

## Project Structure

- **Prototype Construction**: Prototypes are built using attribute information from the dataset. These prototypes act as class centers in the feature space, crucial for the zero-shot classification process.
  
- **Classification Process**: Classification is conducted by measuring the distance between a sample and the constructed prototypes, with the class being assigned based on proximity to the nearest prototype.

## Dataset

The dataset used for this project can be accessed through the following link:
[Dataset Download](https://tinyurl.com/cs771-a23-hw1dat)

## Dependencies

- Python 3.x
- NumPy
- scikit-learn
- pandas
- matplotlib


