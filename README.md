# Housing Price Prediction

A machine learning project that predicts California housing prices
using Linear Regression.

## Dataset
California Housing dataset (built into scikit-learn)
- 20,640 samples
- Features: median income, house age, average rooms, location, etc.
- Target: median house value

## Model
Linear Regression — chosen as a starting point for understanding
the relationship between features and price.

## Results
- R² Score: ~0.60
- Meaning: the model explains about 60% of price variance

## Tech Stack
- Python 3.x
- scikit-learn
- numpy

## How to Run
pip install -r requirements.txt
python housing_predict.py

## What I learned
- How to load and split a real dataset
- How to train and evaluate a regression model
- What R² score means in practice
