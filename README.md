# PINN for Glucose–Insulin Dynamics (Diabetes Modeling)

## Overview
This project implements a Physics-Informed Neural Network (PINN) to model glucose–insulin dynamics using the Bergman minimal model.

## Medical Relevance
Used in diabetes research to analyze glucose regulation and insulin sensitivity.

## Governing Equations
dG/dt = −(X + p1)G  
dX/dt = −p2X + p3(G − Gb)

Where:
- G → Glucose concentration
- X → Insulin action
- Gb → Basal glucose level

## Method
The network is trained using:
- Data loss from observed glucose values
- Physics loss from ODE residuals using automatic differentiation

## Features
- Physics-constrained learning
- Data-efficient biomedical modeling
- Glucose curve prediction and visualization

## Run
python train.py