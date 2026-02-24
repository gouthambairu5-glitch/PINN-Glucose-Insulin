# System Architecture

## Model
Fully connected neural network:
Input: time (t)  
Output: [Glucose, Insulin action]

## Loss Function
Total Loss = Data Loss + Physics Loss

### Data Loss
MSE between predicted glucose and observed glucose values.

### Physics Loss
Residuals of the Bergman minimal model:
- dG/dt + (X + p1)G
- dX/dt + p2X − p3(G − Gb)

Computed using automatic differentiation.

## Training Pipeline
1. Sample collocation points in time domain
2. Forward pass through PINN
3. Compute ODE residuals
4. Backpropagate combined loss
5. Optimize with Adam

## Output
Predicted glucose dynamics curve