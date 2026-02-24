import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import PINN
from physics import physics_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()

t_data = torch.linspace(0, 10, 50).view(-1, 1).to(device)
G_true = 90 + 40 * torch.exp(-0.5 * t_data)

for epoch in range(3000):
    t_collocation = torch.rand(100, 1).to(device) * 10

    pred = model(t_data)
    G_pred = pred[:, 0:1]

    data_loss = mse(G_pred, G_true)
    phys_loss = physics_loss(model, t_collocation)

    loss = data_loss + phys_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(epoch, loss.item())

t_test = torch.linspace(0, 10, 100).view(-1, 1).to(device)
pred = model(t_test).detach().cpu().numpy()

plt.plot(t_test.cpu(), pred[:, 0], label="Predicted")
plt.plot(t_data.cpu(), G_true.cpu(), "o", label="Data")
plt.legend()
plt.title("Glucose PINN")
plt.show()
