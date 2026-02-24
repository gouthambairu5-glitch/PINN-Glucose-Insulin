import torch

def physics_loss(model, t, p1=0.03, p2=0.02, p3=0.01, Gb=90.0):
    t.requires_grad_(True)

    out = model(t)
    G = out[:, 0:1]
    X = out[:, 1:2]

    dG_dt = torch.autograd.grad(G, t, torch.ones_like(G), create_graph=True)[0]
    dX_dt = torch.autograd.grad(X, t, torch.ones_like(X), create_graph=True)[0]

    eq1 = dG_dt + (X + p1) * G
    eq2 = dX_dt + p2 * X - p3 * (G - Gb)

    return torch.mean(eq1**2) + torch.mean(eq2**2)
