import torch
from sparcity_op import sparcity_op

# Reconstruction operators

# x is a column
def ComputeTVlin(x, nx, ny):
    batch_size = x.size(0)
    X = x.view(batch_size, ny, nx)
    Du = torch.diff(X, dim=2, prepend=torch.zeros(batch_size, ny, 1, device=x.device))
    Dv = torch.diff(X, dim=1, prepend=torch.zeros(batch_size, 1, nx, device=x.device))
    return torch.cat([Du.view(batch_size, -1), Dv.view(batch_size, -1)], dim=1)

def ComputeTVlinAdj(z, nx, ny):
    batch_size = z.size(0)
    Z = z.view(batch_size, 2 * ny, nx)
    U = torch.diff(Z[:, :ny], dim=2, append=torch.zeros(batch_size, ny, 1, device=z.device))
    V = torch.diff(Z[:, ny:], dim=1, append=torch.zeros(batch_size, 1, nx, device=z.device))
    return (U + V).view(batch_size, -1)

def prox_tv(x, lambda_, nx, ny):
    batch_size = x.size(0)
    u = x[:, :nx * ny]
    v = x[:, nx * ny:]
    sqrtuv = torch.sqrt(u**2 + v**2)
    nonzero = sqrtuv > lambda_
    zu = torch.zeros_like(u)
    zv = torch.zeros_like(v)
    factor = (1 - lambda_ / sqrtuv[nonzero])
    zu[nonzero] = factor * u[nonzero]
    zv[nonzero] = factor * v[nonzero]
    return torch.cat([zu, zv], dim=1)

def Wadj(z, nx, ny):
    batch_size = z.size(0)
    Z = z.view(batch_size, 2 * ny, nx)
    U = torch.zeros((batch_size, ny, nx), device=z.device)
    V = torch.zeros((batch_size, ny, nx), device=z.device)

    # Horizontal adjoint operation
    U[:, :, :-1] += Z[:, :ny, :-1] - Z[:, :ny, 1:]
    U[:, :, -1] -= Z[:, :ny, -1]

    # Vertical adjoint operation
    V[:, :-1, :] += Z[:, ny:2 * ny - 1, :] - Z[:, ny + 1:2 * ny, :]
    V[:, -1, :] -= Z[:, 2 * ny - 1, :]

    return (U + V).view(batch_size, -1)

# Algorithm parameters

Nx = 28
Ny = 28

lambda_ = 0.03
lambda2 = 0.001
normH = 1
normW = 2 * torch.sqrt(torch.tensor(2.0))

sigma = 1 / normW**2
tau = 0.99 / (normH**2 / 2 + sigma * normW**2)

W = lambda x: ComputeTVlin(x, Nx, Ny)
Wt = lambda u: Wadj(u, Nx, Ny)


def approx_TV(im_noisy, lambd, H, H_star, NbIt):
    norm_it = 0

    ydata = im_noisy
    x = H_star(im_noisy)

    v = W(x)

    for it in range(NbIt):
        vold = v
        xold = x

        # x = x - tau * H_star(H(x) - ydata) - tau * Wt(v)
        # x = torch.clamp(x, min = 0, max = 1)
        # v = v + sigma * W(2*x-xold)
        # v = torch.clamp(v, min = -sigma*lambd, max = sigma*lambd)
        
        vtmp = v + sigma * W(x)
        #v = vtmp - sigma * prox_tv(vtmp / sigma, lambda_ / sigma, Nx, Ny)
        v = torch.clamp(vtmp, min = -lambd, max = lambd)


        x_ = x - tau * H_star(H(x) - ydata) - tau * Wt(2 * v - vold)
        x = x_ # soft(x_, lambda2)
        x = torch.clamp(x, 0, 1)

        norm_it = torch.norm(x - xold) / torch.norm(x)
        if norm_it < 1e-6:
            break
    return x, it