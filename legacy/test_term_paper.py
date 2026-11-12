import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing wave-optics engine on: {device}")

# -------------------------------------------------------------
# 1. Physical Parameters (Compact Cascade)
# -------------------------------------------------------------
wavelength = 0.75e-3          # 0.4 THz (0.75 mm)
k0 = 2 * np.pi / wavelength
dx = wavelength / 2.0         # Spatial Nyquist sampling (0.375 mm)
grid_n = 128                  # 128x128 computational window
d = 9 * wavelength            # 6.75 mm drift per layer (reduces beam expansion)

fx = torch.fft.fftfreq(grid_n, d=dx, device=device)
fy = torch.fft.fftfreq(grid_n, d=dx, device=device)
FY, FX = torch.meshgrid(fy, fx, indexing='ij')
kx = 2 * np.pi * FX
ky = 2 * np.pi * FY
kz_sq = k0**2 - kx**2 - ky**2

prop_mask = kz_sq > 0
kz = torch.zeros_like(kz_sq)
kz[prop_mask] = torch.sqrt(kz_sq[prop_mask])
H_kernel = torch.where(prop_mask, torch.exp(1j * kz * d), torch.zeros_like(kz_sq, dtype=torch.complex64))

def asm_propagate(field):
    return torch.fft.ifft2(torch.fft.fft2(field) * H_kernel)

# -------------------------------------------------------------
# 2. FOV & Radial Aperture Confinement Mask
# -------------------------------------------------------------
fov_size = 8
fov_start = (grid_n - fov_size) // 2
fov_end = fov_start + fov_size

# Radial spatial mask to prevent light scattering into outer edges
r_grid = torch.sqrt(FX**2 + FY**2)
r_norm = r_grid / r_grid.max()
spatial_mask = torch.exp(-(r_norm / 0.45)**8).to(device)

def embed_fov(small_patch):
    B = small_patch.shape[0]
    full_field = torch.zeros((B, grid_n, grid_n), dtype=torch.complex64, device=device)
    full_field[:, fov_start:fov_end, fov_start:fov_end] = small_patch
    return full_field

def extract_fov(full_field):
    return full_field[:, fov_start:fov_end, fov_start:fov_end]

# -------------------------------------------------------------
# 3. Model: 8-Layer Phase-Only D^2NN
# -------------------------------------------------------------
class OpticalD2NN(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.num_layers = num_layers
        self.phases = nn.ParameterList([
            nn.Parameter((torch.rand(grid_n, grid_n, device=device) - 0.5) * np.pi)
            for _ in range(num_layers)
        ])

    def forward(self, u_in):
        u = u_in
        for phi in self.phases:
            u = asm_propagate(u)
            u = u * torch.exp(1j * phi)
            u = u * spatial_mask      # Enforce spatial confinement
        u = asm_propagate(u)
        return u

# -------------------------------------------------------------
# 4. Optimization Loop (Kulce et al. Exponential Efficiency Loss)
# -------------------------------------------------------------
model = OpticalD2NN(num_layers=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.015)
epochs = 3000
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)

batch_size = 64
c_M = 1.0                     # Weight for fidelity loss
c_D = 0.15                    # Weight for efficiency loss [Eq. 24]
alpha = 25.0                  # Exponential rate parameter [Eq. 25]

print(f"Starting optimized training ({epochs} epochs, compact d=9λ, exponential loss)...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # Unit input power normalization
    rand_real = torch.randn(batch_size, fov_size, fov_size, device=device)
    rand_imag = torch.randn(batch_size, fov_size, fov_size, device=device)
    raw_inputs = torch.complex(rand_real, rand_imag)
    norm_const = torch.sqrt(torch.sum(torch.abs(raw_inputs)**2, dim=(-2, -1), keepdim=True))
    inputs_8x8 = raw_inputs / norm_const

    targets_8x8 = torch.fft.fftshift(torch.fft.fft2(inputs_8x8), dim=(-2, -1)) / float(fov_size)
    
    u_out = model(embed_fov(inputs_8x8))
    pred_8x8 = extract_fov(u_out)
    
    in_power = torch.sum(torch.abs(inputs_8x8)**2, dim=(-2, -1))
    pred_power = torch.sum(torch.abs(pred_8x8)**2, dim=(-2, -1))
    eff = torch.mean(pred_power / (in_power + 1e-8))
    
    overlap = torch.sum(pred_8x8 * torch.conj(targets_8x8), dim=(-2, -1))
    norm_pred = torch.sqrt(pred_power + 1e-8)
    norm_targ = torch.sqrt(torch.sum(torch.abs(targets_8x8)**2, dim=(-2, -1)) + 1e-8)
    
    fidelity = torch.abs(overlap) / (norm_pred * norm_targ)
    loss_fidelity = 1.0 - torch.mean(fidelity)
    
    # Kulce et al. formulation: L_D = exp(-alpha * eff)
    loss_eff = torch.exp(-alpha * eff)
    
    total_loss = c_M * loss_fidelity + c_D * loss_eff
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 300 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{epochs} | Fidelity: {torch.mean(fidelity).item()*100:6.2f}% | Efficiency: {eff.item()*100:5.2f}% | Total Loss: {total_loss.item():.4f}")

# -------------------------------------------------------------
# 5. Evaluate Linear Operator Transfer Matrix
# -------------------------------------------------------------
print("Evaluating linear operator transfer matrix...")
A_pred = torch.zeros((64, 64), dtype=torch.complex64, device=device)
A_true = torch.zeros((64, 64), dtype=torch.complex64, device=device)

with torch.no_grad():
    for idx in range(64):
        delta_in = torch.zeros((1, fov_size, fov_size), dtype=torch.complex64, device=device)
        delta_in[0, idx // fov_size, idx % fov_size] = 1.0 + 0j
        
        dft_out = torch.fft.fftshift(torch.fft.fft2(delta_in), dim=(-2, -1)) / float(fov_size)
        A_true[:, idx] = dft_out.reshape(-1)
        
        pred_out = extract_fov(model(embed_fov(delta_in)))
        A_pred[:, idx] = pred_out.reshape(-1)

scale_c = torch.sum(A_true * torch.conj(A_pred)) / (torch.sum(torch.abs(A_pred)**2) + 1e-8)
A_pred_aligned = scale_c * A_pred

phase_true = torch.angle(A_true).cpu().numpy()
phase_pred = torch.angle(A_pred_aligned).cpu().numpy()
phase_diff = np.abs(np.angle(np.exp(1j * (phase_pred - phase_true))))

# -------------------------------------------------------------
# 6. Export Figure
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(phase_true, cmap='twilight', vmin=-np.pi, vmax=np.pi)
axes[0].set_title(r"Ground-Truth Target Phase $\angle A$", fontsize=11)
axes[0].set_xlabel("Input Vector Index")
axes[0].set_ylabel("Output Vector Index")
cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.set_ticks([-np.pi, 0, np.pi])
cbar0.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

im1 = axes[1].imshow(phase_pred, cmap='twilight', vmin=-np.pi, vmax=np.pi)
axes[1].set_title(r"Synthesized Optical Phase $\angle A'$", fontsize=11)
axes[1].set_xlabel("Input Vector Index")
cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_ticks([-np.pi, 0, np.pi])
cbar1.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

im2 = axes[2].imshow(phase_diff, cmap='magma', vmin=0, vmax=np.pi)
axes[2].set_title(r"Absolute Phase Residual $|\Delta \phi|$", fontsize=11)
axes[2].set_xlabel("Input Vector Index")
cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_ticks([0, np.pi/2, np.pi])
cbar2.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

plt.tight_layout()
plt.savefig("optical_dft_kulce_optimized.png", dpi=300)
print("Saved figure as 'optical_dft_kulce_optimized.png'.")