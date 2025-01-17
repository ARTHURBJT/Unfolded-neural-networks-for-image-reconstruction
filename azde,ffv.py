import torch
import torch.nn as nn
import torch.nn.functional as F

# Définir une couche de convolution
conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False)




# Taille de l'image d'entrée et du noyau
C, H, W = 1, 5, 5
K = 3
P = 1

# Générer une image d'entrée de taille (1, 1, H, W)
input_image = torch.randn(1, C, H, W)

output = conv(input_image)


# Appliquer la transformation unfold
input_unfold = F.unfold(input_image, kernel_size=K, padding=P)

# Dimensions de la matrice unfold
out_H = (H + 2 * P - K) + 1
out_W = (W + 2 * P - K) + 1
unfold_dim = C * K * K
output_size = out_H * out_W

# Initialiser la matrice unfold M
M = torch.zeros((output_size * unfold_dim, C * H * W))



# Remplir la matrice M
patch_idx = 0
for i in range(out_H):
    for j in range(out_W):
        for ki in range(K):
            for kj in range(K):
                input_i = i - ki - P
                input_j = j - kj - P
                row = patch_idx * unfold_dim + ki * K + kj
                if 0 <= input_i < H and 0 <= input_j < W:
                    col = input_i * W + input_j
                    M[row, col] = 1
        patch_idx += 1

# Aplatir l'image d'entrée
input_flat = input_image.view(-1)

# Effectuer la multiplication matricielle
output_unfold_flat = M @ input_flat

# Reshaper la sortie pour correspondre à celle de F.unfold
output_unfold = output_unfold_flat.view(1, unfold_dim, output_size)

# Comparer les deux sorties
print(torch.allclose(input_unfold, output_unfold, atol=1e-6))  # Devrait être True

# Afficher les matrices pour vérifier visuellement
print("Matrice générée par unfold:\n", input_unfold)
print("Matrice générée par multiplication:\n", output_unfold)










# Reshaper les poids de la couche de convolution
conv_weight = conv.weight.view(64, -1)
print(input_unfold.shape, conv_weight.shape)
# Effectuer la multiplication matricielle équivalente à la convolution
output_unfold = conv_weight @ input_unfold

print(output_unfold.shape)


# Reshaper la sortie pour correspondre à la sortie de la convolution classique
output_unfold = output_unfold.view(1, 64, H, W)

# Comparer les deux sorties
print(torch.allclose(output, output_unfold, atol=1e-6))  # Devrait être True
