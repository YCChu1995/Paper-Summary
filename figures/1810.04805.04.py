import matplotlib.pyplot as plt
from transformers import BertModel
import numpy as np

model = BertModel.from_pretrained('bert-base-uncased')
pos_emb = model.embeddings.position_embeddings.weight.detach().cpu().numpy()
print(np.max(pos_emb), np.min(pos_emb))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Position Embeddings Heatmap
im = axes[0].imshow(pos_emb.T, aspect='auto', cmap='coolwarm', vmin=-0.1, vmax=0.1)
axes[0].set_ylabel('Embedding dimension')
axes[0].set_xlabel('Position')
axes[0].set_title('Position Embeddings', fontsize=14)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

# Histogram
axes[1].hist(pos_emb.flatten(), bins=100, color='skyblue', edgecolor='black')
axes[1].set_xlabel('Embedding Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Position Embedding Values\n(max ~ 0.727, min ~ -0.949)')

# Center x-axis at zero
axes[1].set_xlim(-0.3, 0.3)
axes[1].axvline(0, color='red', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()
