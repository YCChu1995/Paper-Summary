from sentence_transformers import SentenceTransformer

# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embed_model.save_pretrained("./models/MiniLM") # Save the model to a local directory

embed_model = SentenceTransformer("./models/MiniLM") # Load the model from the local directory
def embed_texts(texts):
    return embed_model.encode(texts, normalize_embeddings=True)

texts = ["I love machine learning.", "The cat is on the table.", "Dods are under the bed.", "I hate animals.", "I am mayjoring in computer science."]
embeddings = embed_texts(texts)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce embeddings to 3D for visualization

pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# Plot the embeddings in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, text in enumerate(texts):
    ax.scatter(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], s=100)
    ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], text, fontsize=9)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Text Embeddings Visualization (3D)')
plt.tight_layout()
plt.show()