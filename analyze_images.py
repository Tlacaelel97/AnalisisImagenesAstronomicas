import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.measure import shannon_entropy
from scipy.ndimage import laplace
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def analyze_dataset(data_path='data', sample_size=1000):
    data_dir = Path(data_path)
    image_files = list(data_dir.glob('*.jpg'))
    
    stats = []
    corrupted = []
    
    print(f"Analyzing {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        try:
            with Image.open(img_path) as img:
                # Basic metadata
                width, height = img.size
                mode = img.mode
                
                # Image arrays
                img_array = np.array(img)
                img_gray = img.convert('L')
                img_array_gray = np.array(img_gray)
                
                # Per-channel stats
                if img_array.ndim == 3:
                    r_mean, g_mean, b_mean = np.mean(img_array, axis=(0, 1))
                    r_std, g_std, b_std = np.std(img_array, axis=(0, 1))
                else:
                    r_mean = g_mean = b_mean = np.mean(img_array)
                    r_std = g_std = b_std = np.std(img_array)

                # Global stats
                mean_val = np.mean(img_array)
                std_val = np.std(img_array)
                min_val = np.min(img_array)
                max_val = np.max(img_array)
                
                # Advanced metrics
                ent = shannon_entropy(img_array)
                sharpness = np.var(laplace(img_array_gray))
                
                stats.append({
                    'filename': img_path.name,
                    'width': width,
                    'height': height,
                    'mean_intensity': mean_val,
                    'std_intensity': std_val,
                    'min_intensity': min_val,
                    'max_intensity': max_val,
                    'r_mean': r_mean, 'g_mean': g_mean, 'b_mean': b_mean,
                    'entropy': ent,
                    'sharpness': sharpness
                })
        except Exception as e:
            corrupted.append({'filename': img_path.name, 'error': str(e)})

    df = pd.DataFrame(stats)
    
    # --- Clustering for Representative Sampling ---
    print("\nPerforming clustering to identify representative sample...")
    features = ['mean_intensity', 'std_intensity', 'entropy', 'sharpness', 'r_mean', 'g_mean', 'b_mean']
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster into 5 types of images
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Select representative sample (proportional to cluster size)
    representative_sample = df.groupby('cluster', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // n_clusters), random_state=42)
    )
    
    print(f"Representative sample of {len(representative_sample)} images selected.")
    representative_sample.to_csv('representative_sample.csv', index=False)

    # --- Visualizations ---
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Distribution of Intensity
    plt.subplot(2, 2, 1)
    plt.hist(df['mean_intensity'], bins=50, color='skyblue', edgecolor='black')
    plt.axvline(df['mean_intensity'].mean(), color='red', linestyle='dashed', linewidth=1, label='Media')
    plt.title('Distribución de Intensidad Media')
    plt.legend()
    
    # 2. Clusters Visualization (Mean vs Entropy)
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(df['mean_intensity'], df['entropy'], c=df['cluster'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('Clusters de Imágenes (Media vs Entropía)')
    plt.xlabel('Intensidad Media')
    plt.ylabel('Entropía (Complejidad)')
    
    # 3. Bar chart of global Min, Mean, Max
    plt.subplot(2, 2, 3)
    labels = ['Mínimo', 'Promedio', 'Máximo']
    values = [df['min_intensity'].min(), df['mean_intensity'].mean(), df['max_intensity'].max()]
    plt.bar(labels, values, color=['red', 'blue', 'green'], alpha=0.7)
    plt.title('Resumen General de Intensidades')
    for i, v in enumerate(values):
        plt.text(i, v + 2, f'{v:.2f}', ha='center')

    # 4. RGB Channel Means
    plt.subplot(2, 2, 4)
    channel_means = [df['r_mean'].mean(), df['g_mean'].mean(), df['b_mean'].mean()]
    plt.bar(['Rojo', 'Verde', 'Azul'], channel_means, color=['red', 'green', 'blue'], alpha=0.6)
    plt.title('Intensidad Promedio por Canal RGB')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis_final.png')
    
    print("\n--- Estadísticas de los Clusters ---")
    print(df.groupby('cluster')[features].mean())
    
    print(f"\nProcesadas: {len(df)} | Corruptas: {len(corrupted)}")
    print("Muestra representativa guardada en 'representative_sample.csv'")
    print("Gráficas guardadas en 'dataset_analysis_final.png'")

if __name__ == "__main__":
    analyze_dataset()
