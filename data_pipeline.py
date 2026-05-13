import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from skimage.measure import shannon_entropy
from scipy.ndimage import laplace
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def extract_features(data_path='data'):
    """
    Extracts visual features from all images in the specified directory.
    Based on the exploration done in src.ipynb.
    """
    data_dir = Path(data_path)
    image_files = list(data_dir.glob('*.jpg'))
    
    features = []
    corrupted = []
    
    print(f"--- Step 1: Feature Extraction (Processing {len(image_files)} images) ---")
    
    for img_path in tqdm(image_files, desc="Extracting features"):
        try:
            with Image.open(img_path) as img:
                # Basic metadata
                width, height = img.size
                
                # Image arrays
                img_array = np.array(img)
                img_gray = img.convert('L')
                img_array_gray = np.array(img_gray)
                
                # Global stats
                mean_val = np.mean(img_array)
                std_val = np.std(img_array)
                
                # Advanced metrics
                ent = shannon_entropy(img_array)
                # Sharpness: Variance of the Laplacian
                sharpness = np.var(laplace(img_array_gray))
                
                features.append({
                    'filename': img_path.name,
                    'width': width,
                    'height': height,
                    'mean_intensity': mean_val,
                    'std_intensity': std_val,
                    'entropy': ent,
                    'sharpness': sharpness
                })
        except Exception as e:
            corrupted.append({'filename': img_path.name, 'error': str(e)})

    df = pd.DataFrame(features)
    print(f"Success: {len(df)} images | Failed/Corrupted: {len(corrupted)}")
    
    if corrupted:
        pd.DataFrame(corrupted).to_csv('corrupted_files.csv', index=False)
        print("Corrupted files list saved to 'corrupted_files.csv'")
        
    df.to_csv('full_dataset_stats.csv', index=False)
    return df

def calculate_sample_size(N, confidence_level=0.95, margin_error=0.03):
    """
    Calculates a statistically significant sample size using Cochran's Formula
    with finite population correction.
    """
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    Z = z_map.get(confidence_level, 1.96)
    p = 0.5 
    n0 = (Z**2 * p * (1 - p)) / (margin_error**2)
    n = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))

def perform_sampling(df, target_n):
    """
    Performs Stratified Proportional Sampling using KMeans.
    """
    print(f"\n--- Step 2: Statistical Sampling (Target n={target_n}) ---")
    sampling_features = ['mean_intensity', 'std_intensity', 'entropy', 'sharpness']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[sampling_features])
    
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['strata'] = kmeans.fit_predict(X_scaled)
    
    sample_df = df.groupby('strata', group_keys=False).apply(
        lambda x: x.sample(n=int(np.ceil(target_n * len(x) / len(df))), random_state=42)
    ).head(target_n)
    
    print(f"Sample selected with {len(sample_df)} images.")
    sample_df.to_csv('significant_sample.csv', index=False)
    return sample_df

def validate_sample(population_df, sample_df):
    """
    Validates the sample using Kolmogorov-Smirnov test.
    """
    print("\n--- Step 3: Statistical Validation ---")
    metrics = ['mean_intensity', 'entropy', 'sharpness']
    results = []
    for metric in metrics:
        ks_stat, p_value = stats.ks_2samp(population_df[metric], sample_df[metric])
        results.append({
            'Metric': metric,
            'KS Statistic': ks_stat,
            'P-Value': p_value,
            'Significant Difference?': 'Yes' if p_value < 0.05 else 'No'
        })
    print(pd.DataFrame(results))
    
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        plt.hist(population_df[metric], bins=30, alpha=0.5, label='Población', density=True)
        plt.hist(sample_df[metric], bins=30, alpha=0.5, label='Muestra', density=True)
        plt.title(f'Distribución de {metric}')
        plt.legend()
    plt.tight_layout()
    plt.savefig('sampling_validation.png')
    print("\nValidation plots saved to 'sampling_validation.png'")

def run_workflow():
    if os.path.exists('full_dataset_stats.csv'):
        print("Loading existing features from 'full_dataset_stats.csv'...")
        df = pd.read_csv('full_dataset_stats.csv')
    else:
        df = extract_features()
    
    N = len(df)
    n_target = calculate_sample_size(N, confidence_level=0.95, margin_error=0.03)
    print(f"For a population of {N}, a significant sample size is {n_target} (95% CI, 3% MoE)")
    
    sample_df = perform_sampling(df, n_target)
    validate_sample(df, sample_df)
    print("\nWorkflow completed successfully.")

if __name__ == "__main__":
    run_workflow()
