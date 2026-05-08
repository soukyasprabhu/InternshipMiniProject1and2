import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'CustomerID'      : range(1, n + 1),
    'Gender'          : np.random.choice(['Male', 'Female'], n),
    'Age'             : np.random.randint(18, 70, n),
    'Annual_Income'   : np.random.randint(15, 140, n),   # in $k
    'Spending_Score'  : np.random.randint(1, 100, n),
    'Purchase_Freq'   : np.random.randint(1, 50, n),
})

print('Dataset loaded successfully!')
print(f'Shape: {df.shape}')
print('\nFirst 5 rows:')
print(df.head())
print('\n--- Dataset Info ---')
print(df.info())

print('\n--- Statistical Summary ---')
print(df.describe())

print('\n--- Missing Values ---')
print(df.isnull().sum())

print('\n--- Gender Distribution ---')
print(df['Gender'].value_counts())

# Plot distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

axes[0,0].hist(df['Age'], bins=15, color='steelblue', edgecolor='white')
axes[0,0].set_title('Age Distribution')
axes[0,0].set_xlabel('Age')

axes[0,1].hist(df['Annual_Income'], bins=15, color='coral', edgecolor='white')
axes[0,1].set_title('Annual Income Distribution ($k)')
axes[0,1].set_xlabel('Income ($k)')

axes[1,0].hist(df['Spending_Score'], bins=15, color='mediumseagreen', edgecolor='white')
axes[1,0].set_title('Spending Score Distribution')
axes[1,0].set_xlabel('Spending Score (1-100)')

axes[1,1].hist(df['Purchase_Freq'], bins=15, color='orchid', edgecolor='white')
axes[1,1].set_title('Purchase Frequency Distribution')
axes[1,1].set_xlabel('No. of Purchases')

plt.tight_layout()
plt.savefig('01_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print('Chart saved: 01_distributions.png')
# 1. Handle missing values (drop rows with any NaN)
df.dropna(inplace=True)
print(f'Rows after cleaning: {len(df)}')

# 2. Encode categorical column: Gender -> 0 (Female) / 1 (Male)
df['Gender_Encoded'] = df['Gender'].map({'Female': 0, 'Male': 1})
print('Gender encoded: Female=0, Male=1')

# 3. Select features for clustering
#    We use Annual_Income and Spending_Score (most impactful)
features = ['Annual_Income', 'Spending_Score']
X = df[features].copy()

# 4. Scale the features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('\nFeatures scaled using StandardScaler')
print(f'Feature matrix shape: {X_scaled.shape}')
wcss = []          # Within-Cluster Sum of Squares
sil_scores = []    # Silhouette scores for validation
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# Plot Elbow Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Choosing the Optimal Number of Clusters', fontsize=14, fontweight='bold')

axes[0].plot(list(k_range), wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method (WCSS)')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('WCSS')
axes[0].axvline(x=5, color='red', linestyle='--', label='Elbow at K=5')
axes[0].legend()

axes[1].plot(list(k_range), sil_scores, 'gs-', linewidth=2, markersize=8)
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Score')

plt.tight_layout()
plt.savefig('02_elbow_method.png', dpi=150, bbox_inches='tight')
plt.show()
print('Chart saved: 02_elbow_method.png')

print('\nWCSS values per K:')
for k, w, s in zip(k_range, wcss, sil_scores):
    print(f'  K={k}: WCSS={w:.2f}, Silhouette={s:.4f}')
    OPTIMAL_K = 5  # <-- Change this based on your Elbow plot

    kmeans = KMeans(
        n_clusters=OPTIMAL_K,
        init='k-means++',  # Smart centroid initialization
        n_init=10,  # Run 10 times, pick best result
        max_iter=300,  # Max iterations per run
        random_state=42  # For reproducibility
    )

    kmeans.fit(X_scaled)

    # Assign cluster labels back to original dataframe
    df['Cluster'] = kmeans.labels_

    print(f'Model trained with K={OPTIMAL_K} clusters')
    print(f'\nCluster Distribution:')
    print(df['Cluster'].value_counts().sort_index())

    # Show cluster centroids (in original scale)
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids_original, columns=features)
    centroid_df.index.name = 'Cluster'
    print('\nCluster Centroids (original scale):')
    print(centroid_df.round(2))
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    labels = [f'Cluster {i}' for i in range(OPTIMAL_K)]

    # --- Plot 1: Annual Income vs Spending Score ---
    fig, ax = plt.subplots(figsize=(11, 7))

    for i in range(OPTIMAL_K):
        subset = df[df['Cluster'] == i]
        ax.scatter(
            subset['Annual_Income'],
            subset['Spending_Score'],
            c=colors[i], label=labels[i],
            s=80, alpha=0.75, edgecolors='white', linewidths=0.5
        )

    # Plot centroids
    ax.scatter(
        centroid_df['Annual_Income'],
        centroid_df['Spending_Score'],
        c='black', marker='X', s=250, zorder=5, label='Centroids'
    )

    ax.set_title('Customer Segments: Annual Income vs Spending Score',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Annual Income ($k)', fontsize=12)
    ax.set_ylabel('Spending Score (1-100)', fontsize=12)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('03_clusters_income_spending.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Chart saved: 03_clusters_income_spending.png')

    # --- Plot 2: Cluster Size Bar Chart ---
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cluster_counts.index, cluster_counts.values,
                  color=colors[:OPTIMAL_K], edgecolor='white', width=0.6)
    for bar, val in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title('Number of Customers per Cluster', fontsize=13, fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Customer Count')
    ax.set_xticks(range(OPTIMAL_K))
    plt.tight_layout()
    plt.savefig('04_cluster_sizes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Chart saved: 04_cluster_sizes.png')
    analysis_cols = ['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Freq']
    summary = df.groupby('Cluster')[analysis_cols].mean().round(2)
    summary['Customer_Count'] = df.groupby('Cluster').size()

    print('\n' + '=' * 60)
    print('         CLUSTER ANALYSIS SUMMARY')
    print('=' * 60)
    print(summary.to_string())
    print('=' * 60)

    # --- Interpret each cluster automatically ---
    print('\n--- Business Interpretation ---')
    for i in range(OPTIMAL_K):
        row = summary.loc[i]
        inc = row['Annual_Income']
        spd = row['Spending_Score']
        age = row['Age']
        cnt = int(row['Customer_Count'])

        if inc > 80 and spd > 60:
            segment = 'HIGH VALUE  — High Income, High Spender (Premium)'
        elif inc < 40 and spd < 40:
            segment = 'BUDGET      — Low Income, Low Spender'
        elif inc > 80 and spd < 40:
            segment = 'SAVER       — High Income, Low Spender (Conservative)'
        elif inc < 40 and spd > 60:
            segment = 'IMPULSE     — Low Income, High Spender (Risk)'
        else:
            segment = 'AVERAGE     — Moderate Income & Spending'

        print(f'  Cluster {i}: {segment}')
        print(f'            Avg Income=${inc}k  |  Score={spd}  |  Age={age:.0f}  |  Count={cnt}')
        print()

    # --- Heatmap of cluster profiles ---
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(summary[analysis_cols].T, annot=True, fmt='.1f',
                cmap='YlOrRd', linewidths=0.5, ax=ax)
    ax.set_title('Cluster Profile Heatmap', fontsize=13, fontweight='bold')
    ax.set_xlabel('Cluster')
    plt.tight_layout()
    plt.savefig('05_cluster_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Chart saved: 05_cluster_heatmap.png')
    output_file = 'customer_segments_output.csv'
    df.to_csv(output_file, index=False)
    print(f'Results saved to: {output_file}')

    # Preview output
    print('\nSample output:')
    print(df[['CustomerID', 'Annual_Income', 'Spending_Score', 'Cluster']].head(10))
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    sil = silhouette_score(X_scaled, df['Cluster'])
    db = davies_bouldin_score(X_scaled, df['Cluster'])
    ch = calinski_harabasz_score(X_scaled, df['Cluster'])

    print("=== Clustering Validation Metrics ===")
    print(f"Silhouette Score      : {sil:.4f}  (higher is better, best=1.0)")
    print(f"Davies-Bouldin Index  : {db:.4f}  (lower is better, best=0.0)")
    print(f"Calinski-Harabasz     : {ch:.2f}  (higher is better)")
    