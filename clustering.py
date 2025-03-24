import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

class StockClusterer:
    def __init__(self, data_path, features_to_use=None, max_k=10):
        """
        Initialize the clusterer with data path and features to use
        
        Args:
            data_path (str): Path to the CSV data file
            features_to_use (list): List of column names to use for clustering (can be any number of features)
            max_k (int): Maximum number of clusters to try in elbow method
        """
        self.data_path = data_path
        self.features_to_use = features_to_use
        self.max_k = max_k
        self.n_components = 2
        self.data = None
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.svd_model = None
        self.reduced_data = None
        
    def load_and_prepare_data(self):
        """Load and prepare the data for clustering"""
        # Load the data
        self.data = pd.read_csv(self.data_path)
        
        # If no features specified, use all numeric columns
        if self.features_to_use is None:
            self.features_to_use = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
        # Normalize the features
        self.scaled_features = self.scaler.fit_transform(self.data[self.features_to_use])
        self.scaled_data = pd.DataFrame(self.scaled_features, columns=self.features_to_use)
        
        # Perform SVD dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.reduced_data = self.svd_model.fit_transform(self.scaled_data)
        
        # Calculate explained variance ratio
        explained_variance_ratio = self.svd_model.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        
        print("\nExplained variance ratio for each component:")
        for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio), 1):
            print(f"Component {i}: {var:.4f} (Cumulative: {cum_var:.4f})")
        
        return self.reduced_data
    
    def find_optimal_k(self):
        """Use elbow method to find optimal number of clusters"""
        distortions = []
        K_range = range(1, self.max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.reduced_data)
            distortions.append(kmeans.inertia_)
            
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, distortions, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion (Inertia)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.show()
        
        return distortions
    
    def perform_clustering(self, n_clusters):
        """Perform KMeans clustering with specified number of clusters"""
        self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        self.data['cluster'] = self.kmeans_model.fit_predict(self.reduced_data)
        
        return self.data['cluster']
    
    def plot_clusters(self):
        """
        Plot clusters using available features. Supports 2D visualizations
        """
        n_features = self.n_components
        
        if n_features == 2:
            self._plot_2d()
        else:
            raise ValueError("Number of components must be 2")
    
    def _plot_2d(self):
        """Plot 2D visualization"""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.reduced_data[:, 0], 
            self.reduced_data[:, 1], 
            c=self.data['cluster'],
            cmap='viridis',
            edgecolor='k',
            s=100
        )
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('2D Cluster Visualization (SVD Components)')
        plt.grid(True)
        self._add_hover_annotations(scatter)
        plt.show()
    
    def _add_hover_annotations(self, scatter):
        """Add hover annotations to the plot"""
        annot = plt.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="->")
        )
        annot.set_visible(False)

        def hover(event):
            if event.inaxes is None:
                return
            cont, ind = scatter.contains(event)
            if cont:
                pos = scatter.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                row = self.data.iloc[ind['ind'][0]]
                text = f"Ticker: {row.get('Ticker', 'N/A')}\n"
                text += f"Industry: {row.get('Industry', 'N/A')}\n"
                
                # Handle share price formatting
                share_price = row.get('Current Share Price', 'N/A')
                if share_price != 'N/A':
                    try:
                        share_price = float(share_price)
                        text += f"Share Price: ${share_price:.2f}"
                    except (ValueError, TypeError):
                        text += f"Share Price: {share_price}"
                else:
                    text += "Share Price: N/A"
                
                annot.set_text(text)
                annot.set_visible(True)
                plt.draw()
            else:
                annot.set_visible(False)
                plt.draw()

        plt.gcf().canvas.mpl_connect("motion_notify_event", hover)
    
    def analyze_clusters(self):
        """Analyze the characteristics of each cluster"""
        cluster_stats = []
        
        for cluster in range(self.kmeans_model.n_clusters):
            cluster_data = self.data[self.data['cluster'] == cluster]
            stats = cluster_data[self.features_to_use].describe()
            cluster_stats.append({
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Stats': stats
            })
            
            print(f"\nCluster {cluster} Statistics:")
            print(f"Size: {len(cluster_data)} samples")
            print(stats)
            
        return cluster_stats

# Example usage
if __name__ == "__main__":
    DATA_PATH = "data.csv"  # Replace with your data path
    
    FEATURES_TO_USE = [
        # Current Data, to be compared
        '2024-Operating Expense',
        '2024-Net Interest Income',
        '2024-Diluted EPS',

        # Historical Data, for trends over time
        '2023-Tax Rate For Calcs',
        '2022-Tax Rate For Calcs',
        '2021-Operating Expense',

        # Normalized Data, for comparison
        '2024-Net Income From Continuing Operation Net Minority Interest_pct',
        '2024-Operating Expense_pct',
        '2024-Net Interest Income_pct',
    ]
    
    # Initialize and run clustering with 2 components for visualization
    clusterer = StockClusterer(DATA_PATH, features_to_use=FEATURES_TO_USE, max_k=20)
    
    # Load and prepare data
    clusterer.load_and_prepare_data()
    
    # Find optimal k using elbow method
    clusterer.find_optimal_k()
    
    # Get optimal k from user with error handling
    while True:
        try:
            optimal_k = int(input("\nEnter the optimal number of clusters based on the elbow plot: "))
            if 1 <= optimal_k <= clusterer.max_k:
                break
            else:
                print(f"Please enter a number between 1 and {clusterer.max_k}")
        except ValueError:
            print("Please enter a valid number")
    
    # Perform clustering
    clusters = clusterer.perform_clustering(optimal_k)
    
    # Save the normalized data with clusters
    preprocessed_df = clusterer.scaled_data.copy()
    preprocessed_df['cluster'] = clusters.values
    preprocessed_df.to_csv('preprocessedData.csv', index=False)

    # Plot clusters
    clusterer.plot_clusters()
    
    # Analyze clusters
    cluster_analysis = clusterer.analyze_clusters()