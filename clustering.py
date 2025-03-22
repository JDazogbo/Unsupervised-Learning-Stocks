import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import TruncatedSVD

class StockClusterer:
    def __init__(self, data_path, features_to_use=None, max_k=10, n_components=None):
        """
        Initialize the clusterer with data path and features to use
        
        Args:
            data_path (str): Path to the CSV data file
            features_to_use (list): List of column names to use for clustering (2-4 features)
            max_k (int): Maximum number of clusters to try in elbow method
            n_components (int): Number of components to keep after SVD reduction
        """
        self.data_path = data_path
        if features_to_use is not None and not (2 <= len(features_to_use) <= 4):
            raise ValueError("Number of features must be between 2 and 4")
        self.features_to_use = features_to_use
        self.max_k = max_k
        self.n_components = n_components
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
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.features_to_use = numeric_columns[:min(4, len(numeric_columns))]
            
        # Normalize the features
        self.scaled_features = self.scaler.fit_transform(self.data[self.features_to_use])
        self.scaled_data = pd.DataFrame(self.scaled_features, columns=self.features_to_use)
        
        # Perform SVD dimensionality reduction
        if self.n_components is None:
            self.n_components = min(len(self.features_to_use), 4)  # Default to number of features or 4
        
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.reduced_data = self.svd_model.fit_transform(self.scaled_data)
        
        # Calculate explained variance ratio
        explained_variance_ratio = self.svd_model.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('SVD: Cumulative Explained Variance')
        plt.grid(True)
        plt.show()
        
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
        Plot clusters using available features. Supports 2D, 3D (with color), and 4D (3D with color) visualizations
        """
        n_features = self.n_components
        
        if n_features == 2:
            self._plot_2d()
        elif n_features == 3:
            self._plot_3d_as_2d()
        elif n_features == 4:
            self._plot_4d_as_3d()
        else:
            raise ValueError("Number of components must be between 2 and 4")
    
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
    
    def _plot_3d_as_2d(self):
        """Plot 3D data on 2D plane with color encoding third dimension"""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.reduced_data[:, 0], 
            self.reduced_data[:, 1], 
            c=self.data['cluster'],
            s=100,
            cmap='viridis',
            edgecolor='k'
        )
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('3D Data Visualization (SVD Components)')
        
        # Add cluster information
        for cluster in range(self.kmeans_model.n_clusters):
            cluster_points = self.reduced_data[self.data['cluster'] == cluster]
            plt.scatter(
                cluster_points[:, 0].mean(),
                cluster_points[:, 1].mean(),
                marker='*',
                s=200,
                c='red',
                label=f'Cluster {cluster} Center'
            )
        
        plt.legend()
        plt.grid(True)
        self._add_hover_annotations(scatter)
        plt.show()
    
    def _plot_4d_as_3d(self):
        """Plot 4D data using 3D visualization with color"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.reduced_data[:, 0],
            self.reduced_data[:, 1],
            self.reduced_data[:, 2],
            c=self.data['cluster'],
            cmap='viridis',
            s=100
        )
        
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        plt.title('4D Data Visualization (SVD Components)')
        
        # Add cluster centers
        for cluster in range(self.kmeans_model.n_clusters):
            cluster_points = self.reduced_data[self.data['cluster'] == cluster]
            ax.scatter(
                cluster_points[:, 0].mean(),
                cluster_points[:, 1].mean(),
                cluster_points[:, 2].mean(),
                marker='*',
                s=200,
                c='red',
                label=f'Cluster {cluster} Center'
            )
        
        ax.legend()
        plt.grid(True)
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
    # Example features to use (modify as needed)
    FEATURES_TO_USE = ['2024-Net Interest Income', '2024-Operating Expense', '2024-Normalized EBITDA']  # Replace with your actual feature names
    DATA_PATH = "data.csv"  # Replace with your data path
    
    try:
        # Initialize and run clustering
        clusterer = StockClusterer(DATA_PATH, features_to_use=FEATURES_TO_USE, max_k=10, n_components=2)
        
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
        
        # Plot clusters
        clusterer.plot_clusters()
        
        # Analyze clusters
        cluster_analysis = clusterer.analyze_clusters()
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file at {DATA_PATH}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
