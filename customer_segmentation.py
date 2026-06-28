import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def customer_segmentation():

    # Read Dataset
    df = pd.read_csv("Mall_Customers.csv")

    print("\nFirst Five Rows")
    print(df.head())

    # Select Numerical Columns
    X = df.select_dtypes(include=['int64', 'float64'])

    print("\nDataset Shape:", X.shape)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------
    # Elbow Method
    # ----------------------------
    inertia = []

    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertia.append(model.inertia_)

    plt.figure(figsize=(7,5))
    plt.plot(range(1,11), inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.show()

    # ----------------------------
    # Silhouette Score
    # ----------------------------
    print("\nSilhouette Scores")

    max_k = min(10, X_scaled.shape[0] - 1)

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        print(f"K={k}  Score={score:.3f}")

    if max_k < 10:
        print(f"\nNote: silhouette scores were computed only for K up to {max_k} because the dataset has {X_scaled.shape[0]} samples.")

    # ----------------------------
    # PCA
    # ----------------------------
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    print("\nExplained Variance Ratio")
    print(pca.explained_variance_ratio_)

    # ----------------------------
    # Final KMeans
    # ----------------------------
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

    clusters = kmeans.fit_predict(X_scaled)

    df["Cluster"] = clusters

    print("\nCluster Counts")
    print(df["Cluster"].value_counts())

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(8,6))

    plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=clusters
    )

    plt.title("Customer Segmentation using PCA")

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.show()

    # ----------------------------
    # Customer Personas
    # ----------------------------
    print("\nCustomer Personas")

    summary = df.groupby("Cluster").mean(numeric_only=True)

    print(summary)

    df.to_csv("Clustered_Customers.csv", index=False)

    print("\nClustered dataset saved successfully.")