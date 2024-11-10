import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import yaml
import os

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(data_path, embedding_path):
    df = pd.read_csv(data_path)
    with open(embedding_path, 'rb') as f:
        embedding_dict = pickle.load(f)
    df['Embedding'] = df['Image'].map(embedding_dict)
    return df

def process_data(train_path, train_embedding_path, valid_path, valid_embedding_path, dev_path, dev_embedding_path, max_neighbors, similarity_thresholds, output_dir):
    df_train = load_data(train_path, train_embedding_path)
    df_valid = load_data(valid_path, valid_embedding_path)
    df_merged = pd.concat([df_train, df_valid], ignore_index=True)

    X_merged = np.array(df_merged['Embedding'].to_list()).squeeze()
    knn_merged = NearestNeighbors(metric='cosine')
    knn_merged.set_params(n_neighbors=max_neighbors)
    knn_merged.fit(X_merged)

    df_dev = load_data(dev_path, dev_embedding_path)

    mean_neighbors_list = []
    similarity_values = []

    for similarity_threshold in similarity_thresholds:
        neighbors_count = []
        neighbor_info_dict = {}

        for index, row in tqdm(df_dev.iterrows(), total=len(df_dev), desc=f"Processing for {max_neighbors} neighbors at similarity {similarity_threshold}"):
            test_image_path = row['Image'] + '.jpg'
            test_image_embedding = row['Embedding']
            neighbor_info_dict[test_image_path] = []

            if test_image_embedding is not None:
                distances_merged, indices_merged = knn_merged.kneighbors(test_image_embedding.reshape(1, -1))

                for neighbor_index, cosine_distance in zip(indices_merged[0], distances_merged[0]):
                    similarity = 1 - cosine_distance

                    if similarity >= similarity_threshold:
                        neighbor_image = df_merged.iloc[neighbor_index]['Image'] + '.jpg'
                        neighbor_caption = df_merged.iloc[neighbor_index]['Caption']
                        neighbor_info_dict[test_image_path].append({
                            'Neighbor': neighbor_image,
                            'Caption': neighbor_caption,
                            'Similarity': similarity
                        })

                neighbors_count.append(len(neighbor_info_dict[test_image_path]))
                similarity_values.extend([info['Similarity'] for info in neighbor_info_dict[test_image_path]])

        mean_neighbors = np.mean(neighbors_count)
        mean_neighbors_list.append(mean_neighbors)
        print(f"Mean number of neighbors found at similarity {similarity_threshold}: {mean_neighbors}")

        # Save neighbor information to CSV
        output_csv = os.path.join(output_dir, f"dev_neighbors_{similarity_threshold:.2f}_1_neighbor.csv")
        df_neighbors = pd.DataFrame([
            {
                'Test Image': test_image,
                'Neighbor Images': ', '.join([neighbor['Neighbor'] for neighbor in neighbors]),
                'Neighbor Captions': ', '.join([neighbor['Caption'] for neighbor in neighbors]),
                'Similarities': ', '.join([f"{neighbor['Similarity']:.2f}" for neighbor in neighbors])
            }
            for test_image, neighbors in neighbor_info_dict.items()
        ])
        df_neighbors.to_csv(output_csv, index=False)

    return similarity_thresholds, mean_neighbors_list, similarity_values

def main():
    config = load_config()

    train_path = config['paths']['train_dataset_path']
    train_embedding_path = config['paths']['train_embedding_path']
    valid_path = config['paths']['valid_dataset_path']
    valid_embedding_path = config['paths']['valid_embedding_path']
    dev_path = config['paths']['dev_dataset_path']
    dev_embedding_path = config['paths']['dev_embedding_path']
    output_dir = config['paths']['output_dir']

    max_neighbors = config['hyperparameters']['max_neighbors']
    similarity_thresholds = config['hyperparameters']['similarity_thresholds']

    thresholds, mean_neighbors, similarity_values = process_data(
        train_path, train_embedding_path, valid_path, valid_embedding_path, dev_path, dev_embedding_path, max_neighbors, similarity_thresholds, output_dir
    )

    # Save mean neighbors to a text file
    output_txt = os.path.join(output_dir, 'mean_neighbors_1.txt')
    with open(output_txt, 'w') as f:
        for threshold, mean in zip(thresholds, mean_neighbors):
            f.write(f"{threshold}: {mean}\n")

    # Create bar plot for mean neighbors
    plt.figure(figsize=(10, 6))
    plt.bar(thresholds, mean_neighbors, width=0.03, align='center', alpha=0.7)
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Mean Number of Neighbors')
    plt.title('Mean Number of Neighbors by Similarity Threshold')
    plt.xticks(thresholds)
    plt.grid(True)

    # Save the plot as a PDF
    output_pdf = os.path.join(output_dir, 'mean_neighbors_plot_1.pdf')
    plt.savefig(output_pdf)
    plt.show()

    # Create histogram of similarity values
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_values, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Neighbor Similarities')
    plt.grid(True)

    # Save the similarity histogram as a PDF
    similarity_pdf = os.path.join(output_dir, 'similarity_distribution.pdf')
    plt.savefig(similarity_pdf)
    plt.show()

if __name__ == "__main__":
    main()
