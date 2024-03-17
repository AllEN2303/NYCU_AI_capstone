from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
class KmeansEvaluator():
    def __init__(self, corpus_path, stopwords_path=None):
        self.corpus_path = corpus_path
        self.stopwords_path = stopwords_path

    def preprocess_data(self):
        #Every comment is seen a data
        corpus = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(line.strip())
        return corpus

    def evaluate_kmeans(self, n_clusters_list):
        # perform feature extraction
        # Calculate the WCSS and silhouette_scores to evaluate the models
        corpus = self.preprocess_data()
        vectorizer = CountVectorizer(stop_words=self.load_stopwords(), max_df=0.5, max_features=1000)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        weights = tfidf.toarray()
        wcss_scores = []
        silhouette_scores = []
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(weights)
            wcss_scores.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(weights, kmeans.labels_))

        return wcss_scores, silhouette_scores

    def plot_evaluation(self, n_clusters_list, wcss_scores, silhouette_scores):
        # Plot the evaluation with different numbers of clusters 
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(n_clusters_list, wcss_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.title('Within-Cluster Sum of Squares (WCSS)')

        plt.subplot(1, 2, 2)
        plt.plot(n_clusters_list, silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')

        plt.tight_layout()
        plt.show()

    def load_stopwords(self):
        stopwords = []
        if self.stopwords_path:
            with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f]
        return stopwords
def merge_files(train_pos, train_neg, test_data, pos_lines=0, neg_lines=0):
    with open(train_pos, 'r', encoding='utf-8') as pos_file:
        pos_data = [next(pos_file).strip() for _ in range(pos_lines)]
    
    with open(train_neg, 'r', encoding='utf-8') as neg_file:
        neg_data = [next(neg_file).strip() for _ in range(neg_lines)]

    merged_data = pos_data + neg_data

    with open(test_data, 'w', encoding='utf-8') as test_file:
        for line in merged_data:
            test_file.write(line + '\n')
if __name__ == '__main__':
    n_Badminton = 1500
    n_Stock = 1500
    merge_files("train_Badminton.txt", "train_Stock.txt", "test_data.txt", n_Badminton, n_Stock)

    corpus_path = 'test_data.txt'
    stopwords_path = 'stop_words.txt'
    evaluator = KmeansEvaluator(corpus_path, stopwords_path)
    n_clusters_list = [3, 5, 7 , 9 ,11]

    wcss_scores, silhouette_scores = evaluator.evaluate_kmeans(n_clusters_list)
    #print(wcss_scores)
    print(silhouette_scores)
    #evaluator.plot_evaluation(n_clusters_list, wcss_scores, silhouette_scores)
    