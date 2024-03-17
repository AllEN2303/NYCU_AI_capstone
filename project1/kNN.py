import time
import re
import numpy as np
from sklearn.metrics import roc_auc_score
import math
import random

class KNN(object):

    def __init__(self):
        self._word_dict = {}

    def extract_chinese(self, sentence: str) -> str:
        #Remove the symbols
        pattern = "[\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]"
        regex = re.compile(pattern)
        l = regex.findall(sentence)
        return "".join(l)

    def cosine(self, a: set, b: set) -> float:
        #Calculate the distances between two instances
        mod_a = math.sqrt(len(a))
        mod_b = math.sqrt(len(b))
        numerator = len(a & b)
        return numerator / (mod_a * mod_b)

    def read_file(self, filepath: str, polarity: str, n, r_delete=False) -> list:
        # "n" controls the amount of data
        # Define the feature 
        result = []
        j = 0
        with open(filepath, "r", encoding="utf8") as fr:
            for line in fr:
                j += 1
                if j >= n: break
                sentence = line.strip()
                sentence = self.extract_chinese(sentence)
                if len(sentence):
                    if(r_delete): sentence = self.random_deletion(sentence)
                    sentence_vector = set()
                    for i in range(len(sentence) - 1):
                        single_word = sentence[i:i + 1]
                        double_words = sentence[i:i + 2]

                        if single_word not in self._word_dict:
                            self._word_dict[single_word] = len(self._word_dict)
                        if double_words not in self._word_dict:
                            self._word_dict[double_words] = len(self._word_dict)

                        sentence_vector.add(self._word_dict[single_word])
                        sentence_vector.add(self._word_dict[double_words])
                    if sentence[-1] not in self._word_dict:
                        self._word_dict[sentence[-1]] = len(self._word_dict)
                    sentence_vector.add(self._word_dict[sentence[-1]])
                    result.append((sentence_vector, polarity))
        return result

    def top_k(self, sorted_arr: list, K: int, element: tuple) -> None:
        # Find the nearest k neighbors
        flag = 0
        for idx, item in enumerate(sorted_arr):
            if element[0] > item[0]:
                flag = 1
                break
        if flag == 1:
            if len(sorted_arr) == K:
                sorted_arr.pop()
            sorted_arr.insert(idx, element)
        else:
            if len(sorted_arr) != K:
                sorted_arr.append(element)

    def get_most_topK(self, sorted_arr: list) -> str:
        # Find the category
        pos_num = 0
        neg_num = 0
        for distance, category in sorted_arr:
            if category == "positive":
                pos_num += 1
            else:
                neg_num += 1
        return "positive" if pos_num >= neg_num else "negative"

    def classify(self, train: list, test: list, K: int) -> tuple:
        # Calculate the metrics
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for test_sentence, test_category in test:
            sorted_arr = []
            for train_sentence, train_category in train:
                distance = self.cosine(test_sentence, train_sentence)
                self.top_k(sorted_arr, K, (distance, train_category))
            predict_cate = self.get_most_topK(sorted_arr)

            if test_category == "positive" and predict_cate == "positive":
                true_positives += 1
            elif test_category == "negative" and predict_cate == "negative":
                true_negatives += 1
            elif test_category == "positive" and predict_cate == "negative":
                false_negatives += 1
            elif test_category == "negative" and predict_cate == "positive":
                false_positives += 1

        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

        probabilities = []
        for test_sentence, _ in test:
            sorted_arr = []
            for train_sentence, train_category in train:
                distance = self.cosine(test_sentence, train_sentence)
                self.top_k(sorted_arr, K, (distance, train_category))
            neighbors_categories = [category for _, category in sorted_arr]
            probability = neighbors_categories.count("positive") / K
            probabilities.append(probability)

        y_true = np.array([1 if category == "positive" else 0 for _, category in test])
        y_score = np.array(probabilities)
        roc_auc = roc_auc_score(y_true, y_score)

        return accuracy, precision, recall, f1_score, roc_auc

    def cross_validation(self, dataset, K, n_folds=5):
        # Perform cross-validation to evaluate the performance of the KNN classifier.
        # K: The number of neighbors to consider.
        # n_folds: The number of folds for cross-validation (default is 5).
        np.random.shuffle(dataset)  # Shuffle the dataset to ensure randomness
        fold_size = len(dataset) // n_folds
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        roc_aucs = []

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            test_set = dataset[start_idx:end_idx]
            train_set = dataset[:start_idx] + dataset[end_idx:]

            accuracy, precision, recall, f1_score, roc_auc = self.classify(train_set, test_set, K)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            roc_aucs.append(roc_auc)

            print("Fold {}: Accuracy = {:.4f}, Precision = {:.4f}, Recall = {:.4f}, F1 Score = {:.4f}, ROC AUC = {:.4f}".format(
                i + 1, accuracy, precision, recall, f1_score, roc_auc))

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1_score = sum(f1_scores) / len(f1_scores)
        avg_roc_auc = sum(roc_aucs) / len(roc_aucs)

        print("Average Metrics: Accuracy = {:.4f}, Precision = {:.4f}, Recall = {:.4f}, F1 Score = {:.4f}, ROC AUC = {:.4f}".format(
            avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_roc_auc))

        return avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_roc_auc

    def resample(self, dataset):
        #Resample the instances
        pos_samples = [data for data in dataset if data[1] == 'positive']
        neg_samples = [data for data in dataset if data[1] == 'negative']
        num_pos = len(pos_samples)
        num_neg = len(neg_samples)

        oversample_size = max(num_pos, num_neg) - min(num_pos, num_neg)

        if num_pos > num_neg:
            oversampled_neg = random.choices(neg_samples, k=oversample_size)
            dataset.extend(oversampled_neg)
        else:
            oversampled_pos = random.choices(pos_samples, k=oversample_size)
            dataset.extend(oversampled_pos)

        return dataset

    def random_deletion(self, sentence, p=0.5):
        #Random delete the data
        words = sentence.split()
        if len(words) == 1:
            return sentence
        remaining_words = [word for word in words if random.uniform(0, 1) > p]
        if len(remaining_words) == 0:
            return sentence
        else:
            return ' '.join(remaining_words)


if __name__ == "__main__":
    start = time.time()

    solution = KNN()
    train_pos = "train_Badminton.txt"
    train_neg = "train_Stock.txt"
    test_pos = "test_Badminton.txt"
    test_neg = "test_Stock.txt"
    n_Badminton = 2000
    n_Stock = 2000
    test_Badminton = 1000
    test_Stock = 1000
    train_set = solution.read_file(train_pos, "positive", n_Badminton, r_delete=True) + solution.read_file(train_neg, "negative",
                                                                                          n_Stock, r_delete=True)
    test_set = solution.read_file(test_pos, "positive", test_Badminton) + solution.read_file(test_neg, "negative",
                                                                                            test_Stock)

    # train_set = solution.resample(train_set)  # over sampling

    # Cross-validation
    accuracy, precision, recall, f1_score, roc_auc = solution.cross_validation(train_set, K=5)
    print("===========================================")
    print("Cross-Validation Results:")
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1_score))
    print("AUROC: {:.2f}".format(roc_auc))
    print("===========================================")

    accuracy, precision, recall, f1_score, roc_auc = solution.classify(train_set, test_set, K=5)
    print("Final Test: ")
    print("===========================================")
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1_score))
    print("AUROC: {:.2f}".format(roc_auc))
    print("===========================================")
    end = time.time()
    print("Execution time: {:.2f} seconds".format(end - start))
