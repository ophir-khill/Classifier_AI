import subprocess

import numpy as np

from KNN import KNNClassifier
from utils import *

target_attribute = 'Outcome'


def run_knn(k, x_train, y_train, x_test, y_test, formatted_print=True):
    neigh = KNNClassifier(k=k)
    neigh.train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    acc = accuracy(y_test, y_pred)
    print(f'{acc * 100:.2f}%' if formatted_print else acc)


def get_top_b_features(x, y, b=5, k=51):
    """
    :param k: Number of nearest neighbors.
    :param x: array-like of shape (n_samples, n_features).
    :param y: array-like of shape (n_samples,).
    :param b: number of features to be selected.
    :param method: sklearn method
    :return: indices of top 'b' features as the result of selection/dimensionality reduction on sample
            sets using sklearn.feature_selection module
    """
    # TODO: Implement get_top_b_features function
    #   - Note: The brute force approach which examines all subsets of size `b` will not be accepted.

    assert 0 < b < x.shape[1], f'm should be 0 < b <= n_features = {x.shape[1]}; got b={b}.'
    max_val = 0
    max_b_feature = []
    for b_val in range(x.shape[1]-1):
        for i in range(10):
            b_feature, val = b_aux2(x, y, b_val+1, k)
            if val >= max_val:
                max_val = val
                max_b_feature = b_feature
    return max_b_feature


def b_aux2(x,y,b=5,k=51):
    x_copy = x.copy()
    feature_count = x.shape[1]
    top_b_features_indices = [x for x in range(feature_count)]
    # ====== YOUR CODE: ======
    n_samples = x.shape[0]
    for a in range(feature_count - b):
        min_acc = np.inf
        worst_feature_index = None
        for j in range(x_copy.shape[1]):
            if j in top_b_features_indices:
                array = []
                shuffle_array = []
                top_b_features_indices.remove(j)
                x = x_copy[:, top_b_features_indices]
                for r in range(5):
                    shuffle_array.append(np.random.permutation(len(x)))
                for shuffle in shuffle_array:
                    x = np.array(x[shuffle])
                    y = np.array(y[shuffle])
                    a = int(0.8 * n_samples)
                    x_train1 = x[:a]
                    y_train1 = y[:a]
                    x_valid = x[a:]
                    y_valid = y[a:]
                    neigh = KNNClassifier(k=k)
                    neigh.train(x_train1, y_train1)
                    y_pred = neigh.predict(x_valid)
                    acc = accuracy(y_valid, y_pred)
                    array.append(acc)
                mean_acc = np.mean(array)
                if mean_acc < min_acc:
                    min_acc = mean_acc
                    worst_feature_index = j
                top_b_features_indices.append(j)
        top_b_features_indices.remove(worst_feature_index)
    a = int(0.8 * n_samples)
    x_train1 = x_copy[:a]
    y_train1 = y[:a]
    x_valid = x_copy[a:]
    y_valid = y[a:]
    neigh = KNNClassifier(k=k)
    neigh.train(x_train1, y_train1)
    y_pred = neigh.predict(x_valid)
    acc = accuracy(y_valid, y_pred)
    return np.sort(top_b_features_indices), acc


def run_cross_validation():
    """
    cross validation experiment, k_choices = [1, 5, 11, 21, 31, 51, 131, 201]
    """
    file_path = str(pathlib.Path(__file__).parent.absolute().joinpath("KNN_CV.pyc"))
    subprocess.run(['python', file_path])


def exp_print(to_print):
    print(to_print + ' ' * (30 - len(to_print)), end='')


# ========================================================================
if __name__ == '__main__':
    """
       Usages helper:
       (*) cross validation experiment
            To run the cross validation experiment over the K,Threshold hyper-parameters
            uncomment below code and run it
    """
    # run_cross_validation()

    # # ========================================================================

    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')

    best_k = 51
    b = 5

    # # ========================================================================
    print("-" * 10 + f'k  = {best_k}' + "-" * 10)
    exp_print('KNN in raw data: ')
    run_knn(best_k, x_train, y_train, x_test, y_test)
    top_m = get_top_b_features(x_train, y_train, b=5, k=best_k)
    x_train_new = x_train[:, top_m]
    x_test_test = x_test[:, top_m]
    exp_print(f'KNN in selected feature data: ')
    run_knn(best_k, x_train_new, y_train, x_test_test, y_test)
