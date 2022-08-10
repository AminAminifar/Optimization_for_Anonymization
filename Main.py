from evaluate import evaluation, print_results
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from find_clusters import find_clusters
from scipy.stats import entropy
from numpy.linalg import norm
import warnings


warnings.filterwarnings(action='ignore')
warnings.simplefilter("ignore")

data_all = pd.read_csv('data/heart/clevland.csv', header=None)
data_mat = np.zeros((282, 75))
for i in range(282):
    list = []
    for j in range(10):
        index = i * 10 + j
        vec = pd.to_numeric(data_all.iloc[index, :], errors='coerce')
        for k in range(8):
            if not np.isnan(vec[k]):
                list.append(vec[k])
    data_mat[i, :] = np.array(list)


sensitive_att_famhist = data_mat[:, 17]


data_temp = np.genfromtxt('data/heart/processed.cleveland.data', delimiter=',')
data_set = data_temp[0:282, :]
data_set = np.append(sensitive_att_famhist.reshape(-1,1), data_set, axis=1)


for i in range(0, len(data_set[:, 0])):
    if data_set[i, -1] != 0:
        data_set[i, -1] = 1


def data_preprocessing_for_missing_values (train, test):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(train)
    train = imp.transform(train)
    test = imp.transform(test)
    return train, test


def preparedata(data_set):

    # split data to train and test sets

    train_set, test_set = train_test_split(data_set, train_size=200)

    train_set[:, :-1], test_set[:, :-1] = data_preprocessing_for_missing_values(train_set[:, :-1], test_set[:, :-1])

    return train_set, test_set


def outputresults(groundTruth_vec, prediction_ert_vec, prediction_rf_vec, prediction_xgb_vec, prediction_dt_vec, prediction_ls_vec, prediction_rbfsvm):
    print("groundTruth_vec and prediction_ert_vec:")
    print_results(groundTruth_vec, prediction_ert_vec)
    print("groundTruth_vec and prediction_rf_vec:")
    print_results(groundTruth_vec, prediction_rf_vec)
    print("groundTruth_vec and prediction_xgb_vec:")
    print_results(groundTruth_vec, prediction_xgb_vec)
    print("groundTruth_vec and prediction_dt_vec:")
    print_results(groundTruth_vec, prediction_dt_vec)
    print("groundTruth_vec and prediction_ls_vec:")
    print_results(groundTruth_vec, prediction_ls_vec)
    print("groundTruth_vec and prediction_rbfsvm:")
    print_results(groundTruth_vec, prediction_rbfsvm)


def calculate_entropy_l_diversity(sensitive_attribute, cluster_num, k):
    entropy_vals = np.zeros((k, 1))
    l_vals = np.zeros((k, 1))
    D = np.zeros((k, 1))
    l = np.zeros((k, 1))
    c_min_for_l_max = np.zeros((k, 1))
    unique1, counts1 = np.unique(sensitive_attribute, return_counts=True)
    print("FOR ALL: ", "unique:", unique1, ", counts: ", counts1)
    p = np.zeros((2,1))
    p[0] = counts1[0]/sum(counts1)
    p[1] = counts1[1]/sum(counts1)
    for cluster in range(k):
        unique, counts = np.unique(sensitive_attribute[cluster_num==cluster], return_counts=True)
        print("unique:", unique,", counts: ", counts)
        q = np.zeros((2, 1))
        if len(unique)==2:
            q[0] = counts[0] / sum(counts)
            q[1] = counts[1] / sum(counts)
        else:
            if unique[0]==0:
                q[0] = 1
                q[1] = 0
            else:
                q[0] = 0
                q[1] = 1
        D[cluster] = norm((p-q), 1)
        entropy_vals[cluster] = entropy(counts, base=2)
        l_vals[cluster] = 2 ** (entropy_vals[cluster])
        l[cluster] = len(unique)
        c_min_for_l_max[cluster] = max(counts)/min(counts)

        # temp
        print("cluster ", cluster, ": entropy = ", entropy_vals[cluster], ", value of l (entropy l-diversity)= ", l_vals[cluster],
              ", value of l = ", l[cluster], "min c for max l ((c,l)-diversity)", c_min_for_l_max[cluster], ", value of D = ", D[cluster])
    return min(l_vals), max(D), min(l), max(c_min_for_l_max)


def anonymize(train_set):
    n = 200
    m = 13  # number of QID features
    k = 10  # 13  # number of clusters

    sensitive_attribute = train_set[0:n, 0]
    sensitive_attribute = sensitive_attribute.reshape(-1, 1)
    for i in range(n):
        if sensitive_attribute[i] != 1:
            sensitive_attribute[i] = 0

    OIDs = train_set[0:n, 1:14]

    C, B, attribute_range, output_data, cluster_num, kmeans_anonymized_data, kmeans_labels = \
        find_clusters(n, m, k, OIDs, sensitive_attribute, alpha=1.8,
                      anonymization_type="with l-diversity consideration")

    C2, B2, attribute_range2, output_data2, cluster_num2, kmeans_anonymized_data, kmeans_labels = \
        find_clusters(n, m, k, OIDs, sensitive_attribute, alpha=1.8,
                      anonymization_type="without l-diversity consideration")

    data_new = np.append(train_set[0:n, 0].reshape(-1, 1), output_data, axis=1)
    data_new = np.append(data_new, train_set[0:n, -1].reshape(-1, 1), axis=1)
    data_new2 = np.append(train_set[0:n, 0].reshape(-1, 1), output_data2, axis=1)
    data_new2 = np.append(data_new2, train_set[0:n, -1].reshape(-1, 1), axis=1)

    # print clusters info
    print_cluster_infor_flag = True
    if print_cluster_infor_flag:
        print("Privacy for kmeans annymization")
        min_l_e_v1, max_D_v1, min_l_v1, max_c_min_for_l_max_v1 = calculate_entropy_l_diversity(sensitive_attribute, cluster_num, k)
        print("Privacy for our approach annymization without l-diversity consideration")
        min_l_e_v2, max_D_v2, min_l_v2, max_c_min_for_l_max_v2 = calculate_entropy_l_diversity(sensitive_attribute, cluster_num2, k)

    return data_new, data_new2, min_l_e_v1, max_D_v1, min_l_v1, max_c_min_for_l_max_v1, min_l_e_v2, max_D_v2, min_l_v2, max_c_min_for_l_max_v2


def evaluate_and_update_lists(trainset, testset, list_groundTruth, list_prediction_ert, list_prediction_rf,
                              list_prediction_xgb, list_prediction_dt, list_prediction_ls, list_prediction_rbfsvm,
                              update_groundTruth=False):

    groundTruth, prediction_ert, prediction_rf, prediction_xgb, prediction_dt, prediction_ls, prediction_rbfsvm = \
        evaluation(trainset, testset)

    if update_groundTruth:
        list_groundTruth.append(groundTruth)
    list_prediction_ert.append(prediction_ert)
    list_prediction_rf.append(prediction_rf)
    list_prediction_xgb.append(prediction_xgb)
    list_prediction_dt.append(prediction_dt)
    list_prediction_ls.append(prediction_ls)
    list_prediction_rbfsvm.append(prediction_rbfsvm)
    return


def list_to_vec_and_print_results(list_groundTruth, list_prediction_ert, list_prediction_rf,
                              list_prediction_xgb, list_prediction_dt, list_prediction_ls, list_prediction_rbfsvm):

    groundTruth_vec = np.concatenate(np.asarray(list_groundTruth))
    prediction_ert_vec = np.concatenate(np.asarray(list_prediction_ert))
    prediction_rf_vec = np.concatenate(np.asarray(list_prediction_rf))
    prediction_xgb_vec = np.concatenate(np.asarray(list_prediction_xgb))
    prediction_dt_vec = np.concatenate(np.asarray(list_prediction_dt))
    prediction_ls_vec = np.concatenate(np.asarray(list_prediction_ls))
    prediction_rbfsvm_vec = np.concatenate(np.asarray(list_prediction_rbfsvm))

    print("Shape groundTruth_vec: ", np.shape(groundTruth_vec), " Shape prediction_ert_vec", np.shape(prediction_ert_vec))

    outputresults(groundTruth_vec, prediction_ert_vec, prediction_rf_vec, prediction_xgb_vec, prediction_dt_vec,
                  prediction_ls_vec, prediction_rbfsvm_vec)


def pipeline(data_set, num_iterations=1):

    # declare lists
    list_groundTruth_org, list_prediction_ert_org, list_prediction_rf_org, list_prediction_xgb_org, list_prediction_dt_org,\
    list_prediction_ls_org, list_prediction_rbfsvm_org = [], [], [], [], [], [], []

    list_prediction_ert_v1, list_prediction_rf_v1, list_prediction_xgb_v1, list_prediction_dt_v1,\
    list_prediction_ls_v1, list_prediction_rbfsvm_v1 = [], [], [], [], [], []

    list_prediction_ert_v2, list_prediction_rf_v2, list_prediction_xgb_v2, list_prediction_dt_v2,\
    list_prediction_ls_v2, list_prediction_rbfsvm_v2 = [], [], [], [], [], []

    global_min_l_e_v1 = 1000
    global_min_l_e_v2 = 1000
    global_min_l_v1 = 1000
    global_min_l_v2 = 1000
    global_max_D_v1 = 0
    global_max_D_v2 = 0
    global_max_c_min_for_l_max_v1 = 0
    global_max_c_min_for_l_max_v2 = 0

    # have a loop
    for i in range(num_iterations):
        print("Iteration round: ", i)

        # prepare data
        train_set, test_set = preparedata(data_set)

        # anonymize
        train_set_v1, train_set_v2, \
        min_l_e_v1, max_D_v1, min_l_v1, max_c_min_for_l_max_v1,\
        min_l_e_v2, max_D_v2, min_l_v2, max_c_min_for_l_max_v2 = anonymize(train_set)

        # worst clustering with respect to privacy
        global_min_l_e_v1 = min(global_min_l_e_v1, min_l_e_v1)
        global_min_l_e_v2 = min(global_min_l_e_v2, min_l_e_v2)
        global_min_l_v1 = min(global_min_l_v1, min_l_v1)
        global_min_l_v2 = min(global_min_l_v2, min_l_v2)
        global_max_D_v1 = max(global_max_D_v1, max_D_v1)
        global_max_D_v2 = max(global_max_D_v2, max_D_v2)
        global_max_c_min_for_l_max_v1 = max(global_max_c_min_for_l_max_v1, max_c_min_for_l_max_v1)
        global_max_c_min_for_l_max_v2 = max(global_max_c_min_for_l_max_v2, max_c_min_for_l_max_v2)

        # evaluate on original data
        evaluate_and_update_lists(train_set, test_set, list_groundTruth_org, list_prediction_ert_org, list_prediction_rf_org,
                   list_prediction_xgb_org, list_prediction_dt_org, list_prediction_ls_org, list_prediction_rbfsvm_org,
                   update_groundTruth=True)

        # evaluate on data anonymzation with our approach
        evaluate_and_update_lists(train_set_v1, test_set, [], list_prediction_ert_v1, list_prediction_rf_v1, list_prediction_xgb_v1,
                   list_prediction_dt_v1, list_prediction_ls_v1, list_prediction_rbfsvm_v1)

        # evaluate on data anonymzation without diversity constraint
        evaluate_and_update_lists(train_set_v2, test_set, [], list_prediction_ert_v2, list_prediction_rf_v2, list_prediction_xgb_v2,
                   list_prediction_dt_v2, list_prediction_ls_v2, list_prediction_rbfsvm_v2)

    # list_to_vec_and_print_results

    # org data
    print("=====================================")
    print("ORIGINAL DATA:")
    list_to_vec_and_print_results(list_groundTruth_org, list_prediction_ert_org, list_prediction_rf_org,
                                  list_prediction_xgb_org, list_prediction_dt_org, list_prediction_ls_org,
                                  list_prediction_rbfsvm_org)

    # anonymzation with our approach
    print("=====================================")
    print("ANONYMIZATION OUR APPROACH:")
    list_to_vec_and_print_results(list_groundTruth_org, list_prediction_ert_v1, list_prediction_rf_v1,
                                  list_prediction_xgb_v1, list_prediction_dt_v1, list_prediction_ls_v1,
                                  list_prediction_rbfsvm_v1)
    print("Privacy results for entropy l-diversity and t-closeness:")
    print("Minimum l : ", global_min_l_v1)
    print("Minimum l (entropy l-diversity): ", global_min_l_e_v1)
    print("Maximum D: ", global_max_D_v1)
    print("Maximum c for maximum l (recursive (c,l)-diversity): ", global_max_c_min_for_l_max_v1)

    # anonymzation without diversity constraint
    print("=====================================")
    print("ANONIZATION WITHOUT DIVERSITY CONSTRAINT:")
    list_to_vec_and_print_results(list_groundTruth_org, list_prediction_ert_v2, list_prediction_rf_v2,
                                  list_prediction_xgb_v2, list_prediction_dt_v2, list_prediction_ls_v2,
                                  list_prediction_rbfsvm_v2)
    print("Privacy results for entropy l-diversity and t-closeness:")
    print("Minimum l : ", global_min_l_v2)
    print("Minimum l (entropy l-diversity): ", global_min_l_e_v2)
    print("Maximum D: ", global_max_D_v2)
    print("Maximum c for maximum l (recursive (c,l)-diversity): ", global_max_c_min_for_l_max_v2)

# original_stdout = sys.stdout
# with open('terminal.txt', 'w') as f:
#     sys.stdout = f
#     pipeline(data_set, num_iterations=100)


pipeline(data_set, num_iterations=10)  # num_iterations=1000
