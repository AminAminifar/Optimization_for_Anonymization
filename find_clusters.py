import cvxpy as cp
import numpy as np
import time
from sklearn.cluster import KMeans
import copy


def find_clusters(n, m, k, QID_attributes_values, sensitive_attribute, alpha=1.80, anonymization_type="with l-diversity consideration"):
    C = cp.Variable((k, m))  # centers to be optimized
    B = cp.Variable((n, k), boolean=True)  # Coefficients (0 or 1) to be optimized
    X = QID_attributes_values  # (m) samples (QID attributes)
    C_2 = cp.Variable(k)
    margin = cp.Variable((k, 1))

    #################

    kmeans = KMeans(n_clusters=k).fit(X)
    SKL_C = kmeans.cluster_centers_

    kmeans_labels = kmeans.labels_

    kmeans_anonymized_data = copy.deepcopy(QID_attributes_values)
    cluster_num = np.zeros((n, 1))
    for i in range(n):
        kmeans_anonymized_data[i,:] = SKL_C[kmeans_labels[i]]
    kmeans_anonymized_data = np.array(kmeans_anonymized_data)
    #################

    # objective
    Lambda = 1000
    main_sum = 0
    for j in range(k):  # cluster
        for i in range(n):  # sample
            norm_2 = cp.norm(B[i, j] * (X[i, :] - SKL_C[j, :]), 1)   #
            # norm_2 = cp.norm(C[j, :] - SKL_C[j, :], 1)
            sum_operand = norm_2   #
            main_sum = main_sum + sum_operand
    objective = cp.Minimize(main_sum+cp.norm(Lambda*margin, 1))

    # constraints
    const = []
    for i in range(n):  # sample
        const_val = 0
        for j in range(k):  # cluster
            const_val = const_val + B[i, j]
        const.append(const_val==1)

    for j in range(k):  # cluster
        const_val = 0
        for i in range(n):  # sample
            const_val = const_val + B[i, j]
        const.append(const_val == n/k)  # n/k

    for j in range(k):  # cluster
        for f in range(m):  # feature
            const_val = 0
            for i in range(n):  # sample
                const_val = const_val + (B[i, j] * X[i, f])
            const_val = const_val/(n/k) - C[j, f]
            const.append(const_val == 0)

    if anonymization_type == "with l-diversity consideration":
        for j in range(k):  # cluster
            const_val = 0
            for i in range(n):  # sample
                const_val = const_val + (B[i, j] * sensitive_attribute[i])
            ratio = 1
            const.append(const_val <= (ratio + margin[j])*sum(sensitive_attribute)/k)
            const.append(margin[j]>=0)
    else:
        const.append(margin[:] == 0)

    p = cp.Problem(objective, const)  # cp.Problem(alpha*objective, const)
    print("before solve")
    start_time = time.time()
    p.solve(solver=cp.GUROBI, verbose = False, max_iters = 1)
    elapsed_time = time.time() - start_time
    print("done!")
    print("elapsed time:", elapsed_time)

    print(p.status)
    # print(B.value)
    # print(C.value)

    attribute_range = np.empty((k, m, 2))

    anonymized_data = copy.deepcopy(QID_attributes_values)
    cluster_num = np.zeros((n, 1))
    for i in range(n):
        cluster = np.where(np.round(B.value[i, :]) == 1)[0]
        cluster_num[i] = cluster
        anonymized_data[i,:] = C.value[cluster,:]
    anonymized_data = np.array(anonymized_data)

    return C, B, attribute_range, anonymized_data, cluster_num, kmeans_anonymized_data, kmeans_labels


