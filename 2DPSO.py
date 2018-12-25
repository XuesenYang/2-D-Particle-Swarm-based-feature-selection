import numpy as np
import random
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold


class PSO:
    def __init__(self, X, y, pN, dim, max_iter):
        self.w = 0.729
        self.c1 = 1.49
        self.c2 = 1.49
        self.pN = pN
        self.RG = 3
        self.max_iter = max_iter
        self.X = X
        self.y = y
        self.dim = dim
        self.x = np.zeros((self.pN, self.dim), dtype=np.bool)  # 所有粒子的位置
        self.v = np.random.random((self.pN, 2, self.dim))  # 所有粒子的速度，已随机生成
        self.pbest = np.zeros((self.pN, self.dim), dtype=np.bool)
        self.gbest = np.zeros((1, self.dim), dtype=np.bool)
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值

    def velocity_update(self, vi, xi, pbesti, gbest, Δi):
        r1 = random.random()
        r2 = random.random()
        length = self.dim  # len(xi)
        φself = np.zeros(length, dtype=np.bool)
        φcog = np.zeros(length, dtype=np.bool)
        φsoc = np.zeros(length, dtype=np.bool)
        ψcog = np.zeros(length, dtype=np.bool)
        ψsoc = np.zeros(length, dtype=np.bool)
        # 求基数集φ
        φself[np.count_nonzero(xi) - 1] = True
        φcog[np.count_nonzero(pbesti) - 1] = True
        φsoc[np.count_nonzero(gbest) - 1] = True
        # 求特征集ψ
        ψself = xi[:]
        for i in range(length):
            ψcog[i] = pbesti[i] and (not xi[i])
            ψsoc[i] = gbest[i] and (not xi[i])
        # 求最终学习集L
        Lcog = np.array([φcog, ψcog], dtype=int)
        Lsoc = np.array([φsoc, ψsoc], dtype=int)
        Lself = np.array([φself, ψself], dtype=int)
        # 速度更新
        vi = (self.w * vi) + (self.c1 * r1 * Lcog) + (self.c2 * r2 * Lsoc) + (Δi * Lself)
        return vi

    def position_update(self, vi):
        ρ = vi[0]
        σ = vi[1]
        length = len(ρ)
        xi = np.zeros(length, dtype=np.bool)
        n = 0
        r = random.uniform(0, sum(ρ))
        # 轮盘赌算法求基数ξi
        for i in range(length):
            n += ρ[i]
            if (n >= r):
                ξi = i + 1
                break
        # 从大到小排序并取得下标
        index = np.argsort(σ)
        index = index[::-1]
        # 取前ξi大的特征放入xi
        for i in range(ξi):
            xi[index[i]] = 1
        return xi

    # 目标函数
    def func(self, xi):
        arr = []
        # 根据xi提取特征X的列
        for i in range(self.dim):
            if (xi[i] == True):
                arr.append(i)
        X = self.X[:, arr][:]
        y = self.y[:]
        # 10折分层交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=False)
        skf.get_n_splits(X, y)
        fit = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # 放入kNN分类器计算错误率
            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)
            #knn.predict(X_test)
            score = knn.score(X_test, y_test)
            fit += score
        fit /= 10
        return 1 - fit

    # 初始化种群
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.x[i][j] = random.choice([True, False])
            self.x[i][random.randint(0, self.dim - 1)] = True

        for i in range(self.pN):
            self.pbest[i] = self.x[i]
            tmp = self.func(self.x[i])
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.x[i]

    def iterator(self):
        #fitness = []
        δ = np.zeros(self.pN)
        Δ = np.zeros(self.pN)
        flag = np.zeros(self.pN)
        for t in range(self.max_iter):
            for i in range(self.pN):
                flag[i] = -1
                # 更新gbest、pbest、个体最优、全局最优
                temp = self.func(self.x[i])
                if (temp < self.p_fit[i]):
                    self.p_fit[i] = temp
                    self.pbest[i] = self.x[i]
                    flag[i] = 1
                    if (self.p_fit[i] < self.fit):
                        self.gbest = self.x[i]
                        self.fit = self.p_fit[i]

            print(self.fit)
            for i in range(self.pN):
                δ[i] = 1 - self.p_fit[i] / max(self.p_fit)
                Δ[i] = flag[i] * δ[i]
                # 对每个粒子进行速度与位置更新
                self.v[i] = self.velocity_update(self.v[i], self.x[i], self.pbest[i], self.gbest, Δ[i])
                self.x[i] = self.position_update(self.v[i])
            #fitness.append(self.fit)
        return self.gbest, self.fit


def feature_selection(X, y):
    dim = len(X[0])
    myPSO = PSO(X, y, 30, dim, 20)
    myPSO.init_Population()
    gbest, fitness = myPSO.iterator()
    print('gbest:',gbest)
    print('fitness:',fitness)
    arr = []
    # 根据gbest提取X的特征
    for i in range(dim):
        if (gbest[i] == True):
            arr.append(i)
    return X[:, arr][:]


if __name__ == '__main__':
    #iris = datasets.load_iris()
    wine = datasets.load_wine()
    data = wine.data
    target = wine.target
    X = feature_selection(data, target)
