import matplotlib
import matplotlib.pyplot as plt
import numpy as np
class MF():

    def __init__(self, train_R,test_R, K, lr, reg_u, reg_v, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.train_R = train_R
        self.test_R = test_R
        self.num_users, self.num_items = train_R.shape
        self.K = K
        self.lr = lr
        self.alpha = lr
        self.reg_u = reg_u
        self.reg_v = reg_v
        self.iterations = iterations
        self.confidence = np.zeros((self.train_R.shape))
        self.confidence[self.train_R == 1] = 1
        self.confidence[self.train_R != 1] = 0.01


    def get_samples(self, add_negative_samples):
        # Create a list of training samples
        self.samples = []
        for i in range(self.num_users):
            train_item_idx = np.nonzero(self.train_R[i])
            test_item_idx = np.nonzero(self.test_R[i])
            [self.samples.append((i,j,self.train_R[i,j])) for j in train_item_idx[0]]
            if add_negative_samples:
                negativ_samples = self.get_negative_samples(train_item_idx[0], test_item_idx[0])
                #add negative samples
                [self.samples.append((i,j,0)) for j in negativ_samples]

        print('Generate samples: \n %d samples have been generated ' % len(self.samples))


    def get_negative_samples(self, train_idx, test_idx):
        idx = np.arange(self.num_items, dtype=np.int32)
        mask = np.ones(self.num_items, dtype=bool)
        mask[[train_idx]] = False
        # mask[[test_idx]] = False
        negative_idx = idx[mask]
        np.random.seed(42)
        np.random.shuffle(negative_idx)
        return negative_idx[:len(train_idx)]

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b =  0 #np.mean(self.train_R[np.where(self.train_R != 0)])

        self.get_samples(add_negative_samples=False)

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            # if (i+1) % 10 == 0:
            print("Iteration: %d ; error = %.4f" % (i+1, mse))
            self.decay_lr(i)

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.train_R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.train_R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = self.confidence[i,j] * (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.reg_u * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.reg_v * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.reg_u * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.reg_v * self.Q[j, :])

    def decay_lr(self, step):
        decay_rate =0.70
        decay_steps = 10
        self.alpha = self.lr * np.power(decay_rate,(step / decay_steps))
        print('lr %f' % self.alpha)

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)


def read_dataset(path,num_user,num_item):
    M = np.zeros([num_user, num_item], dtype=int)
    with open(path, 'r') as f:
        i =0
        for line in f.readlines():
            items_idx= line.split()[1:]
            items_idx = [int(x) - 1 for x in items_idx]
            user_id = i  # 0 base index
            rating = 1
            M[user_id, items_idx] = rating
            i +=1
    return M

R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
ratings_path = '/home/wanli/data/Extended_ctr/citeulike_a_extended/users.dat'

for fold in range(3,6):
    train_ratings_path = '/home/wanli/data/Extended_ctr/citeulike_a_extended/in-matrix-item_folds/fold-{0}/train-fold_{0}-users.dat'.format(fold)
    R_train = read_dataset(train_ratings_path,5551,16980)
    test_ratings_path = '/home/wanli/data/Extended_ctr/citeulike_a_extended/in-matrix-item_folds/fold-{0}/test-fold_{0}-users.dat'.format(fold)
    R_test = read_dataset(test_ratings_path,5551,16980)
    # ratings_path = '/home/wanli/data/Extended_ctr/dummy/users.dat'
    # R = read_dataset(ratings_path,50,1929)

    mf = MF(R_train,R_test, K=200, lr=0.1, reg_u=0.01,reg_v=0.01, iterations=100)
    training_process = mf.train()
    print()
    # print("P x Q:")
    # print(mf.full_matrix())
    # print()
    np.save('/home/wanli/data/Extended_ctr/citeulike_a_extended/in-matrix-item_folds/fold-{0}/score.npy'.format(fold),mf.full_matrix())
    # print("Global bias:")
    # print(mf.b[:10])
    # print()
    # print("User bias:")
    # print(mf.b_u[:10])
    # print()
    # print("Item bias:")
    # print(mf.b_i[:10])

# accuracy = np.mean(np.power(R -mf.full_matrix(),2))

x = [x for x, y in training_process]
y = [y for x, y in training_process]
plt.figure(figsize=((16,4)))
plt.plot(x, y)
plt.xticks(x, x)
plt.xlabel("Iterations")
plt.ylabel("Mean Square Error")
plt.grid(axis="y")