import pickle
import numpy as np
from numpy import cov
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy.stats as stats
from scipy.stats import multivariate_normal
import cv2


# Load the CIFAR-10 data
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Load test data using the folder path
datadict = unpickle("C:/Users/augus/OneDrive/Documents/IPRML/cifar-10-batches-py/test_batch")

X_test = datadict["data"]
Y_test = datadict["labels"]

# Load the labels
labeldict = unpickle("C:/Users/augus/OneDrive/Documents/IPRML/cifar-10-batches-py/batches.meta")
label_names = labeldict["label_names"]

# Load training data
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32")
Y_test = np.array(Y_test)

# Load the data_batch_N files
trainset1=unpickle('C:/Users/augus/OneDrive/Documents/IPRML/cifar-10-batches-py/data_batch_1')
trainset2=unpickle('C:/Users/augus/OneDrive/Documents/IPRML/cifar-10-batches-py/data_batch_2')
trainset3=unpickle('C:/Users/augus/OneDrive/Documents/IPRML/cifar-10-batches-py/data_batch_3')
trainset4=unpickle('C:/Users/augus/OneDrive/Documents/IPRML/cifar-10-batches-py/data_batch_4')
trainset5=unpickle('C:/Users/augus/OneDrive/Documents/IPRML/cifar-10-batches-py/data_batch_5')

X_train1=trainset1["data"]
Y_train1=trainset1["labels"]

X_train2=trainset2["data"]
Y_train2=trainset2["labels"]

X_train3=trainset3["data"]
Y_train3=trainset3["labels"]

X_train4=trainset4["data"]
Y_train4=trainset4["labels"]

X_train5=trainset5["data"]
Y_train5=trainset5["labels"]

X_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_train5))
Y_train =np.concatenate((Y_train1,Y_train2,Y_train3,Y_train4,Y_train5))
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("int32")
Y_train = np.array(Y_train)

# Calculate the classification accuracy
def class_acc(pred,gt):
    classification_count = len(pred)
    classification_err = 0
    for i in range(0, classification_count):
        if pred[i] != gt[i]:
            classification_err += 1
    acc_err = (classification_err / classification_count)*100
    acc = 100 - acc_err
    return acc

accuracies = []
N=[]

# Task 1
# convert the original images X (50000 × 32 × 32 × 3) to Xp (50000 × 3)
def cifar10_color(X):
    Xp = np.empty((X.shape[0], 3))
    for idx, image in enumerate(X):
        Xp[idx] = resize(image, (1, 1)).flatten()
    return Xp


# compute the normal distribution parameters
def cifar_10_naivebayes_learn(Xp,Y):
    df = pd.DataFrame(Xp)
    df[len(df.columns)] = pd.DataFrame(Y)
    mu = np.array(df.groupby(len(df.columns) - 1).mean())
    sigma = np.array(df.groupby(len(df.columns) - 1).std())
    p = np.array(df.groupby(len(df.columns) - 1).size().div(len(df))).reshape(10, 1)
    return (mu, sigma, p)


# the Bayesian optimal class c for the sample x.
def cifar10_classifier_naivebayes(x,mu,sigma,p):
    prob_class_0=stats.norm.pdf(x[0],mu[0][0], sigma[0][0])*stats.norm.pdf(x[1],mu[0][1], sigma[0][1])*stats.norm.pdf(x[2],mu[0][2], sigma[0][2])*p[0]
    prob_class_1=stats.norm.pdf(x[0],mu[1][0], sigma[1][0])*stats.norm.pdf(x[1],mu[1][1], sigma[1][1])*stats.norm.pdf(x[2],mu[1][2], sigma[1][2])*p[1]
    prob_class_2=stats.norm.pdf(x[0],mu[2][0], sigma[2][0])*stats.norm.pdf(x[1],mu[2][1], sigma[2][1])*stats.norm.pdf(x[2],mu[2][2], sigma[2][2])*p[2]
    prob_class_3=stats.norm.pdf(x[0],mu[3][0], sigma[3][0])*stats.norm.pdf(x[1],mu[3][1], sigma[3][1])*stats.norm.pdf(x[2],mu[3][2], sigma[3][2])*p[3]
    prob_class_4=stats.norm.pdf(x[0],mu[4][0], sigma[4][0])*stats.norm.pdf(x[1],mu[4][1], sigma[4][1])*stats.norm.pdf(x[2],mu[4][2], sigma[4][2])*p[4]
    prob_class_5=stats.norm.pdf(x[0],mu[5][0], sigma[5][0])*stats.norm.pdf(x[1],mu[5][1], sigma[5][1])*stats.norm.pdf(x[2],mu[5][2], sigma[5][2])*p[5]
    prob_class_6=stats.norm.pdf(x[0],mu[6][0], sigma[6][0])*stats.norm.pdf(x[1],mu[6][1], sigma[6][1])*stats.norm.pdf(x[2],mu[6][2], sigma[6][2])*p[6]
    prob_class_7=stats.norm.pdf(x[0],mu[7][0], sigma[7][0])*stats.norm.pdf(x[1],mu[7][1], sigma[7][1])*stats.norm.pdf(x[2],mu[7][2], sigma[7][2])*p[7]
    prob_class_8=stats.norm.pdf(x[0],mu[8][0], sigma[8][0])*stats.norm.pdf(x[1],mu[8][1], sigma[8][1])*stats.norm.pdf(x[2],mu[8][2], sigma[8][2])*p[8]
    prob_class_9=stats.norm.pdf(x[0],mu[9][0], sigma[9][0])*stats.norm.pdf(x[1],mu[9][1], sigma[9][1])*stats.norm.pdf(x[2],mu[9][2], sigma[9][2])*p[9]
    class_predicted=np.argmax([prob_class_0,prob_class_1,prob_class_2,prob_class_3,prob_class_4,prob_class_5,prob_class_6,prob_class_7,prob_class_8,prob_class_9])
    return class_predicted


# 1. CIFAR-10 – Bayesian classifier (good)
Xf_train=cifar10_color(X_train)
[mu,sigma,p]=cifar_10_naivebayes_learn(Xf_train,Y_train)
Xf_test=cifar10_color(X_test)
y_pred_a= np.zeros(10000)
for i in range(0,10000):
    y_pred_a[i] = cifar10_classifier_naivebayes(Xf_test[i],mu,sigma,p)
acc = class_acc(y_pred_a,Y_test)
accuracies.append(acc)
N.append((1,1))
print("The accuracy for naive Bayes classifier: %.3f" %(acc) +"%")


# Task 2
# compute the multivariate normal distribution parameters
def cifar_10_bayes_learn(Xf, Y):
    df = pd.DataFrame(Xf)
    df[len(df.columns)] = pd.DataFrame(Y)
    mu = np.array(df.groupby(len(df.columns) - 1).mean())
    sigma = np.array(df.groupby(len(df.columns) - 1).cov()).reshape(10, Xf.shape[1], Xf.shape[1])
    prior = np.array(df.groupby(len(df.columns) - 1).size().div(len(df))).reshape(10, 1)
    return (mu, sigma, prior)

# compute probabilities
def cifar10_classifier_bayes(x, mu, cov, p):
    prob = np.empty((10,1))
    for j in range(0, 10):
        p_total = stats.multivariate_normal.pdf(x, mu[j], cov[j])*p[j]
        denom = 0
        for i in range(0,10):
            denom+=stats.multivariate_normal.pdf(x, mu[i], cov[i])*p[i]

        prob[j] = p_total/denom
    return np.argmax(prob)


# 2. CIFAR-10 – Bayesian classifier (better)
Xf2 = cifar10_color(X_train)
[mu, cov, p] = cifar_10_bayes_learn(Xf2, Y_train)
test_images2 = cifar10_color(X_test)
pred_cls2 = np.zeros(10000)
for x in range(0, 10000):
    pred_cls2[x] = cifar10_classifier_bayes(test_images2[x], mu, cov, p)
acc2 = class_acc(pred_cls2, Y_test)
accuracies.append(acc2)
N.append((1,1))
print(f"The accuracy for better Bayesian classifier is: %.3f" %(acc2) +"%")

def cifar10_2x2_color(X):
    Xp = np.array([])
    for i in range(X.shape[0]):
        img = X[i]
        img_2x2 = resize(img, (2, 2))
        r_vals = img_2x2[:, :, 0].reshape(2 * 2)
        g_vals = img_2x2[:, :, 1].reshape(2 * 2)
        b_vals = img_2x2[:, :, 2].reshape(2 * 2)
        vector = np.concatenate((np.transpose(r_vals), np.transpose(g_vals), np.transpose(b_vals)))
        Xp = np.append(Xp, vector)
    return Xp


def cifar10_classifier_bayes3(x, ms, cov, p): #x: 1x3, cov:10x3x3
    prob = np.zeros([10000, 10])
    for i in range(0, 10):
        p_total = stats.multivariate_normal.logpdf(x, ms[i, :], cov[i, :, :]) #cov size 1x3x3
        prob[:, i] = p_total*p
    cls = np.argmax(prob, axis=1)
    return cls

X_2x2 = cifar10_2x2_color(X_train)
X_2x2 =X_2x2.reshape(50000,12)
[mu,sigma,p] = cifar_10_bayes_learn(X_2x2,Y_train)
Xt_2x2 = cifar10_2x2_color(X_test)
Xt_2x2 = Xt_2x2.reshape(10000,12)
preds=[]
for i in range(0,len(Xt_2x2)):
    y_bayesian_super=cifar10_classifier_bayes(Xt_2x2[i],mu,cov,p)
    preds.append(y_bayesian_super)

N.append((2,2))
accuracies.append(class_acc(preds,Y_test))
print("Bayes Accuracy 2x2: ",class_acc(preds,Y_test))

#Graph
plt.plot(N,accuracies, 'r-')
plt.xlabel('N')
plt.ylabel('Accuracy')
plt.show()