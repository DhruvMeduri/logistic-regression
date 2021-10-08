import q5
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import matplotlib.pyplot as plt
# this code implements logistic regression using the softmax function

def p1(temp_input,alpha1,alpha2,alpha3):# this function outputs P(y = 1)
    e1 = np.dot(alpha1,np.transpose(temp_input))
    e2 = np.dot(alpha2,np.transpose(temp_input))
    e3 = np.dot(alpha3,np.transpose(temp_input))
    return (1/(1+np.exp(e2-e1)+np.exp(e3-1)))

def p2(temp_input,alpha1,alpha2,alpha3):# this function outputs P(y = 2)
    e1 = np.dot(alpha1,np.transpose(temp_input))
    e2 = np.dot(alpha2,np.transpose(temp_input))
    e3 = np.dot(alpha3,np.transpose(temp_input))
    return (1/(np.exp(e1-e2)+1+np.exp(e3-e2)))

def p3(temp_input,alpha1,alpha2,alpha3):# this function outputs P(y = 3)
    e1 = np.dot(alpha1,np.transpose(temp_input))
    e2 = np.dot(alpha2,np.transpose(temp_input))
    e3 = np.dot(alpha3,np.transpose(temp_input))
    return (1/(np.exp(e1-e3)+np.exp(e2-e3)+1))

def comp_grad1(alpha1,alpha2,alpha3,training):# computes the gradient wrt alpha1
    term1 = np.array([0,0,0,0,0])
    for i in training:
        if q5.desired[i][0] == 1:
            temp_input = 0.001 * np.array(q5.data_lst[i])
            term1 = term1 + temp_input
    term2 = np.array([0,0,0,0,0])
    for i in training:
        temp_input = 0.001 * np.array(q5.data_lst[i])
        e1 = np.dot(alpha1,np.transpose(temp_input))
        e2 = np.dot(alpha2,np.transpose(temp_input))
        e3 = np.dot(alpha3,np.transpose(temp_input))
        term2 = term2 - (temp_input/(1+np.exp(e2-e1)+np.exp(e3-e1)))

    return (term1 + term2)

def comp_grad2(alpha1,alpha2,alpha3,training): # computes the gradient wrt alpha2
    term1 = np.array([0,0,0,0,0])
    for i in training:
        if q5.desired[i][1] == 1:
            temp_input = 0.001 * np.array(q5.data_lst[i])
            term1 = term1 + temp_input
    term2 = np.array([0,0,0,0,0])
    for i in training:
        temp_input = 0.001 * np.array(q5.data_lst[i])
        e1 = np.dot(alpha1,np.transpose(temp_input))
        e2 = np.dot(alpha2,np.transpose(temp_input))
        e3 = np.dot(alpha3,np.transpose(temp_input))
        term2 = term2 - (temp_input/(np.exp(e1-e2)+1+np.exp(e3-e2)))
    return (term1 + term2)

def comp_grad3(alpha1,alpha2,alpha3,training):# computes the gradient wrt alpha3
    term1 = np.array([0,0,0,0,0])
    for i in training:
        if q5.desired[i][2] == 1:
            temp_input = 0.001 * np.array(q5.data_lst[i])
            term1 = term1 + temp_input
    term2 = np.array([0,0,0,0,0])
    for i in training:
        temp_input = 0.001 * np.array(q5.data_lst[i])
        e1 = np.dot(alpha1,np.transpose(temp_input))
        e2 = np.dot(alpha2,np.transpose(temp_input))
        e3 = np.dot(alpha3,np.transpose(temp_input))
        term2 = term2 - (temp_input/(np.exp(e1-e3)+np.exp(e2-e3)+1))
    return (term1 + term2)

# this code runs the logistic regression and plots the error trajectory
err1 = []
full_order = np.random.permutation(150)
training = []
testing = []
for f in range(100):# using 100 data points for training
    training.append(full_order[f])
for g in range(100,150):# using 50 data points for testing
    testing.append(full_order[g])
#initialising the weights
alpha1 = np.random.rand(1,5)
#alpha1 = np.array([1,1,1,1,1])
alpha2 = np.random.rand(1,5)
#alpha2 = np.array([1,1,1,1,1])
alpha3 = np.random.rand(1,5)
#alpha3 = np.array([1,1,1,1,1])
learn = 25
#Now we are doing the gradient descent and upadating the weights
for i in range(500):

    np.random.shuffle(training)
    temp_alpha1 = np.copy(alpha1 + learn*(comp_grad1(alpha1,alpha2,alpha3,training)))
    temp_alpha2 = np.copy(alpha2 + learn*(comp_grad2(alpha1,alpha2,alpha3,training)))
    temp_alpha3 = np.copy(alpha3 + learn*(comp_grad3(alpha1,alpha2,alpha3,training)))
    alpha1 = np.copy(temp_alpha1)
    alpha2 = np.copy(temp_alpha2)
    alpha3 = np.copy(temp_alpha3)
# now we compute the error after every 25 gradient descent updates of the weights
    if i%25 == 0:
        error = 0
        for i in testing:
            des = np.array(q5.desired[i])
            temp_inp = 0.001 * np.array(q5.data_lst[i])
            output = np.array([p1(temp_inp,alpha1,alpha2,alpha3),p2(temp_inp,alpha1,alpha2,alpha3),p3(temp_inp,alpha1,alpha2,alpha3)])
            if np.argmax(des) != np.argmax(output):
                #print(np.argmax(output))
                error = error + 1
        err1.append(2*error)
print('Alpha1: ',alpha1)
print('Alpha2: ',alpha2)
print('Alpha3: ',alpha3)
# this is for plotting the graph
x_list = []
for i in range(1, 501):
    if i%25 == 0:
       x_list.append(i)
plt.scatter(x_list,err1)
plt.plot(x_list,err1)
plt.xlabel("Iterations")
plt.ylabel("%error")
plt.title("Error Trajectory(100 training data points)")
plt.show()

'''
# the below code is for comparing the error trajectories using 70% and 80% training sets
err1 = []
full_order = np.random.permutation(150)
training = []
testing = []
for f in range(105):# using 70% data points for training
    training.append(full_order[f])
for g in range(105,150):# using 30% data points for testing
    testing.append(full_order[g])
#initialising the weights
alpha1 = np.random.rand(1,5)
#alpha1 = np.array([1,1,1,1,1])
alpha2 = np.random.rand(1,5)
#alpha2 = np.array([1,1,1,1,1])
alpha3 = np.random.rand(1,5)
#alpha3 = np.array([1,1,1,1,1])
learn = 25
#Now we are doing the gradient descent
for i in range(500):

    np.random.shuffle(training)
    temp_alpha1 = np.copy(alpha1 + learn*(comp_grad1(alpha1,alpha2,alpha3,training)))
    temp_alpha2 = np.copy(alpha2 + learn*(comp_grad2(alpha1,alpha2,alpha3,training)))
    temp_alpha3 = np.copy(alpha3 + learn*(comp_grad3(alpha1,alpha2,alpha3,training)))
    alpha1 = np.copy(temp_alpha1)
    alpha2 = np.copy(temp_alpha2)
    alpha3 = np.copy(temp_alpha3)
# now we compute the error after every 25 updates of the weights using gradient descent
    if i%25 == 0:
        error = 0
        for i in testing:
            des = np.array(q5.desired[i])
            temp_inp = 0.001 * np.array(q5.data_lst[i])
            output = np.array([p1(temp_inp,alpha1,alpha2,alpha3),p2(temp_inp,alpha1,alpha2,alpha3),p3(temp_inp,alpha1,alpha2,alpha3)])
            if np.argmax(des) != np.argmax(output):
                #print(np.argmax(output))
                error = error + 1
        err1.append(2.222*error)


err2 = []
full_order = np.random.permutation(150)
training = []
testing = []
for f in range(120):# using 80% data points for training
    training.append(full_order[f])
for g in range(120,150):# using 20% data points for testing
    testing.append(full_order[g])
#initialising the weights
alpha1 = np.random.rand(1,5)
#alpha1 = np.array([1,1,1,1,1])
alpha2 = np.random.rand(1,5)
#alpha2 = np.array([1,1,1,1,1])
alpha3 = np.random.rand(1,5)
#alpha3 = np.array([1,1,1,1,1])
learn = 25
#Now we are doing the gradient descent
for i in range(500):

    np.random.shuffle(training)
    temp_alpha1 = np.copy(alpha1 + learn*(comp_grad1(alpha1,alpha2,alpha3,training)))
    temp_alpha2 = np.copy(alpha2 + learn*(comp_grad2(alpha1,alpha2,alpha3,training)))
    temp_alpha3 = np.copy(alpha3 + learn*(comp_grad3(alpha1,alpha2,alpha3,training)))
    alpha1 = np.copy(temp_alpha1)
    alpha2 = np.copy(temp_alpha2)
    alpha3 = np.copy(temp_alpha3)
# now we compute the error after every 25 updates of the weights using gradient descent
    if i%25 == 0:
        error = 0
        for i in testing:
            des = np.array(q5.desired[i])
            temp_inp = 0.001 * np.array(q5.data_lst[i])
            output = np.array([p1(temp_inp,alpha1,alpha2,alpha3),p2(temp_inp,alpha1,alpha2,alpha3),p3(temp_inp,alpha1,alpha2,alpha3)])
            if np.argmax(des) != np.argmax(output):
                #print(np.argmax(output))
                error = error + 1
        err2.append(3.333*error)

# this is for plotting the graph
x_list = []
for i in range(1, 501):
    if i%25 == 0:
       x_list.append(i)
plt.scatter(x_list,err1,color='blue')
plt.plot(x_list,err1,color='blue',label='70% training data')
plt.scatter(x_list,err2,color='red')
plt.plot(x_list,err2,color='red',label = '80% training data')
plt.xlabel("Iterations")
plt.ylabel("%error")
plt.title("Error Trajectory(70% vs 80% comparison)")
plt.legend()
plt.show()
'''
