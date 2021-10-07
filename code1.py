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
    #print(temp_input,((np.exp(e1)/(np.exp(e1)+np.exp(e2)+np.exp(e3)))*temp_input))
    for i in training:
        temp_input = 0.001 * np.array(q5.data_lst[i])
        #print(temp_input)
        e1 = np.dot(alpha1,np.transpose(temp_input))
        e2 = np.dot(alpha2,np.transpose(temp_input))
        e3 = np.dot(alpha3,np.transpose(temp_input))
        #rint(e1,e2,e3)
        #print(alpha1,temp_input,e1)
        term2 = term2 - (temp_input/(1+np.exp(e2-e1)+np.exp(e3-e1)))
    #print("grad1: ",term1 + term2)
    #print("grad1-term2: ",term2)

    #print(term1 + term2)
    return (term1 + term2)

def comp_grad2(alpha1,alpha2,alpha3,training): # computes the gradient wrt alpha2
    term1 = np.array([0,0,0,0,0])
    for i in training:
        #print(q5.desired[i])
        if q5.desired[i][1] == 1:
            temp_input = 0.001 * np.array(q5.data_lst[i])
            term1 = term1 + temp_input
    term2 = np.array([0,0,0,0,0])
    #print(temp_input,((np.exp(e2)/(np.exp(e1)+np.exp(e2)+np.exp(e3)))*temp_input))
    for i in training:
        temp_input = 0.001 * np.array(q5.data_lst[i])
        e1 = np.dot(alpha1,np.transpose(temp_input))
        e2 = np.dot(alpha2,np.transpose(temp_input))
        e3 = np.dot(alpha3,np.transpose(temp_input))
        #print(np.exp(e2))
        term2 = term2 - (temp_input/(np.exp(e1-e2)+1+np.exp(e3-e2)))
    #print(term1+term2)
    return (term1 + term2)

def comp_grad3(alpha1,alpha2,alpha3,training):# computes the gradient wrt alpha3
    term1 = np.array([0,0,0,0,0])
    for i in training:
        if q5.desired[i][2] == 1:
            temp_input = 0.001 * np.array(q5.data_lst[i])
            term1 = term1 + temp_input
    term2 = np.array([0,0,0,0,0])
    for i in training:
        #term2 = np.array([0,0,0,0,0])
        temp_input = 0.001 * np.array(q5.data_lst[i])
        e1 = np.dot(alpha1,np.transpose(temp_input))
        e2 = np.dot(alpha2,np.transpose(temp_input))
        e3 = np.dot(alpha3,np.transpose(temp_input))
        term2 = term2 - (temp_input/(np.exp(e1-e3)+np.exp(e2-e3)+1))
    #print(term1+term2)
    return (term1 + term2)

full_order = np.random.permutation(150)
training = []
testing = []
for f in range(100):# using 100 data points for training
    training.append(full_order[f])
for g in range(100,150):# using 50 data points for testing
    testing.append(full_order[g])
#initialising the weights
#alpha1 = np.random.rand(1,5)
alpha1 = np.array([1,1,1,1,1])
#alpha2 = np.random.rand(1,5)
alpha2 = np.array([1,1,1,1,1])
#alpha3 = np.random.rand(1,5)
alpha3 = np.array([1,1,1,1,1])
learn = 25
#Now we are doing the gradient descent
for i in range(500):

    #np.random.shuffle(training)
    temp_alpha1 = np.copy(alpha1 + learn*(comp_grad1(alpha1,alpha2,alpha3,training)))
    temp_alpha2 = np.copy(alpha2 + learn*(comp_grad2(alpha1,alpha2,alpha3,training)))
    temp_alpha3 = np.copy(alpha3 + learn*(comp_grad3(alpha1,alpha2,alpha3,training)))
    #print(comp_grad1(alpha1,alpha2,alpha3,training))
    #print(comp_grad1(alpha1,alpha2,alpha3,training),comp_grad2(alpha1,alpha2,alpha3,training),comp_grad3(alpha1,alpha2,alpha3,training))
    alpha1 = np.copy(temp_alpha1)
    alpha2 = np.copy(temp_alpha2)
    alpha3 = np.copy(temp_alpha3)

    #print("grad2 : ",comp_grad2(alpha1,alpha2,alpha3,training))
    #print("grad3: ",comp_grad3(alpha1,alpha2,alpha3,training))
    #print(alpha1,alpha2,alpha3)
#Now we test the final accuracy
error = 0
for i in testing:
    des = np.array(q5.desired[i])
    temp_inp = 0.001 * np.array(q5.data_lst[i])
    output = np.array([p1(temp_inp,alpha1,alpha2,alpha3),p2(temp_inp,alpha1,alpha2,alpha3),p3(temp_inp,alpha1,alpha2,alpha3)])

    if np.argmax(des) != np.argmax(output):
        #print(np.argmax(output))
        error = error + 1
print(100*(1 - error/50))
