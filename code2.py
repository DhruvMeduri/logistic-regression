import numpy as np
import q5
import matplotlib.pyplot as plt

full_order = np.random.permutation(150)
training = []
testing = []
for f in range(100):# using 100 data points for training
    training.append(full_order[f])
for g in range(100,150):# using 50 data points for training
    testing.append(full_order[g])
# Train weight vector
lr = 0.5                                                   # learning rate
w1 = np.random.rand(1,5)                                   # weight vector for P(y=1) (1x5)
w2 = np.random.rand(1,5)                                   # weight vector for P(y=2) (1x5)
w3 = np.random.rand(1,5)                                   # weight vector for P(y=3) (1x5)
for j in range(100):

    #np.random.shuffle(training_set)
    sum1 = np.zeros((1,5))                                    # gradient vector for w1 (1x5)
    sum2 = np.zeros((1,5))                                    # gradient vector for w2 (1x5)
    sum3 = np.zeros((1,5))                                    # gradient vector for w3 (1x5)
    for i in training:
        temp_1 = np.dot(w1,np.transpose(q5.data_lst[i]))       # w1T*xi
        temp1 = np.exp(temp_1)                                    # exp(w1T*xi)
        temp_2 = np.dot(w2,np.transpose(q5.data_lst[i]))       # w2T*xi
        temp2 = np.exp(temp_2)                                    # exp(w2T*xi)
        temp_3 = np.dot(w3,np.transpose(q5.data_lst[i]))       # w3T*xi
        temp3 = np.exp(temp_3)                                    # exp(w3T*xi)
        temp4 = temp1 + temp2 + temp3
        #print(temp1/temp3,temp2/temp3)
        if q5.desired[i][0] == 1:
            sum1 = sum1 + np.array(q5.data_lst[i])
        if q5.desired[i][1] == 1:
            sum2 = sum2 + np.array(q5.data_lst[i])
        if q5.desired[i][2] == 1:
            sum3 = sum3 + np.array(q5.data_lst[i])
        sum1 = sum1 - ((np.array(q5.data_lst[i]))/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1)))
        sum2 = sum2 - ((np.array(q5.data_lst[i]))/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2)))
        sum3 = sum3 - ((np.array(q5.data_lst[i]))/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3)))




    w1 = w1 - lr*sum1
    w2 = w2 - lr*sum2
    w3 = w3 - lr*sum3
print(w1,w2,w3)

# Test weight vector
correct = 0
for i in testing:
    temp_1 = np.dot(w1,np.transpose(q5.data_lst[i]))       # w1T*xi
    temp1 = np.exp(temp_1)                                    # exp(w1T*xi)
    temp_2 = np.dot(w2,np.transpose(q5.data_lst[i]))       # w2T*xi
    temp2 = np.exp(temp_2)                                    # exp(w2T*xi)
    temp_3 = np.dot(w3,np.transpose(q5.data_lst[i]))       # w3T*xi
    temp3 = np.exp(temp_3)                                    # exp(w3T*xi)
    temp4 = temp1 + temp2 + temp3
    res_1 = 1/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1))
    res_2 = 1/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2))
    res_3 = 1/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3))
    res = np.array([res_1,res_2,res_3])
    #print(res_1,res_2,res_3)
    des = np.array(q5.desired[i])
    if np.argmax(res) == np.argmax(des):
        correct = correct + 1

print("Accuracy: ")
print(correct)
