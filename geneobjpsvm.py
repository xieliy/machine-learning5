'''
Welcome to use our differentially private machine learning algorithm!
Please see 'User Guide' file in our github account for all the details!
Thank you and we welcome any kind advices and opinions!
'''
##implement Private Convex Empirical Risk Minimization and High-dimensional Regression paper(algorithm1)
##objective perturbation with huber loss

from numpy import *
from pylab import norm
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from random import SystemRandom

rng = SystemRandom()#generate random numbers from sources provided by the operating system.
seed = rng.seed()#initialize the basic random number generator use current system time
Epsilon_global = 0
Lambda_global = 0
Huber_global = 0
delta_global = 0

def load_private_dataset(txt_file_url):
    '''
    load private data from .txt file
    '''
    data_set = []#store data set
    txt_file_object = open(txt_file_url, 'r')
    for line in txt_file_object:
        data_set.append(line)
    txt_file_object.close()
    return data_set

def stringToInt(lists):
    '''
    Transform components in every data point from string to int
    '''
    lists2 = []
    for i in range(len(lists)):
        List = []#store every row
        for j in range(len(lists[0])):
            List.append(int(lists[i][j]))
        lists2.append(List)
    return lists2

def separation(strings):
    '''
    Remove character '\n' except last line
    Transform data points from string type to list type
    Transform components in every data point from string to int
    '''
    lists = []
    for i in range(len(strings)-1):
        lists.append(strings[i][:-1].split(' '))
    lists.append(strings[len(strings)-1].split(' '))
    lists = stringToInt(lists)
    return lists#every row is a data point, the last element is the corresponding label

def data_label_split(datalabels):
    '''
    split data part and label part.
    '''
    l = len(datalabels[0])
    data = datalabels[:,0:(l-1)]
    labels = datalabels[:,l-1]
    return (data,labels)

def parameters_set(Epsilon_set, Lambda_set, Huber_set, delta_set):
    '''
    set parameters used in this paper
    '''
    global Epsilon_global, Lambda_global, Huber_global, delta_global
    Epsilon_global = Epsilon_set#privacy parameter
    Lambda_global = Lambda_set#regularization parameter
    Huber_global = Huber_set#huber constant
    delta_global = delta_set#privacy parameter
    return

def parameters():
    '''
    set parameters used in this paper
    '''
    Epsilon = Epsilon_global#privacy parameter
    Lambda = Lambda_global#regularization parameter
    Huber = Huber_global#huber constant
    delta = delta_global#privacy parameter
    return (Epsilon, Lambda, Huber, delta)

def noisevector2(scale,length):
    '''
    Generate a noise vector according to paper 'Private Convex Empirical Risk Minimization and High-dimensional Regression'(algorithm1, line5)
    '''
    res = random.normal(0, scale, length)
    return res

def huber(z):#chaudhuri2011differentially corollary 21
    '''
    huber loss 
    '''
    if z>1+parameters()[2]:
        hb = 0
    elif fabs(1-z)<=parameters()[2]:
        hb = (1+parameters()[0]-z)**2/(4*parameters()[2])
    else:
        hb = 1-z
    return hb

def svm_objective_train(data, labels):
    '''
    generalized objective peturbation
    huber loss
    '''
    n = len(labels)#number of data points in the data set
    l = len(data[0])#length of a data point
    x0 = zeros(l)#starting point with same length as any data point

    zeta = 1#bound of norm of gradient of loss function of certain data point

    var = (zeta**2)*(8*log(2/parameters()[3])+4*parameters()[0])/(parameters()[0]**2)#variance of noise distribution
    std = sqrt(var)#standard variance of noise distribution
    noise2 = noisevector2(std,l)#Generate noise vector

    Delta = 2*sqrt((l/(2*parameters()[2]))**2+l*(1+1/parameters()[2]))/parameters()[0]# set Delta = 2*Lambda/Epsilon

    def func(x):
        jfd = huber(labels[0]*dot(data[0],x))
        for i in range(1,n):
            jfd = jfd + huber(labels[i]*dot(data[i],x))
        f = (1/n)*jfd + (norm(x)**2)/(2*n) + Delta*(norm(x)**2)/(2*n) + dot(noise2,x)/n
        return f
    
    #constraint, we need a convex domain
    cons = {'type': 'ineq', 'fun': lambda x: 10**100 - norm(x)}
            
    #minimization procedure
    fpriv = minimize(func,x0,method='SLSQP',constraints=cons).x#empirical risk minimization using scipy.optimize minimize function
    return fpriv

def train(data, labels):
    '''
    train objective peturbation and output classifer
    ''' 
    classifer_output = svm_objective_train(data, labels)
    return classifer_output

def change_file(input_txt_file_url):
    '''
    Since we need to output 'output classifier.txt' file in the same directory with
    input file, we need to change the URL string 
    '''
    for i in range(len(input_txt_file_url)):
        if input_txt_file_url[len(input_txt_file_url)-i-1] == '\\':
            input_txt_file_url = input_txt_file_url[:len(input_txt_file_url)-i]
            ourput_txt_file_url = input_txt_file_url + 'output classifier.txt'
            break
    return ourput_txt_file_url

def write_txt(output,ourput_txt_file_url):
    '''
    write the output into 'output classifier.txt' file
    '''
    with open(ourput_txt_file_url, 'w') as f_out:
        for i in output:
            if i == len(output)-1:
                f_out.write(str(i))
            f_out.write(str(i) + ' ')
    return

def run(input_txt_file_url):
    '''
    The only function that others can call
    '''
    print(__doc__)
    
    print('please set the parameters before continue.')
    Epsilon_set = input('please enter privacy parameter epsilon: ')
    print ('Epsilon is: ' + Epsilon_set)
    Lambda_set = input('please enter regularization parameter lambda: ')
    print ('lambda is: ' + Lambda_set)
    Huber_set = input('please enter huber constant: ')
    print ('huber constant is: ' + Huber_set)
    delta_set = input('please enter delta constant: ')
    print ('delta constant is: ' + delta_set)
    parameters_set(float(Epsilon_set),float(Lambda_set),float(Huber_set), float(delta_set))#input are string, convert to float

    data_set = load_private_dataset(input_txt_file_url)
    print('training data load successfully!')
    
    data_set_n = separation(data_set)
    data_set_n = array(data_set_n)
    data,labels = data_label_split(data_set_n)
    print('labels are: ' + str(labels))
    
    classifer_output = train(data, labels)
    print('compute classifer successfully!')
    print('classifer_output is: ' + str(classifer_output))
    
    ourput_txt_file_url = change_file(input_txt_file_url)
    write_txt(classifer_output,ourput_txt_file_url)
    print('write classifer successfully!')
    
    
