'''
test file of oplr.py
'''
from geneobjpsvm import run
from geneobjpsvmer import error_run
from numpy import array

#you need to change input_txt_file_url to your absolute directory below
training_url = r'C:\Users\xieliy\Desktop\master thesis\code\Github\machine-learning5\training data.txt'
run(training_url)

#test error rate using testing data set and computed classifier
testing_url = r'C:\Users\xieliy\Desktop\master thesis\code\Github\machine-learning5\testing data.txt'
classifier_url = r'C:\Users\xieliy\Desktop\master thesis\code\Github\machine-learning5\output classifier.txt'
error_run(testing_url, classifier_url)
