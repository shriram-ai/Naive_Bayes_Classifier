import pandas as pd  
from random import randrange, randint
from math import exp, pi, sqrt

def separation_by_class(iris_data):
    iris_dict = iris_data.to_dict(orient = 'index') 
    separated_by_class = {}
    for i in range(0,len(iris_data)) :
        variety = iris_dict[i].get('variety')
        iris_list = iris_dict[i]
        if variety not in separated_by_class :
            separated_by_class[variety] = list()

        iris_values = []        
        for keys in iris_list :
            iris_values.append(iris_list[keys])        

        del iris_values[-1]
        separated_by_class[variety].append(iris_values)
    return separated_by_class
        
def summary_of_data_by_class(iris_data):
    separated = separation_by_class(iris_data)
    summaries = {}
    for variety in iris_data['variety'].unique():
        summaries[variety] = []
        iris_class_data = pd.DataFrame(data= separated[variety])
        describe_dict = (iris_class_data.describe()).to_dict(orient = 'index')
        for key in describe_dict:
            describe_list = []
            for key2 in describe_dict[key]:
                if key in ['mean', 'count', 'std']:
                    describe_list.append(describe_dict[key].get(key2))
            if len(describe_list) != 0 :
                summaries[variety].append(describe_list)
           
    return summaries

# Gaussian Probability distribution
def calc_probability(x, mean, std):
    exponent = exp(-((x-mean)**2 / (2 * std**2)))  
    return (1 / (sqrt(2 * pi) * std)) * exponent

def probability_by_class(iris_summary, row):
    total_count = 0
    mean_dict = []
    std_dict = []
    prob_class = {}
    class_count = {}
    prob_data_given_class = {}
    
    
    for variety in iris_summary : 
        total_count += (iris_summary.get(variety))[0][0]
        class_count[variety] = (iris_summary.get(variety))[0][0]
     
    for variety in iris_summary : 
        prob_class[variety] = class_count[variety] / total_count
        mean_dict = (iris_summary.get(variety))[1]
        std_dict = (iris_summary.get(variety))[2]
        
        prob_each_row = 1
        for (item, mean, std) in zip(row, mean_dict, std_dict):
            prob_each_row *= calc_probability(item, mean, std)
        prob_data_given_class[variety] = prob_each_row*prob_class[variety]    

    return prob_data_given_class

def train_test_split(iris_dict, fraction_of_test):
    dataset = []
    for variety in iris_dict :
        for row in iris_dict[variety]:
            row.append(variety)
            dataset.append(row)

    count = len(dataset)
    if fraction_of_test < 1:
        test_count = int(fraction_of_test * count)
        test = []
        train = []
        output = []
        for i in range(test_count):
            index = randrange(len(dataset))
            test.append(dataset.pop(index))
            output.append(test[i][-1])
            del test[i][-1]
            train = dataset
        train_data = pd.DataFrame(data = train, columns = ['sepal.length', 
                                                          'sepal.width', 
                                                          'petal.length',
                                                          'petal.width',
                                                          'variety'])
    return (train_data, test, output)   
   

def predict(summaries, row):
    probabilities = probability_by_class(summaries, row)
    best_prob = 0
    best_class = None
    for variety in probabilities:
        probability = probabilities[variety]
        if probability > best_prob or best_class is None :
            best_class = variety 
            best_prob = probability
    return best_class
    
def naive_bayes(train_set, test_set):
    summaries = summary_of_data_by_class(train_set)
    predictions = []
    for row in test_set:
        output = predict(summaries, row)
        predictions.append(output)
        
    return predictions


def accuracy_metrics(actual, predicted):
    correct = 0 ;
    for (predictions, actual_value) in zip(predicted,actual) :
        if predictions == actual_value:
            correct += 1
        
    return correct/len(actual)
    
iris_data = pd.read_csv('iris.csv')
iris_dict = separation_by_class(iris_data)
# iris_summary = summary_of_data_by_class(iris_data)
train_set, test_set, actual = train_test_split(iris_dict, 0.2)
predictions = naive_bayes(train_set,test_set)
accuracy = accuracy_metrics(actual, predictions)
print(accuracy)



