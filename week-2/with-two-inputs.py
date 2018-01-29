import math
import numpy as np

def error(expected_output, actual_output):
    sum = 0
    zipped = zip(expected_output, actual_output)
    for (y, yhat) in zipped:
        difference = yhat - y
        sum += math.pow(difference, 2)        
    error = sum/ (2.0 * len(expected_output))
    return error

def error_derivative_with_respect_to_m(expected_output, actual_output, input):
    sum = 0
    for index, item in enumerate(expected_output):
        sum += (actual_output[index] - item) * input[index]       
    derivative = sum/ len(expected_output)
    return derivative

def error_derivative_with_respect_to_c(expected_output, actual_output):
    sum = 0
    for index, item in enumerate(expected_output):
        sum += (actual_output[index] - item) 
    derivative = sum/ len(expected_output)
    return derivative
    
def forward(inputs, weight, bias):
    return np.add(np.dot(inputs, weight), bias)

def back_propagate(parameter, derivative_with_respect_to_parameter):
    return parameter - (step * derivative_with_respect_to_parameter)

def main():
    input = [[1, 2], [3, 4], [5, 6]]
    expected_output = [3.5, 5.5, 7.5] 
    actual_output = []
    bias = 0.13456
    weight = 0.66
    step = 0.1
    err = 1
    while(abs(err) > 0.000000001):
        output = forward(input, weight, bias)
        print(output)
        err = error(expected_output, output)
        print("Error = ", err)
        error_deriv_wrt_m = error_derivative_with_respect_to_m(expected_output, output, input)
        weight = back_propagate(weight, error_deriv_wrt_m)
        error_deriv_wrt_c = error_derivative_with_respect_to_c(expected_output, output)
        bias = back_propagate(bias, error_deriv_wrt_c)


def test():
    print("running tests")
    forward_test()
    error_test()

def forward_test():
    inputs = [[1, 2], [3, 4]]
    weights = [5, 6]
    bias = 7
    expected_result = [24, 46]
    result = forward(inputs, weights, bias)
    print("forward test passes: ", np.array_equal(result, expected_result))

def error_test():
    computed_output = [15, 22]
    training_output = [8, 11]
    expected_error = 42.5
    result = error(computed_output, training_output)
    print("Error test passes: ", result == expected_error)

test()
