import math

input = [1, 2, 3]

expected_output = [3.5, 5.5, 7.5] 
actual_output = []

bias = 0.13456
weight = 0.66
step = 0.1

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
    
def forward(input, weight, bias):
    output = []
    for item in input:
        output.append(item * weight + bias)
    return output

def back_propagate(parameter, derivative_with_respect_to_parameter):
    return parameter - (step * derivative_with_respect_to_parameter)

print(error(expected_output, [1.22, 2.44, 3.66]) == 1.4196)
print(error_derivative_with_respect_to_m(expected_output, [1.22, 2.44, 3.66], input) == -3.64)
print(error_derivative_with_respect_to_c(expected_output, [1.22, 2.44, 3.66]))

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
