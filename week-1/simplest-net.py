import math

input = [1, 2, 3]

expected_output = [2, 4, 6]
actual_output = []

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

def error_derivative(expected_output, actual_output, input):
    sum = 0
    for index, item in enumerate(expected_output):
        sum += (actual_output[index] - item) * input[index]       
    derivative = sum/ len(expected_output)
    return derivative

def forward(input, weight):
    output = []
    for item in input:
        output.append(item * weight)
    return output

def backward(weight, error_derivative):
    return weight - (step * error_derivative)

print(error(expected_output, [1.22, 2.44, 3.66]) == 1.4196)
print(error_derivative(expected_output, [1.22, 2.44, 3.66], input) == -3.64)
print(forward(input, 1.22) == [1.22, 2.44, 3.66])
print(backward(weight, -5.59) == 1.219)

err = 1
while(err!= 0):
    output = forward(input, weight)
    print(output)
    err = error(expected_output, output)
    print("Error = ", err)
    error_deriv = error_derivative(expected_output, output, input)
    weight = backward(weight, error_deriv)
