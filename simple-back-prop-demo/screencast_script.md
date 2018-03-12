# Building the simplest neural network

Start with a diagram of the network. Explain that there is some input and some output and some function to forward propagate. Explain how y is calculated from mx

Introduce X and our training output Y and explain that our target is y = 2x

Talk about starting with a randomly initialized network e.g. *m = 0.66*

Talk about how the neural network is able to be trained

>We need a way to tell our network "how wrong it is"

Introduce the error function (with the formula)

(1/2N)\Sigma\to\i(\hat\y\subscript\i - y\subscript\i)\superscript\2)

>The error function is a way of measuring the wrongness of a solution

Explain how we use the error function only as a diagnostic tool

Talk about gradient descent ideally using a graph as an example

Introduce the calculus and explain that the derivitive is a way of seeing the gradient and the direction of the gradient

\delta\E/\delta\m = (1/N)\Sigma\to\i(\hat\y\subscript\i - y\subscript\i)x\subscript\i

Talk about how this function represents how y changes when we change m

Mention the step parameter alpha

\alpha = 0.1

Introduce m'

m' = m - \alpha \delta\E/\delta\m

Begin stepping through first calculation

for my own reference
Iterations
|iteration number|m|y1|y2|y3|E|dE/dm|m'|
|1|0.66|0.66|1.32|1.98|4.19|-6.26|1.29|
|2|1.29|1.29|2.56|3.87|1.18|-3.31|1.62|
|3|1.62|1.62|3.24|4.86|0.34|-1.77|1.80|
