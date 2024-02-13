# Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function

# Understanding-and-Implementing-the-Activation-Function

**Objective:**

**1.**	To comprehend the conceptual and mathematics underpinnings of the Activation Function.

**2.**	To execute the Activation Function in a programming language (such as Python).

**3.**	The objective is to examine the attributes and consequences of using the Activation Function inside neural networks.

**Tasks:**

**1.**	**Theoretical Understanding:**

o	**Explain the Activation Function, including its equation and graph.**

Ans: Activation functions play a crucial role in artificial neural networks by introducing non-linearity, which enables the network to learn complex mappings between input and output spaces. An activation function is a crucial component of a neural network, responsible for introducing non-linearity into the model. It operates on the weighted sum of inputs and biases of a neuron, determining whether it should be activated or not. The output of the activation function decides the input for the next layer in the neural network.

**What is an activation function and why use them?** 

The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The purpose of the activation function is to introduce non-linearity into the output of a neuron. 

**Explanation:** 

We know, the neural network has neurons that work in correspondence with weight, bias, and their respective activation function. In a neural network, we would update the weights and biases of the neurons on the basis of the error at the output. This process is known as back-propagation. Activation functions make the back-propagation possible since the gradients are supplied along with the error to update the weights and biases. 

**Why do we need Non-linear activation function?**

A neural network without an activation function is essentially just a linear regression model. The activation function does the non-linear transformation to the input making it capable to learn and perform more complex tasks. 

**Mathematical proof**

Suppose we have a Neural net like this :- 

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ae814e08-9076-474f-a78c-ba524d68b635)

Elements of the diagram are as follows: 

**Hidden layer i.e. layer 1**:

z(1) = W(1)X + b(1) a(1)

Here,

z(1) is the vectorized output of layer 1

W(1) be the vectorized weights assigned to neurons of hidden layer i.e. w1, w2, w3 and w4

X be the vectorized input features i.e. i1 and i2

b is the vectorized bias assigned to neurons in hidden layer i.e. b1 and b2

a(1) is the vectorized form of any linear function.

(Note: We are not considering activation function here)

**Layer 2 i.e. output layer** :-

**Note : Input for layer 2 is output from layer 1**

z(2) = W(2)a(1) + b(2)  

a(2) = z(2) 

Calculation at Output layer

z(2) = (W(2) * [W(1)X + b(1)]) + b(2)

z(2) = [W(2) * W(1)] * X + [W(2)*b(1) + b(2)]

Let, 

    [W(2) * W(1)] = W

    [W(2)*b(1) + b(2)] = b

Final output : z(2) = W*X + b

which is again a linear function.

This observation results again in a linear function even after applying a hidden layer, hence we can conclude that, doesn’t matter how many hidden layer we attach in neural net, all layers will behave same way because the composition of two linear function is a linear function itself. Neuron can not learn with just a linear function attached to it. A non-linear activation function will let it learn as per the difference w.r.t error. Hence we need an activation function. 


**Variants of Activation Function**: (**Mathematical Exploration: Derive the Activation function formula and demonstrate its output range.**)


Several types of activation functions have been proposed and utilized in neural network architectures, each with its characteristics, advantages, and disadvantages. Below are some widely studied activation functions:

**1. Sigmoid Function (Logistic Function)**:

The sigmoid function, defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ba2fe442-da1f-4bd8-bd28-98ccd3ae5b7a), maps the input to the range (0, 1). Historically popular, it is mainly used in the output layer of binary classification tasks. However, its tendency to saturate and the vanishing gradient problem limit its effectiveness in deeper networks.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/13f1c962-6017-4037-b3ae-85958b89a702)

It is a function which is plotted as ‘S’ shaped graph.

**Equation** : A = 1/(1 + e-x)

**Nature** : Non-linear. Notice that X values lies between -2 to 2, Y values are very steep. This means, small changes in x would also bring about large changes in the value of Y.

**Value Range** : 0 to 1

**Uses** : Usually used in output layer of a binary classification, where result is either 0 or 1, as value for sigmoid function lies between 0 and 1 only so, result can be predicted easily to be 1 if value is greater than 0.5 and 0 otherwise.


**2. Hyperbolic Tangent Function (tanh)**:

Tanh function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/587140ac-dae6-43c8-be0c-13e7427a4602) ,outputs values in the range (-1, 1). Similar to the sigmoid function, tanh is also prone to the vanishing gradient problem but is preferred for its zero-centered output, which aids in faster convergence.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/d5ebc422-1d60-426c-8551-6aa95de54874)

The activation that works almost always better than sigmoid function is Tanh function also known as Tangent Hyperbolic function. It’s actually mathematically shifted version of the sigmoid function. Both are similar and can be derived from each other.

**Equation**:-

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/9f366d3c-3bc7-42b9-aef5-bf33f4fa1ab6)

**Value Range**:- -1 to +1

**Nature** :- non-linear

**Uses** :- Usually used in hidden layers of a neural network as it’s values lies between -1 to 1 hence the mean for the hidden layer comes out be 0 or very close to it, hence helps in centering the data by bringing mean close to 0. This makes learning for the next layer much easier.


**3. Rectified Linear Unit (ReLU)**:

The ReLU function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/20a43ac9-dd85-4c7f-a437-cfbca2eee71b), is currently the most widely used activation function due to its simplicity and effectiveness. It avoids the vanishing gradient problem and accelerates convergence, making it suitable for deep neural networks.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/540af06a-aef9-4fa9-ae53-bb0e590bf96a)

It Stands for Rectified linear unit. It is the most widely used activation function. Chiefly implemented in hidden layers of Neural network.

**Equation** :- A(x) = max(0,x). It gives an output x if x is positive and 0 otherwise.

**Value Range** :- [0, inf)

**Nature** :- non-linear, which means we can easily backpropagate the errors and have multiple layers of neurons being activated by the ReLU function.

**Uses** :- ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. At a time only a few neurons are activated making the network sparse making it efficient and easy for computation.

In simple words, RELU learns much faster than sigmoid and Tanh function.


**4. Leaky ReLU**:

Leaky ReLU addresses the "dying ReLU" problem by allowing a small gradient for negative inputs. It is defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/fc386701-40ae-4f93-96d2-ccd38e860974) ,where ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/41f63c8e-e227-450d-bc4b-34805b12adfb) is a small positive constant.

Using this function, we can convert negative values to make them close to 0 but not actually 0, solving the dying ReLU issue that arises from using the standard ReLU function during neural network training.

**Note**: If we set the value of alpha to 0 this function will act as the standard ReLU function.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ec3b3af4-5e81-4e86-ac86-25b0c134371a)

Let's suppose we use a small constant value for the variable alpha. In that case, all negative values on the x-axis representing the input to the function get mapped close to zero while the positive values remain unchanged.

**Note**: The value of the constant (alpha) is determined before training, i.e. it is not learned during training.

**5. Exponential Linear Unit (ELU)**:

ELU function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/06979617-c781-4a3b-8c2d-37d7e054b142) ,offers improved robustness to the saturation problem observed in ReLU. It ensures smooth gradients for both positive and negative inputs.

**Mathematical Definition of ELU**

The mathematical definition of this algorithm is:

Given an input x, the output f(x) is such that:

f(x) = x, for x> 0

f(x) = 𝜇(exp(x)- 1), for x ≤ 0

f'(x) = 1, for x>0

f'(x) = 𝜇(exp(x)), for x ≤ 0

where 𝜇 > 0

Exponential Linear Units are different from other linear units and activation functions because ELU's can take negetive inputs. Meaning, the ELU algorithm can process negetive inputs(denoted by x), into usefull and significant outputs.

If x keeps reducing past zero, eventually, the output of the ELU will be capped at -1, as the limit of exp(x) as x approaches negetive infinity is 0. The value for 𝜇 is chosen to control what we want this cap to be regardless of how low the input gets. This is called the saturation point. At the saturation point, and below, there is very little difference in the output of this function(approximately 𝜇), and hence there’s little to no variation(differential) in the information delivered from this node to the other node in the forward propagation.


In contrast to other linear unit activation functions, ELUs give negative outputs(i.e activations). These allow for the mean of the activations to be closer to 0, which is closer to the natural gradient, so the outputs are more accurate. This reduced difference in the unit gradient and the natural gradient makes learning more efficient as the training of the model will hence converge faster.

**Learning Using the Derivative of ELU**

Convolutional Neural Networks employs the use of back propagation algorithms during learning. Basically, the algorithm is going to go back into the neurons to learn the historical steps taken to reach an outcome.

Forward propagation is the steps taken to reach an outcome from input to output. The error of the algorithm is calculated by the (actual value - the outcome) sqaured / 2. Essentially, what back propagation does is to go back and optimize the weights of each node. It does this by finding the effect on the error when you change the weights by a small value(i.e d(error)/d(weight)).So for the node that uses the ELU activation function, the differential of the ELU is needed and will be used in reference to the differential of the output error.

Now let's focus on the derivative function of ELU.

f'(x) = 1, for x>0

f'(x) = 𝜇(exp(x)), for x ≤ 0

for x ≤ 0,

f(x) = 𝜇(exp(x)-1)

hence,

f'(x) = 𝜇 * (exp(x)-1)' + 𝜇' * (exp(x)-1), Product Rule

f'(x) = 𝜇 * (exp(x)) + 0

f'(x)= 𝜇(exp(x)

futher more,

f'(x)= 𝜇(exp(x) - 𝜇 + 𝜇

f'(x) = 𝜇(exp(x) - 1) + 𝜇

therefore,

f'(x) = f(x) + 𝜇

Since back propagation and forward propagation is done simultaneously, we need a function for the derivative of f(x) to have low computational cost. Since the value of f(x) and 𝜇 is already stored you can get f'(x) by finding the sum of f(x) and 𝜇 at a lower computational cost.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/7edcab21-caca-4aab-9916-19036eecf59f)

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/26f18a24-f4c4-4d1c-a1ea-5c4cd415fc5f)


**6. Scaled Exponential Linear Unit (SELU)**:

SELU is designed to maintain the mean and variance of the activations across layers, promoting self-normalization in deep networks. It is defined as
![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/c1d3cf0a-ddfd-4cac-8939-bc5dad92a811) , with carefully chosen constants ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/e1c2f4d0-a7f3-46f4-8a80-b65fbf892e08) and ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/6f098de2-1a58-48d0-8317-a03f759e5ae3).

Where 

λ and α are the following approximate values:

λ≈ 1.0507009873554804934193349852946

a≈1.6732632423543772848170429916717

If x is larger than 0, the output result is x multiplied by lambda lambda. If the input value x is less than or equal to zero, we have a function that goes up to 0, which is our output y, when x is zero. Essentially, when x is smaller than zero, we take the exponential of the x-value minus 1, then we multiply it with alpha α and lambda λ.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/822a01ef-8e08-410c-89ff-47011bc2ab5e)

Unlike ReLU, it can get below 0, allowing the system to have a zero average output. As a result, the model may converge faster.

**SELU is known to be a self-normalizing function, but what is normalization?**

Normalization is a data preparation technique that involves changing the values of numeric columns in a dataset to a common scale. This is usually used when the attributes of the dataset have different ranges.

**There is 3 types of normalization:**

**Input normalization**: One example is scaling the pixel values of grey-scale photographs (0–255) to values between zero and one

**Batch normalization**: Values are changed between each layer of the network so that their mean is zero and their standard deviation is one.

**Internal normalization**: this is where SELU's magic happens. The key idea is that each layer keeps the previous layer's mean and variance.

So, **how does SELU make this possible?** 

More precisely, How can it adjust the mean and variance? Let's take another look at the graph:

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/a866296f-588d-4dbc-a2f3-5d6b79e3fa08)

For y to change the mean, the activation function needs both positive and negative values. Both options are available here. It is also why ReLU ReLU isn't a good option for a self-normalizing activation function since it can not output negative values.


**7. Softmax Function**:

Softmax function is utilized in the output layer for multi-class classification tasks. It transforms raw scores into probabilities, ensuring that the sum of probabilities across classes is 1.

SoftMax function turns logits value into probabilities by taking the exponents of each output and then normalizing each number by the sum of those exponents so that the entire output vector adds up to one. Logits are the raw score values produced by the last layer of the neural network before applying any activation function on it.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/de2943eb-bba7-41df-a59b-7d26765047a5)

The softmax function is similar to the sigmoid function, except that here in the denominator we sum together all of the things in our raw output. In simple words, when we calculate the value of softmax on a single raw output (e.g. z1) we cannot directly take the of z1 value alone. We have to consider z1, z2, z3, and z4 in the denominator.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ae886ae7-c1fc-4554-b2cc-4782f0854a2e)

The softmax function is also a type of sigmoid function but is handy when we are trying to handle multi- class classification problems.

**Nature** :- non-linear

**Uses** :- Usually used when trying to handle multiple classes. the softmax function was commonly found in the output layer of image classification problems.The softmax function would squeeze the outputs for each class between 0 and 1 and would also divide by the sum of the outputs. 

**Output**:- The softmax function is ideally used in the output layer of the classifier where we are actually trying to attain the probabilities to define the class of each input.

**The basic rule of thumb is if you really don’t know what activation function to use, then simply use RELU as it is a general activation function in hidden layers and is used in most cases these days.**

**If your output is for binary classification then, sigmoid function is very natural choice for output layer.**

**If your output is for multi-class classification then, Softmax is very useful to predict the probabilities of each classes.**

**8. Swish Function**:

Swish activation, proposed by Ramachandran et al., is defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/404e07ad-a896-45b1-8e76-08a325b75d72). It has been shown to outperform ReLU in certain scenarios, offering smoother gradients.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/f3bdc275-d43a-4b07-ab65-369b202cd90a)


**9. Linear Function**:

The linear activation function, also known as "no activation," or "identity function" (multiplied x1.0), is where the activation is proportional to the input.

The function doesn't do anything to the weighted sum of the input, it simply spits out the value it was given. 

Linear function has the equation similar to as of a straight line i.e. y = x

No matter how many layers we have, if all are linear in nature, the final activation function of last layer is nothing but just a linear function of the input of first layer.

**Range** : -inf to +inf

**Uses** : Linear activation function is used at just one place i.e. output layer.

**Issues** : If we will differentiate linear function to bring non-linearity, result will no more depend on input “x” and function will become constant, it won’t introduce any ground-breaking behavior to our algorithm.

**For example** : Calculation of price of a house is a regression problem. House price may have any big/small value, so we can apply linear activation at output layer. Even in this case neural net must have any non-linear function at hidden layers. 

Mathematically it can be represented as:

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/7ddeafb4-e951-4158-9a9a-5ded1df99bb1)


![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/07f64c54-da70-447c-b060-3d941305c085)


o	Discuss why activation functions are used in neural networks, focusing on the role of the Activation function.

**2.**	**Mathematical Exploration:**


o Calculate the derivative of the Activation function and explain its significance in the backpropapation process.

**3.	Programming Exercise:**

o Implement the Activation Activation Function in Python. Use the following prototype for your function:
def Activation_Function_Name(x): # Your implementation

o Create a small dataset or use an existing one to apply your function and visualize the results.
 
**4.	Analysis:**

o	Analyze the advantages and disadvantages of using the Activation Function in neural networks.

o	Discuss the impact of the Activation function on gradient descent and the problem of vanishing gradients.

**Assessment Criteria:**

•	**Understanding and Explanation**: Clarity and depth of the explanation about the Activation function and its role in neural networks.

•	**Mathematical Accuracy**: Correctness in the derivation and mathematical explanations provided.

•	**Code Quality**: Efficiency, readability, and correctness of the programming exercise.

•	**Analysis and Critical Thinking**: Depth of the analysis on the advantages, disadvantages, and implications of using the Activation function.

•	**Creativity and Initiative**: For the optional task, the complexity of the implemented neural network model and the thoroughness in evaluating its performance.




