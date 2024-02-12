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

**Explanation:** We know, the neural network has neurons that work in correspondence with weight, bias, and their respective activation function. In a neural network, we would update the weights and biases of the neurons on the basis of the error at the output. This process is known as back-propagation. Activation functions make the back-propagation possible since the gradients are supplied along with the error to update the weights and biases. 

**Why do we need Non-linear activation function?**
A neural network without an activation function is essentially just a linear regression model. The activation function does the non-linear transformation to the input making it capable to learn and perform more complex tasks. 

**Variants of Activation Function**:

Several typesof activation functions have been proposed and utilized in neural network architectures, each with its characteristics, advantages, and disadvantages. Below are some widely studied activation functions:

**1. Sigmoid Function (Logistic Function)**:

The sigmoid function, defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ba2fe442-da1f-4bd8-bd28-98ccd3ae5b7a), maps the input to the range (0, 1). Historically popular, it is mainly used in the output layer of binary classification tasks. However, its tendency to saturate and the vanishing gradient problem limit its effectiveness in deeper networks.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/13f1c962-6017-4037-b3ae-85958b89a702)

It is a function which is plotted as ‘S’ shaped graph.

Equation : A = 1/(1 + e-x)

Nature : Non-linear. Notice that X values lies between -2 to 2, Y values are very steep. This means, small changes in x would also bring about large changes in the value of Y.
Value Range : 0 to 1

Uses : Usually used in output layer of a binary classification, where result is either 0 or 1, as value for sigmoid function lies between 0 and 1 only so, result can be predicted easily to be 1 if value is greater than 0.5 and 0 otherwise.


**2. Hyperbolic Tangent Function (tanh)**:

Tanh function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/587140ac-dae6-43c8-be0c-13e7427a4602) ,outputs values in the range (-1, 1). Similar to the sigmoid function, tanh is also prone to the vanishing gradient problem but is preferred for its zero-centered output, which aids in faster convergence.

**3. Rectified Linear Unit (ReLU)**:

The ReLU function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/20a43ac9-dd85-4c7f-a437-cfbca2eee71b), is currently the most widely used activation function due to its simplicity and effectiveness. It avoids the vanishing gradient problem and accelerates convergence, making it suitable for deep neural networks.

**4. Leaky ReLU**:

Leaky ReLU addresses the "dying ReLU" problem by allowing a small gradient for negative inputs. It is defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/fc386701-40ae-4f93-96d2-ccd38e860974) ,where ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/41f63c8e-e227-450d-bc4b-34805b12adfb) is a small positive constant.

**5. Exponential Linear Unit (ELU)**:

ELU function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/06979617-c781-4a3b-8c2d-37d7e054b142) ,offers improved robustness to the saturation problem observed in ReLU. It ensures smooth gradients for both positive and negative inputs.

**6. Scaled Exponential Linear Unit (SELU)**:

SELU is designed to maintain the mean and variance of the activations across layers, promoting self-normalization in deep networks. It is defined as
![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/c1d3cf0a-ddfd-4cac-8939-bc5dad92a811) , with carefully chosen constants ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/e1c2f4d0-a7f3-46f4-8a80-b65fbf892e08) and ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/6f098de2-1a58-48d0-8317-a03f759e5ae3).

**7. Softmax Function**:

Softmax function is utilized in the output layer for multi-class classification tasks. It transforms raw scores into probabilities, ensuring that the sum of probabilities across classes is 1.

SoftMax function turns logits value into probabilities by taking the exponents of each output and then normalizing each number by the sum of those exponents so that the entire output vector adds up to one. Logits are the raw score values produced by the last layer of the neural network before applying any activation function on it.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/de2943eb-bba7-41df-a59b-7d26765047a5)

The softmax function is similar to the sigmoid function, except that here in the denominator we sum together all of the things in our raw output. In simple words, when we calculate the value of softmax on a single raw output (e.g. z1) we cannot directly take the of z1 value alone. We have to consider z1, z2, z3, and z4 in the denominator.


**8. Swish Function**:

Swish activation, proposed by Ramachandran et al., is defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/404e07ad-a896-45b1-8e76-08a325b75d72). It has been shown to outperform ReLU in certain scenarios, offering smoother gradients.

**9. Linear Function**:

Linear function has the equation similar to as of a straight line i.e. y = x

No matter how many layers we have, if all are linear in nature, the final activation function of last layer is nothing but just a linear function of the input of first layer.

Range : -inf to +inf

Uses : Linear activation function is used at just one place i.e. output layer.

Issues : If we will differentiate linear function to bring non-linearity, result will no more depend on input “x” and function will become constant, it won’t introduce any ground-breaking behavior to our algorithm.

For example : Calculation of price of a house is a regression problem. House price may have any big/small value, so we can apply linear activation at output layer. Even in this case neural net must have any non-linear function at hidden layers. 


**Equation and Graph**:

Commonly used activation functions include:



o	Discuss why activation functions are used in neural networks, focusing on the role of the Activation function.

**2.**	**Mathematical Exploration:**

o Derive the Activation function formula and demonstrate its output range.

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




