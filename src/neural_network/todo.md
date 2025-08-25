test-driven-development - define a test that describes how backpropagation behaviour SHOULD be like.

 - The costs should be generated.

 - The output layer should be a bunch of neurons which are "labelled" 0 through 9.

 - The desired output should be that the neuron for 2 is close to 100% and the other neurons are close to 0%.
 
  - The costs should be used to shift the activation values for the output value to reach the above desired output.
   - The test should assert that this shift occurs correctly.  






implementation:
 - neuron layer class is responsible for having the method that takes a neuron and a cost and determines the proportional changes which that neuron wants to make to every neuron in the previous layer.

 - neural network class uses the propchange function to generate a list of changes for each output neuron, then splits the changes into their weight and activation_value pairs, then adds up the corresponding weight values and applies them directly to the previous layer weights, then calls the recursive function on the previous layer using the activation_value (from pairs) as `costs` input