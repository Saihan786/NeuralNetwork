test-driven-development - define a test that describes how backpropagation behaviour SHOULD be like.

 - The costs should be generated.

 - The output layer should be a bunch of neurons which are "labelled" 0 through 9.

 - The desired output should be that the neuron for 2 is close to 100% and the other neurons are close to 0%.
 
  - The costs should be used to shift the activation values for the output value to reach the above desired output.
   - The test should assert that this shift occurs correctly.  





TODO:
   - (feature/cost-function)
      - (DONE) ACTIONS
         - Calculate squared diffs and return sum (which represents cost)
   
      - LOGIC
         - cost should be calculated by using squared differences instead of direct differences between actual and expected values.
            - This squared difference calculation will be referred to as the cost function, inshaallaah.

            - The input to the cost function is the list of activation values for the output neurons and the output is the cost.
               - The actval list is determined by the input data (we're assuming is unchanged in this example for every iterative calculation of cost during training on one example) and the weights+biases of the network.

               - This means the input to the cost function is implicitly the weights+biases and the output is the cost.

               - On a graph, the y-value of a point on the cost function is the returned cost, and the x-value is the weights+biases of the network when generating that cost. To minimise the y-value, we find the corresponding w+b through partial differentiation.


   - (feature/backpropagation)
      - LOGIC
         - (priority) Backprop should not directly use activation values for proportional changes but should use some derivative
            - Add `def backpropagation` (which is really `def minimise_cost_function`).
               - Consider the xy graph (as a simple example - when there are more neurons, there are more dimensions to the graph).

               - We need to move in a downward direction on the graph (i.e., change w+b to reduce the cost).
                  - Find the gradient (type=List[float], corresponds to w+b for all neurons) at the current position.

                  - The negative version of this gradient can be used to reduce the y-value (by adjusting the w+b).
                  
                  - Backpropagation is used to find this negative gradient. Backpropagation returns a list of changes to w and to b over the whole network.

            - Backpropagation (List[float]).
               - Consider a network with all w+b+a set. We want to know the list of changes (List[float]) that we apply to every w+b in the network (a is determined by w+b).
               
               - The above list of changes can be thought of as a list of partial derivatives, each of which describes how changes to the sqrdiff for its output neuron are affected by changes to just the weights of the network.
                  - Split the partial derivative into three partials (which multiply together to make the original):
                     - How changes to the sqrdiff are affected by the activation value of that output neuron.
                        - sqrdiff = (actval - desired)^2 -> d_cost = 2 * (actval - desired)

                        - Intuitively, changing the activation value to affect the sqrdiff has a more drastic effect if the difference between the desired actval and the actual actval are large.
                     
                     - How changes to that activation value are affected by changes to the formula for that activation value (w*a_prev + b)
                        - not sure. 3b1b just showed notation for the derivative of the sigmoid function (which i'm not even using) rather than affecting the formula.

                        - Can probably skip this as sigmoid isn't being used here.
                     
                     - How changes to the result of that formula (result being a_current) are affected by just the weight of the output neuron.
                        - a_current = w*a_prev + b -> d_a_current = a_prev

                        - Intuitively, changing the weight to affect a_current has a stronger effect when the actval of the previous neuron is larger. 

                  - The above conclusions mean that changing the weights of the output neuron to change its sqrdiff has a greater effect when
                     - diff(actval, desired) is larger 
                     - a_prev is larger

                  - Finally, now we have one change in the list of changes.
                     - The change is to one weight and is equal to `2 * (actval - desired) * a_prev`

               - Now to find a change to a bias (we've defined how we get a change for a weight above).
                  - Changing the sqrdiff with respect to b (as a partial derivative) can be split into:
                     - Change sqrdiff with respect to actval (as before)
                        - sqrdiff = (actval - desired)^2 -> d_sqrdiff = 2 * (actval - desired)

                     - Change actval with respect to b
                        - actval = w*a + b -> d_actval_respect_to_b = 1

                     - (ignoring the sigmoid function partial derivative)
                  
                  - So now, a change to the bias in the list of changes can be calculated as:
                     - 2*(actval-desired)*1 -> `2 * (actval - desired)`

            - (C) (generalising relationships for all sqrdiffs) Keep in mind that the above two functions (in the maths sense of the term) for weight change and bias change only apply for one weight going into an output neuron and one bias of one neuron. To generalise (and find all changes in the list), you iterate this over all w+b in the network.
               - As an example, to get the list of bias changes for the output neurons, you'd get:
                  - [
                     2 * (actval_output_neuron_1 - desired_output_neuron_1),
                     2 * (actval_output_neuron_2 - desired_output_neuron_2),
                     ...
                  ]
               
               - List of weight changes going into output_neuron_1:
                  - [
                     2 * (actval_output_neuron_1 - desired_output_neuron_1) * a_prev_neuron_1,
                     2 * (actval_output_neuron_1 - desired_output_neuron_1) * a_prev_neuron_2,
                     ...
                  ]

            - (simple) Also keep in mind the resulting list of changes can have a multiplier applied to reduce/increase how much you want the network to move towards a training example in one go.

         - This partial derivation has to be applied differently depending on whatever layer you're going to. The above is for the output layer. For the directly previous layer, you must find how cost is affected when a weight in the previous layer is affected (split into 6 partial derivatives, starting with dC / dA(OutputLayer)). How can this be generalised?

   - Other changes.
      - Generalise backprop logic to more than two layers
      
      - Enable variable size layers


   - Need to find out if it's fine to pass in Lists of values to assign to Neurons in Dicts (if a Neuron wants a proportional change for a particular Neuron and indexes it appropriately in the List of proportional changes, is it guaranteed that the indexing of the value in the list will always target the appropriate Neuron?)
      - Dicts preserve insertion order! So it's fine, but consider a safer alternative in the future.
         - ('insertion', so not if the dict is generated from an unordered data structure)

   - why are they all at the same memory address? (0x0352E4B0)

      ---------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------
      {<neural_network.neuron_classes.Neuron object at 0x0352EC90>: 100, <neural_network.neuron_classes.Neuron object at 0x0352E4B0>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E900>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E3D8>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E318>: 0}
      {<neural_network.neuron_classes.Neuron object at 0x0352EC90>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E4B0>: 100, <neural_network.neuron_classes.Neuron object at 0x0352E900>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E3D8>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E318>: 0}
      {<neural_network.neuron_classes.Neuron object at 0x0352EC90>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E4B0>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E900>: 100, <neural_network.neuron_classes.Neuron object at 0x0352E3D8>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E318>: 0}
      {<neural_network.neuron_classes.Neuron object at 0x0352EC90>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E4B0>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E900>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E3D8>: 100, <neural_network.neuron_classes.Neuron object at 0x0352E318>: 0}
      {<neural_network.neuron_classes.Neuron object at 0x0352EC90>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E4B0>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E900>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E3D8>: 0, <neural_network.neuron_classes.Neuron object at 0x0352E318>: 100}


   - separate out NeuronLayer class into NeuronLayer, InitialNeuronLayer, OutputNeuronLayer (and give these three their own file)

implementation:
 - neuron layer class is responsible for having the method that takes a neuron and a cost and determines the proportional changes which that neuron wants to make to every neuron in the previous layer.

 - neural network class uses the propchange function to generate a list of changes for each output neuron, then splits the changes into their weight and activation_value pairs, then adds up the corresponding weight values and applies them directly to the previous layer weights, then calls the recursive function on the previous layer using the activation_value (from pairs) as `costs` input