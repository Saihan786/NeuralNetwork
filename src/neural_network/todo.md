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
      - (priority) Backprop should not directly use activation values for proportional changes but should use some derivative
      
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