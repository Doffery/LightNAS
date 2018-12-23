Aim to build a light and fast version of Neural Network Archetecture Search solution. (Failed)

Basic idea:
We try to build the NN from a dag. The node is layer. The edge is the flow of
data. We try to find the NN in a two-step iterating. First, we decide the layers
for each node by genetic algorithm. Second, with the reinforcement learning 
method, model learn to generate paths in dag given the fixed layers.
LSTM as agent. Generated path is action. The average accuracy of NN made up by
combing this path with others as reward.

Outcome:
Only 90.1% precision for cifar10.
The main problem is the generated NN architecture is simple, namely in a shallow 
local optimal. The simple (short or with less conv layers) paths outrun other
paths during evolution. One tried but didn't find a good solution.

Dong
--
