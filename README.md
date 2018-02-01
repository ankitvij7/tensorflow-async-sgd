# tensorflow-async-sgd
A tensorflow python program running Logistic Regression Stochastic Gradient Descent Algorithm on the input dataset that is spread across 5 VMs in an asynchronous manner

## Dataset
[Kaggle Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) sponsored by Criteo Labs. It is available [here](http://pages.cs.wisc.edu/~akella/CS838/F16/assignment3/criteo-tfr-kaggle.tar.gz).

## Execution
* 5 different processes are spawned on every VM in the cluster, but the graph is initialized only once by the process in the master VM. This allows us to maintain a global state for the gradients.
* The gradient is updated in an asynchronous manner and all the iterations proceed independently without waiting for all the VMs to update the global gradient.
* Cross-validation runs on the master VM after every 100 iterations on 2000 testing samples.

## Environment
5 node cluster where each node has 20GB RAM and 4 cores 
