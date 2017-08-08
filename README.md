# CNN to classify MNIST using Keras and Tensorflow
<h4>Network Structure</h4?

<p>CNN with 4 layers has following architecture.</p>
<ul>
<li>input layer : 784 nodes (MNIST images size)</li>
<li>first convolution layer : 5x5x32</li>
<li>first max-pooling layer</li>
<li>second convolution layer : 5x5x64</li>
<li>second max-pooling layer</li>
<li>third fully-connected layer : 1024 nodes</li>
<li>output layer : 10 nodes (number of class for MNIST)</li>
</ul>

<h4>Usage</h4>
<h6>Train</h6>
<p><code>python mnist_cnn_keras.py</code>
<h6>Test</h6>
<p><code>python evaluate.py</code>

<h4>Results</h4>
This model achieves 99.5% accuracy after 10 Epochs
