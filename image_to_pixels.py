import numpy as np
from sklearn.datasets import fetch_openml

# Fetch data
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

# Training set: 1,000 images
X_train, y_train = mnist.data[:10000], mnist.target[:10000].astype(int)
train_data = np.column_stack((y_train, X_train / 255.0))
np.savetxt("inputs.txt", train_data, fmt='%.6f')

# Test set: 200 images (unseen)
X_test, y_test = mnist.data[10000:11000], mnist.target[10000:11000].astype(int)
test_data = np.column_stack((y_test, X_test / 255.0))
np.savetxt("test_inputs.txt", test_data, fmt='%.6f')

print("Files generated: inputs.txt (10000 samples), test_inputs.txt (1000 samples)")
