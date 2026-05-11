import numpy as np

def save_init_file(filename, in_len, out_len):
    # He Initialization: Std dev = sqrt(2/n_in)
    std = np.sqrt(2.0 / in_len)
    weights = np.random.normal(0, std, size=(out_len, in_len))
    biases = np.zeros(out_len) + 0.01
    
    with open(filename, 'w') as f:
        # Save weights first (row by row)
        for w in weights.flatten():
            f.write(f"{w:.8f}\n")
        # Save biases
        for b in biases:
            f.write(f"{b:.8f}\n")

# Generate files for your architecture
save_init_file("init_guesshlayer.txt", 784, 64)
save_init_file("init_guessolayer.txt", 64, 10)
print("Created initial guess files for hidden and output layers.")
