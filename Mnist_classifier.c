#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <math.h>

typedef struct{
	int inlen;
	int outlen;
	double *biases;// array of biases with length = outlen
	double *weights;// 2D flattened array of weights
	double *delta;//len = outlen
	double *output;//same length as outlen
} Layer;

void Relu( double * output,int outlen){
	for (int i = 0; i < outlen; i++){
		if(output[i] < 0) output[i] = 0 ;
	}
}

void softmax(double *output, int outlen){

    double max = output[0];

    for(int i = 1; i < outlen; i++){
        if(output[i] > max)
            max = output[i];
    }

    double sum = 0.0;

    for(int i = 0; i < outlen; i++){
        output[i] = exp(output[i] - max);
        sum += output[i];
    }

    for(int i = 0; i < outlen; i++){
        output[i] /= sum;
    }
}

int predict(Layer *hidden, Layer *out, double *input_pixels) {
    // Hidden Layer Forward
    for (int j = 0; j < hidden->outlen; j++) {
        double mac = hidden->biases[j];
        for (int k = 0; k < hidden->inlen; k++) {
            mac += input_pixels[k] * hidden->weights[j * hidden->inlen + k];
        }
        hidden->output[j] = (mac > 0) ? mac : 0; // ReLU
    }

    // Output Layer Forward
    for (int j = 0; j < out->outlen; j++) {
        double mac = out->biases[j];
        for (int k = 0; k < out->inlen; k++) {
            mac += hidden->output[k] * out->weights[j * out->inlen + k];
        }
        out->output[j] = mac;
    }
    softmax(out->output, out->outlen);

    // Find the index of the maximum probability (Argmax)
    int max_index = 0;
    double max_prob = out->output[0];
    for (int i = 1; i < out->outlen; i++) {
        if (out->output[i] > max_prob) {
            max_prob = out->output[i];
            max_index = i;
        }
    }
    return max_index;
}

void shuffle(double **inputs, int n) {
    for (int i = n - 1; i > 0; i--) {
        // Pick a random index from 0 to i
        int j = rand() % (i + 1);
        
        // Swap the pointers (very fast because we aren't copying the 785 doubles)
        double *temp = inputs[i];
        inputs[i] = inputs[j];
        inputs[j] = temp;
    }
}

int main(){
	srand(time(NULL));
	int inlen = 784;
	int outlen = 10;
	int hlen = 128;
	int cycles = 500;
//	double inputs[100][785]; 0 index = ground truth and rest is the pixel data in normalized form
	int num_train = 10000;
	double **inputs = malloc(num_train * sizeof(double *));
	for (int i = 0; i < num_train; i++) {
		inputs[i] = malloc(785 * sizeof(double));
	}
	// declaring the layers
	Layer *hidden, *out;
	hidden = malloc(sizeof(Layer));
	out = malloc(sizeof(Layer));

	hidden->inlen = inlen;
	hidden->outlen =  hlen;
	hidden->biases = malloc(sizeof(double)*hlen);
	hidden->weights = malloc(sizeof(double)*hlen*inlen);
	hidden->delta = malloc(sizeof(double)*hlen);
	hidden->output = malloc(sizeof(double)*hlen);

	out->inlen = hlen;
	out->outlen =  outlen;
	out->biases = malloc(sizeof(double)*outlen);
	out->weights = malloc(sizeof(double)*hlen*outlen);
	out->delta = malloc(sizeof(double)*outlen);
	out->output = malloc(sizeof(double)*outlen);

	FILE* init_guessh; 
	FILE* init_guesso; // to get the random weights and biases initally
	FILE* input;
	init_guessh = fopen("./init_guesshlayer.txt", "r");
	init_guesso = fopen("./init_guessolayer.txt", "r");
	input = fopen("./inputs.txt","r");

	for(int i = 0; i < hlen*inlen; i++){
		fscanf(init_guessh, "%lf\n", (hidden->weights)+i);
	}
	for(int i = 0; i < hlen; i++){
		fscanf(init_guessh, "%lf\n", (hidden->biases)+i);
	}

	for(int i = 0; i < hlen*outlen; i++){
		fscanf(init_guesso, "%lf\n", (out->weights)+i);
	}
	for(int i = 0; i < outlen; i++){
		fscanf(init_guesso, "%lf\n", (out->biases)+i);
	}
	
	for(int i = 0; i < num_train; i++){
		for(int j = 0; j < 785; j++){
			fscanf(input,"%lf ", &inputs[i][j]);
		}
	}
	double NLL = 0.0;
	double eps = 0.0001;
	double avgloss;
	double eta = 0.001;
	while(cycles--){
		shuffle(inputs, num_train);
		avgloss = 0.0;
		for(int i = 0 ; i < num_train; i++){
		// implementing forward substitution
			NLL = 0.0;
			for(int j = 0; j < hidden->outlen; j++){
				double mac = hidden->biases[j];
				for(int k = 1 ; k < inlen+1; k++){
					mac += inputs[i][k]*(hidden->weights)[j*inlen+k-1];
				}
				hidden->output[j] = mac;
			}
			Relu(hidden->output, hlen);

			for(int j = 0; j < out->outlen; j++){
				double mac = out->biases[j];
				for(int k = 0; k < out->inlen; k++){
					mac += (hidden->output)[k]*(out->weights)[j*hlen+k];
				}
				out->output[j] = mac;
			}
			softmax(out->output, outlen);
			NLL = -1*log((out->output)[(int)inputs[i][0]]);
			avgloss = (avgloss*(i) + NLL)/(i+1);
			if(avgloss < eps){ 
				cycles = 0;
				break;
			}
			for(int j = 0; j < outlen; j++){
				(out->delta)[j] = (out->output)[j] - ((inputs[i][0] == j) ? 1 : 0);
			}
			for(int j = 0; j < hlen; j++){
				(hidden->delta)[j] = 0;
				for(int k = 0; k < outlen; k++){
					hidden->delta[j]+= (out->weights)[k*hlen + j] * (out->delta)[k];
				}
				(hidden->delta)[j] *= ((hidden->output)[j] > 0 ? 1 : 0);
			}
			//update weights and biases of output layer
			double lambda = 0.001; // The regularization strength
			for(int j = 0; j < outlen; j++){
				for(int k = 0; k < hlen; k++){
					double weight_val = (out->weights)[j*hlen + k];
					// The magic is adding (lambda * weight_val) to the gradient
					(out->weights)[j*hlen + k] -= eta * ((out->delta)[j]*(hidden->output)[k] + lambda * weight_val);
				}
				(out->biases)[j] -= eta*(out->delta)[j];
			}
			//update weights and biases of the hidden layer
			for(int j = 0; j < hlen; j++){
				for(int k = 0; k < inlen; k++){
					(hidden->weights)[j*inlen + k] -= eta*(hidden->delta)[j]*(inputs[i][k+1]);
				}
				(hidden->biases)[j] -= eta*(hidden->delta)[j];
			}
		}
		printf("epoch loss = %f\n", avgloss);
	}

	printf("\n Training Complete. Starting Inference Test \n");
    int correct_preds = 0;
    for (int i = 0; i < 100; i++) {
        // Skip the first element of each row (label) and pass the pixels
        int p = predict(hidden, out, &inputs[i][1]);
        int actual = (int)inputs[i][0];
        
        if (p == actual) {
            correct_preds++;
        }
        
        // Print first 10 results to see it in action
        if (i < 10) {
            printf("Image %d: Predicted %d, Actual %d\n", i, p, actual);
        }
    }
    printf("Final Accuracy on training set: %d/100 (%.1f%%)\n", correct_preds, (float)correct_preds);


	// --- Inference on Unseen Test Data ---
    FILE* test_file = fopen("./test_inputs.txt", "r");
    FILE* results_file = fopen("./test_results.txt", "w"); // File to save results

    if (test_file == NULL) {
        printf("Error: Could not open test_inputs.txt\n");
    } else if (results_file == NULL) {
        printf("Error: Could not create test_results.txt\n");
    } else {
        double test_image[785];
        int test_correct = 0;
        int total_test_samples = 1000;

        fprintf(results_file, "Index,Predicted,Actual,Status\n");
        printf("\n--- Running Inference on Unseen Data ---\n");

        for(int i = 0; i < total_test_samples; i++) {
            for(int j = 0; j < 785; j++) {
                if(fscanf(test_file, "%lf ", &test_image[j]) != 1) break;
            }
            
            // Run prediction
            int p = predict(hidden, out, &test_image[1]);
            int actual = (int)test_image[0];
            
            if(p == actual) {
                test_correct++;
            }

            // Write detailed result to file
            fprintf(results_file, "%d,%d,%d,%s\n", 
                    i, p, actual, (p == actual ? "SUCCESS" : "FAIL"));
        }

        float final_acc = (float)test_correct / total_test_samples * 100.0;
        printf("Test Accuracy: %d/%d (%.1f%%)\n", test_correct, total_test_samples, final_acc);
        printf("Detailed log saved to: test_results.txt\n");

        fclose(test_file);
        fclose(results_file);
    } 

	// Clean up memory
	for (int i = 0; i < num_train; i++) {
		free(inputs[i]);
	}
	free(inputs);
    free(hidden->weights); free(hidden->biases); free(hidden->delta); free(hidden->output); free(hidden);
    free(out->weights); free(out->biases); free(out->delta); free(out->output); free(out);
}

