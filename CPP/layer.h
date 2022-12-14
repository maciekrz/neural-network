#pragma once
#include "matrices_vectors.h"
#include <string>

using namespace std;

class Layer {
protected:
    int neurons_curr;
    int neurons_prev;
    double** W; // weights
    double* b; // bias
    double* z; // neuron value
    double* a; // neuron value after activation
    double* a_prev; // neuron value after activation previous layer
    double* delta; // delta
    double* error; // error
    double learning_rate;
    string activation_function;

public:
    Layer() { }

    Layer(int neurons_curr, int neurons_prev, string activation, double learning_rate)
    {
        this->neurons_curr = neurons_curr;
        this->neurons_prev = neurons_prev;
        this->W = initialize_layer_weights(neurons_curr, neurons_prev);
        this->b = initialize_layer_bias(neurons_curr);
        this->z = zeros_1d(neurons_curr);
        this->a = zeros_1d(neurons_curr);
        this->a_prev = zeros_1d(neurons_curr);
        this->delta = zeros_1d(neurons_curr);
        this->error = zeros_1d(neurons_curr);
        this->learning_rate = learning_rate;
        this->activation_function = activation;
    }

    // check
    double* forward(int neurons_prev, double* a_prev)
    {
        // z -> a_prev * W + b
        // a -> activation(z)
        this->z = multiply_matrix_by_vector(this->neurons_curr, neurons_prev, this->W, a_prev);

        if (this->activation_function == "sigmoid") {
            this->a = sigmoid_vector(this->neurons_curr, this->z);
        } else if (this->activation_function == "relu") {
            this->a = relu_vector(this->neurons_curr, this->z);
        } else {
            throw invalid_argument("Activation function must be either 'sigmoid' or 'relu'");
        }

        this->a_prev = a_prev;
        return this->a;
    }

    void backward(int neurons_next, double** W_next, double* b_next, double* delta_next)
    {
        // error = W_next * delta_next + b_next * delta_next
        // delta -> error . activation_derivative(a) . -> element wise multiplications [1, 2, 3] . [2, 2, 2] = [1*2, 2*2, 3*2]
        // W -> W - learning_rate * a_prev.T * delta  -> [1, 2].T * [1, 2] = matrix
        // b -> b - learning_rate * delta

        // weight
        double** W_nextT = transpose(neurons_next, this->neurons_curr, W_next);
        this->error = multiply_matrix_by_vector(this->neurons_curr, neurons_next, W_nextT, delta_next);

        // delta
        if (this->activation_function == "sigmoid") {
            this->delta = element_wise_multiply(this->neurons_curr, this->error, derivative_sigmoid_vector(this->neurons_curr, this->a));
        } else if (this->activation_function == "relu") {
            this->delta = element_wise_multiply(this->neurons_curr, this->error, derivative_relu_vector(this->neurons_curr, this->a));
        } else {
            throw invalid_argument("Activation function must be either 'sigmoid' or 'relu'");
        }

        // update weights
        double** w = multiply_vectorT_by_vector(this->neurons_curr, this->neurons_prev, this->delta, this->a_prev);
        double** w1 = multiply_matrix_by_constant(this->neurons_curr, this->neurons_prev, w, this->learning_rate);
        this->W = substract_matrices(this->neurons_curr, this->neurons_prev, this->W, w1);

        // update bias
        this->b = substract_vectors(neurons_curr, this->b, element_wise_multiply(this->neurons_curr, this->b, this->delta));
    }

    ~Layer() { }

    double* access_a()
    {
        return this->a;
    }

    double* access_delta()
    {
        return this->delta;
    }

    double* access_b()
    {
        return this->b;
    }

    double** access_W()
    {
        return this->W;
    }

    int access_neurons_curr()
    {
        return this->neurons_curr;
    }
};

class OutputLayer : public Layer {
public:
    using Layer::forward;
    using Layer::Layer;
    void backward(double* y)
    {
        // error -> a - y
        // delta -> error_w* activation_derivative(a)
        //
        // weight
        this->error = substract_vectors(this->neurons_curr, this->a, y);
        if (this->activation_function == "sigmoid") {
            this->delta = element_wise_multiply(this->neurons_curr, this->error, derivative_sigmoid_vector(this->neurons_curr, this->a));
        } else if (this->activation_function == "relu") {
            this->delta = element_wise_multiply(this->neurons_curr, this->error, derivative_relu_vector(this->neurons_curr, this->a));
        } else {
            throw invalid_argument("Activation function must be either 'sigmoid' or 'relu'");
        }

        // update weights
        double** w = multiply_vectorT_by_vector(this->neurons_curr, this->neurons_prev, this->delta, this->a_prev);
        double** w1 = multiply_matrix_by_constant(this->neurons_curr, this->neurons_prev, w, this->learning_rate);
        this->W = substract_matrices(this->neurons_curr, this->neurons_prev, this->W, w1);

        // update bias
        this->b = substract_vectors(neurons_curr, this->b, element_wise_multiply(this->neurons_curr, this->b, this->delta));
    }
};

void test_layer()
{
    double x_raw[2] = { 1, 2 };
    double y_raw[1] = { 0.5 };

    double* x = new double[2];
    double* y = new double[1];

    for (int i = 0; i < 2; i++) {
        x[i] = x_raw[i];
    }

    for (int i = 0; i < 1; i++) {
        y[i] = y_raw[i];
    }

    Layer hidden_layer = Layer(2, 2, "relu", 0.01);
    OutputLayer out_layer = OutputLayer(1, 2, "sigmoid", 0.01);

    hidden_layer.forward(2, x);
    out_layer.forward(2, hidden_layer.access_a());

    out_layer.backward(y);
    hidden_layer.backward(out_layer.access_neurons_curr(), out_layer.access_W(), out_layer.access_b(), out_layer.access_delta());

    delete[] x;
    delete[] y;
}
