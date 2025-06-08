#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <limits>

double sigmoid(double z) {
    if (z > 0) {
        return 1.0 / (1.0 + exp(-z));
    } else {
        double exp_z = exp(z);
        return exp_z / (1.0 + exp_z);
    }
}

class Perceptron {
    public:
        Perceptron(int size, double lr = 0.1) : input_size(size), learning_rate(lr) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, 0.01);
            
            weights.resize(input_size);
            for (int i = 0; i < input_size; i++) {
                weights[i] = dist(gen);
            }
            bias = dist(gen);
        }

        double forward(const std::vector<double>& input) {
            double z = bias;
            for (int i = 0; i < input_size; i++) {
                z += input[i] * weights[i];
            }
            return sigmoid(z);
        }

        void batch_train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& targets, int epochs = 100, int batch_size = 1000) {
            int m = inputs.size();
            for (int epoch = 0; epoch < epochs; epoch++) {
                double total_loss = 0.0;
                int batches = m / batch_size;
                for (int batch = 0; batch < batches; batch++) {
                    std::vector<double> grad_w(input_size, 0.0);
                    double grad_b = 0.0;
                    int startIndex = batch * batch_size;
                    int endIndex = std::min((batch + 1) * batch_size, m);
                    for (int i = startIndex; i < endIndex; i++) {
                        double prediction = forward(inputs[i]);
                        double error = prediction - targets[i];
                        total_loss += -targets[i] * log(std::max(prediction, 1e-15)) - (1 - targets[i]) * log(std::max(1 - prediction, 1e-15));
                        for (int j = 0; j < input_size; j++) {
                            grad_w[j] += error * inputs[i][j];
                        }
                        grad_b += error;
                    }
                    for (int j = 0; j < input_size; j++) {
                        weights[j] -= learning_rate * grad_w[j] / batch_size;
                    }
                    bias -= learning_rate * grad_b / batch_size;
                }
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / m << "\n";
            }
        }

        int predict(const std::vector<double>& input) {
            return forward(input) > 0.5 ? 1 : 0;
        }

        double evaluate(const std::vector<std::vector<double>>& inputs, const std::vector<double>& targets) {
            int correct = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (predict(inputs[i]) == (int)targets[i]) correct++;
            }
            return (double)correct / inputs.size();
        }

    private:
        std::vector<double> weights;
        double bias;
        int input_size;
        double learning_rate;
};

void generate_data(std::vector<std::vector<double>>& X, std::vector<double>& y, int m, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    X.resize(m, std::vector<double>(n));
    y.resize(m);
    std::vector<double> true_weights(n);
    for (int j = 0; j < n; ++j) {
        true_weights[j] = (j+1) * 0.1;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            X[i][j] = dist(gen);
        }
        double z = 0.0;
        for (int j = 0; j < n; ++j) {
            z += X[i][j] * true_weights[j];
        }
        y[i] = (sigmoid(z - 0.5) > 0.5) ? 1 : 0;  // Смещение -0.5 делает данные менее сбалансированными
    }
}

#endif // PERCEPTRON_HPP