#include "Perceptron.hpp"

int main() {
    const int m = 1000000;
    const int n = 30;

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    generate_data(X, y, m, n);

    int train_size = m * (1 - 0.2);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    Perceptron model(n);
    std::cout << "Training started..." << "\n";
    auto start = std::chrono::high_resolution_clock::now();
    model.batch_train(X_train, y_train, 100, 1000);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> delta = end - start;
    std::cout << "Training completed in " << delta.count() << " seconds" << "\n";

    double train_acc = model.evaluate(X_train, y_train);
    double test_acc = model.evaluate(X_test, y_test);
    std::cout << "Train accuracy: " << train_acc * 100 << "%" << "\n";
    std::cout << "Test accuracy: " << test_acc * 100 << "%" << "\n";
}