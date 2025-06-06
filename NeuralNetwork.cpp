#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stddef.h>
#include <string>
#include <utility>
#include <vector>

double RandomDouble(){
    return (double)rand()/(double)RAND_MAX;
}

double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double relu(double z) {
    return (z > 0.0) ? z : 0.0;
}

double reluPrime(double z) {
    return (z > 0.0) ? 1.0 : 0.0;
}

class Neuron{
    private:
        std::vector<double> weights;
        double bias;
        double last_z;
        double last_activation;
        double last_prime;
        std::vector<double> last_inputs;
        bool isOutputLayer;
    public:
        Neuron(int input_size, bool isOutput){
            weights.resize(input_size);
            for(double &w : weights) w = RandomDouble();
            bias = RandomDouble();
            isOutputLayer = isOutput;
        }

        double activate(const std::vector<double>&inputs){
            last_inputs = inputs;
            double sum = 0.0;
            for(size_t i = 0; i < inputs.size();i++){
                sum += inputs[i]*weights[i];
            }
            sum += bias;
            last_z = sum;
            if(isOutputLayer){
                last_activation = sigmoid(sum);
                last_prime = last_activation * (1 - last_activation);
            }
            else {
                last_activation = relu(sum);
                last_prime = reluPrime(sum);
            }
             
            return last_activation;

        }
        
        double getLastPrime() const {
            return last_prime;
        }
        
        double getLastActivation() const{
            return last_activation;
        }
        const std::vector<double>& getAllWeights() const{
            return weights;
        }
        
        std::pair<std::vector<double>, double> computeGradients(double delta) const{
            std::vector<double> grad_w(weights.size(),0.0);
            for(size_t i = 0; i < weights.size();i++){
                grad_w[i] = delta * last_inputs[i];
            }
            double grad_b = delta;
            return {grad_w,grad_b};
        }

        void applyGradients(const std::vector<double>& grad_w,double grad_b,double learning_rate){
            for(size_t i = 0; i < weights.size();i++) {
                weights[i] -= learning_rate * grad_w[i];
            }
            bias -= learning_rate*grad_b;
        }
};

class Layer{
    private:
        std::vector<Neuron> neurons;
    public:
        Layer(size_t input_size, size_t input_neurons, bool makeOutputNeurons){
            for(size_t i = 0; i < input_neurons;i++){
                Neuron neuron(input_size, makeOutputNeurons);
                neurons.push_back(neuron);
            }
        }

        std::vector<double> forward(const std::vector<double>&inputs){
            std::vector<double> output;
            for(auto &neuron : neurons){
                double out = neuron.activate(inputs);
                output.push_back(out);
            }
            return output;
        }

        Neuron& getNeuron(int i){
            return neurons[i];
        }
        
        int size() const{
            return neurons.size();
        }
};

class Model{
    private:
        std::vector<Layer> layers;
        std::vector<std::vector<double>> activations;
    public:
        Model(const size_t input_size,const std::vector<double>& hiddenLayers,const size_t output_size ){
            // First layer
            layers.push_back(Layer(input_size,hiddenLayers[0],false));
            // Middle layers
            for(size_t i = 1; i < hiddenLayers.size();i++){
                int prev_size = hiddenLayers[i-1];
                int next_size = hiddenLayers[i];
                
                layers.push_back(Layer(prev_size,next_size,false));
            }
            // Last layer
            layers.push_back(Layer(hiddenLayers.back(),output_size,true));
        }

        std::vector<double> forwardFeed(const std::vector<double>& inputs){
            activations.clear();
            activations.push_back(inputs);
            std::vector<double> activation = inputs;
            for(auto &layer : layers){
                activation = layer.forward(activation);
                activations.push_back(activation);
            }
            return activations.back();
        }

        double computeLoss(const std::vector<double>& y_pred, const std::vector<double>& y_actual){
            double result = 0.0;
            for(size_t i = 0; i < y_actual.size();i++){
                result += std::pow((y_pred[i]-y_actual[i]),2);
            }
            return result/y_actual.size();
        }  
        
        std::vector<double> getLossGradient(const std::vector<double>& predicted , const std::vector<double>& target){
            std::vector<double> grad(target.size());
            for(size_t i = 0 ; i < target.size(); i++){
                double result = 2.0 * (predicted[i]-target[i]);
                grad[i] = result/target.size();
            }
            return grad;
        }

        double trainOneExample(const std::vector<double>& input,const std::vector<double>&target,double learning_rate){
            
            std::vector<double> predicted = forwardFeed(input);
            double loss = computeLoss(predicted,target);

            int num_layers = layers.size();
            std::vector<std::vector<double>> deltas(num_layers);

            int L = num_layers - 1;
            std::vector<double> dLoss_da = getLossGradient(predicted,target);

            int num_output_neurons = layers[L].size();
            deltas[L].resize(num_output_neurons);

            for (size_t i = 0; i < num_output_neurons; i++) {
                double prime = layers[L].getNeuron(i).getLastPrime();
                deltas[L][i] = dLoss_da[i] * prime;
            }
            
            for(int i = L - 1; i >= 0; i--){
                int num_curr = layers[i].size();
                int num_next = layers[i+1].size();
                deltas[i].resize(num_curr);
                
                for(int j = 0; j < num_curr; j++){
                    double sum = 0.0;
                    
                    for(int k = 0; k < num_next;k++){
                        const auto& next_weights = layers[i+1].getNeuron(k).getAllWeights();
                        if(j < next_weights.size()){
                        sum += next_weights[j] * deltas[i+1][k];
                        }
                    }

                    double prime = layers[i].getNeuron(j).getLastPrime();
                    deltas[i][j] = sum * prime;
                }
            }
            
            for ( int i = 0; i <= L;i++){
                int num_neurons = layers[i].size();
                const std::vector<double>& layer_input = activations[i];
                for(int j = 0; j < num_neurons;j++){
                    double delta_j = deltas[i][j];
                    std::vector<double> grad_w (layer_input.size());
                    for(size_t k = 0; k < layer_input.size();k++){
                        grad_w[k] = delta_j * layer_input[k];
                    }
                    double grad_b = delta_j;
                    layers[i].getNeuron(j).applyGradients(grad_w, grad_b ,learning_rate);
                } 
            }
            return loss; 
        }
        
        void train(const std::vector<std::vector<double>>& all_inputs,const std::vector<std::vector<double>>& all_targets
                   ,double learning_rate, int epochs){
            int n = all_inputs.size();
            for(int i = 0; i < epochs;i++){
                double epoch_loss = 0.0;
                for(int j = 0; j < n;j++){
                    double loss = trainOneExample(all_inputs[j], all_targets[j], learning_rate);
                    epoch_loss += loss;
                }
                
                // std::cout << "Epoch" << i << " | Avg loss = " << (epoch_loss/n) << std::endl;
            }
        }
};
    
std::vector<std::vector<double>> readCSV(const std::string& filename){
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file,line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss,value,',')) {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
    }
    return data;
}  

void saveResultsCSV(const std::vector<std::vector<double>>& trainX,
                    const std::vector<std::vector<double>>& trainY,
                    const std::vector<double>& predictions) {
    std::ofstream out("results.csv");
    // CSV header
    out << "x1,x2,actual,prediction\n";

    for (size_t i = 0; i < trainX.size(); ++i) {
        double x1 = trainX[i][0];
        double x2 = trainX[i][1];
        int actual_label = static_cast<int>(trainY[i][0]);       // 0 or 1
        int predicted_label = (predictions[i] >= 0.5) ? 1 : 0;     // threshold at 0.5

        out << x1 << "," 
            << x2 << "," 
            << actual_label << ","
            << predicted_label 
            << "\n";
    }
    out.close();
}

int main (void){
    srand(time(0));

    std::vector<std::vector<double>> trainX = readCSV("Spiral_X.csv"); 
    std::vector<std::vector<double>> trainY_raw = readCSV("Spiral_y.csv");

    std::vector<std::vector<double>> trainY;

    for (const auto& y : trainY_raw) {
        trainY.push_back({y[0]});
    }


    if (trainX.empty() || trainY.empty()) {
    std::cerr << "CSV data not loaded properly. Check if files exist and contain valid data." << std::endl;
    return 1;
    }
    

    Model model(2, {8,10}, 1); 

    double learning_rate = 0.01;
    int epochs = 5000;

    model.train(trainX, trainY, learning_rate, epochs);

    std::cout << "\n--- Final Predictions After Training ---\n";
    std::vector<double> preds;
    preds.reserve(trainX.size());
    for (int i = 0; i < trainX.size(); i++) {
        const auto& x = trainX[i];
        std::vector<double> out = model.forwardFeed(x);
        double p = out[0];
        preds.push_back(p);
        std::cout << "Input: (" 
                  << x[0] << ", " << x[1] << ")"
                  << " => Predicted: " << out[0]
                  << " | Target: " << trainY[i][0] 
                  << std::endl;

    }

    int correct = 0;
        for (size_t i = 0; i < trainX.size(); i++) {
            int predicted = (preds[i] >= 0.5) ? 1 : 0;
            int actual = static_cast<int>(trainY[i][0]);
            if (predicted == actual) correct++;
        }
    std::cout << "Training accuracy = "
          << (double)correct / trainX.size() * 100.0 << "%\n";

    saveResultsCSV(trainX, trainY, preds);
    return 0;
}

