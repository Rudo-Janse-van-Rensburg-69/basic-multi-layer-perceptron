/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork {

    private final Random random;
    private final ArrayList<Neuron>[] levels;
    private final ArrayList<Double>[] weights;
    private final ArrayList<Double>[] biases;
    private final double learningRate;
    private final int maxEpochs;
    private final int numClasses;
    private final int numNodesHidden;
    private final int numberLayers;

    public NeuralNetwork(int classes, int hidden, int numHiddenNodes, double lr, int me) {
        this.numNodesHidden = numHiddenNodes;
        this.numClasses = classes;
        this.learningRate = lr;
        this.maxEpochs = me;
        this.random = new Random(42069);
        this.numberLayers = 1 + hidden + 1;//1 input +  number hidden + 1 output

        levels = (ArrayList<Neuron>[]) new ArrayList[numberLayers];
        weights = (ArrayList<Double>[]) new ArrayList[numberLayers - 1];
        biases = (ArrayList<Double>[]) new ArrayList[numberLayers - 1];
        /*Initialise Levels*/

        //Input 
        levels[0] = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            levels[0].add(new Neuron()); //create 4 input nodes
        }
        //Hidden
        for (int i = 1; i < numberLayers - 1; i++) {
            levels[i] = new ArrayList<>();
            for (int j = 0; j < this.numNodesHidden; j++) {
                levels[i].add(new Neuron());
            }
            biases[i - 1] = new ArrayList<>();
            for (int j = 0; j < levels[i].size(); j++) {
                //add a bias for every node in the current level
                biases[i - 1].add(Random());
            }
            weights[i - 1] = new ArrayList<>();
            for (int j = 0; j < levels[i].size(); j++) {
                //For each node in the current layer
                for (int k = 0; k < levels[i - 1].size(); k++) {
                    //add a weight from every node in the previous level to the current node
                    weights[i - 1].add(Random());
                }
            }
        }

        //Output
        levels[numberLayers - 1] = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            levels[numberLayers - 1].add(new Neuron()); //create 4 input nodes
        }
        biases[numberLayers - 2] = new ArrayList<>();
        for (int j = 0; j < levels[numberLayers - 1].size(); j++) {
            biases[numberLayers - 2].add(Random());
        }
        weights[numberLayers - 2] = new ArrayList<>();
        for (int j = 0; j < levels[numberLayers - 1].size(); j++) {
            //For each node in the current layer
            for (int k = 0; k < levels[numberLayers - 2].size(); k++) {
                //add a weight from every node in the previous level to the current node
                weights[numberLayers - 2].add(Random());
            }
        }
    }

    public void train(double[][] train_input, double[] train_target, double[][] test_input, double[] test_target,double[][] validation_input, double[] validation_target) {
        boolean debug = false;
        int EPOCH = 0;
        double accuracy = 0;
        double bestAccuracy = 0;
        //Print();
        do {
            if(debug){
                System.out.println("================================");
                System.out.println("EPOCH #" + (EPOCH + 1));
                System.out.println("================================");
            }
            
            for (int i = 0; i < train_input.length; i++) {
                if(debug){
                    System.out.println("--------------------------------");
                    System.out.println("Training Cycle #" + (i + 1));
                }
                Feedforward(train_input[i]);
                if(debug){
                    Print();
                }
                
                if (Classify(train_input[i]) != train_target[i]) {
                    if (debug) System.out.println("\n"+Classify(train_input[i]) + " != " + train_target[i] +"->INCORRECT -> backpropagating\n");
                    Backpropagate(train_target[i]);
                    Feedforward(train_input[i]);
                    if (debug) Print();
                } else if(debug)System.out.println(Classify(train_input[i]) + " == " + train_target[i] + "->CORRECT");
                
                if(debug){
                    System.out.println("Target      :   " + train_target[i]);
                    System.out.println("Classify    :   " + Classify(train_input[i]));
                    System.out.println("Accuracy    :   " + Accuracy(test_input, test_target, debug));
                    System.out.println("");
                    System.out.println("--------------------------------");
                }
                
            }
            accuracy = Accuracy(test_input, test_target, debug);
            if(bestAccuracy < accuracy){
                bestAccuracy = accuracy;
                WriteToTextfile(EPOCH+1,train_input, train_target, test_input, test_target,validation_input,validation_target);
            }
        } while (++EPOCH < maxEpochs);
    }

    private void WriteToTextfile(int epoch, double[][] train_input, double[] train_target, double[][] test_input, double[] test_target,double[][] validation_input, double[] validation_target){
        File file = new File("output.txt");
        try(Writer writer = new BufferedWriter(new FileWriter(file))){
            writer.write(toString(epoch,train_input,train_target,test_input,test_target,validation_input,validation_target));
        }catch(IOException e){
            System.err.println(e.getMessage());
        }
    }
    
    private String toString(int epoch,double[][] train_input, double[] train_target, double[][] test_input, double[] test_target,double[][] validation_input, double[] validation_target){
        String output = "NUMBER EPOCHS : "+epoch;
        output += "\n======================================================\n";
        output += "WEIGHT MATRIX";
        output += "\n------------------------------------------------------\n";
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                output += (new DecimalFormat("0.0000").format(weights[i].get(j))) + " ";
            }
            output += "\n";
        }
        output += "======================================================\n";
        output += "BIAS MATRIX";
        output += "\n------------------------------------------------------\n";
        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].size(); j++) {
                output += (new DecimalFormat("0.0000").format(biases[i].get(j))) + " ";
            }
            output += "\n";
        }
        output += "======================================================\n";
        output += "Training Set Accuracy    : "+(new DecimalFormat("0.0000").format(Accuracy(train_input, train_target, false)));
        output += "\nTesting Set Accuracy     : "+(new DecimalFormat("0.0000").format(Accuracy(test_input, test_target, false)));
        output += "\nValidation Set Accuracy  : "+(new DecimalFormat("0.0000").format(Accuracy(validation_input, validation_target, false)));
        output += "\n======================================================\n";
        return output;
    }
    
    public double Accuracy(double[][] input, double[] target, boolean debug) {
        double correct = 0;
        for (int i = 0; i < input.length; i++) {
            if (debug) {
                System.out.print("input [ ");
                for (int j = 0; j < input[i].length; j++) {
                    System.out.print(input[i][j] + " ");
                }
                System.out.print("] => " + (target[i]) + " | ");
            }
            double classification = Classify(input[i]);
            if (debug) {
                System.out.print(new DecimalFormat("0.00").format(classification));
            }
            if (classification == target[i]) {
                ++correct;
                if (debug) {
                    System.out.print(" - correct");
                }

            } else {
                if (debug) {
                    System.out.print(" - wrong");
                }
            }
            if (debug) {
                System.out.print("\n");
            }
        }
        return (correct / target.length);
    }

    public double Classify(double[] input) {
        //INPUT: feed in input
        Feedforward(input);
        int classification = -1;
        double greatest = Integer.MIN_VALUE;
        for (int j = 0; j < numClasses; j++) {

            if (levels[numberLayers - 1].get(j).output > greatest) {
                classification = j;
                greatest = levels[numberLayers - 1].get(j).output;
            }
        }

        return classification;
    }

    public void Feedforward(double[] input) {
        //INPUT: feed in input
        for (int i = 0; i < levels[0].size(); i++) {
            levels[0].get(i).output = input[i];
        }

        //HIDDEN : for each node in the hidden layer
        int i = 1;
        for (; i < levels.length - 1; i++) {
            for (int j = 0; j < levels[i].size(); j++) {
                //For each node in the current level:
                //Invoke the activation function corresponding to each node in the current layer
                levels[i].get(j).F(CalculateN(i, j));
            }
        }

        //OUTPUT: 
        for (int j = 0; j < levels[i].size(); j++) {
            //For each node in the output level
            levels[i].get(j).F(CalculateN(i, j));
        }

    }

    public void Backpropagate(double t) {
        /*OUTPUT*/
        int m = levels[numberLayers - 1].size();// == size of t 

        //Error Infomation Term
        for (int k = 0; k < m; k++) {
            //For each output node
            double target = 0.5;
            double currentNode = k;
            if (currentNode == t) {
                target = 1;
            }
            //maybe remove this
            else{
               target = 0.8*((levels[numberLayers - 1].get((int)Math.floor(t)).output) - (levels[numberLayers - 1].get(k).output));
            }
            levels[numberLayers - 1].get(k).error
                    = (target - levels[numberLayers - 1].get(k).output)
                    * levels[numberLayers - 1].get(k).F_Derivative(CalculateN(numberLayers - 1, k));

            //Weight Error
            int j = levels[numberLayers - 2].size();
            for (int i = 0; i < j; i++) {
                //For each node in the last node in the hidden layer, adjust its weight to the current output node
                int pos = (i * levels[numberLayers - 1].size()) + k;
                weights[numberLayers - 2]
                        .set(pos, weights[numberLayers - 2]
                                .get(pos)
                                + (learningRate
                                * levels[numberLayers - 1].get(k).error
                                * levels[numberLayers - 1].get(k).output)
                        );
            }
            //Bias Error Term
            biases[biases.length - 1].set(k, learningRate
                    * levels[levels.length - 1].get(k).error);
        }
        /*HIDDEN*/
        //For each hidden layer
        for (int h = numberLayers - 2; h >= 1; h--) {
            int j = levels[h].size();
            //For each hidden node
            for (int i = 0; i < j; i++) {
                levels[h].get(i).error
                        = CaculateSumDeltaInputs(h, i)
                        * levels[h].get(i).F_Derivative(CalculateN(h, i));
                int n = levels[h - 1].size();
                for (int l = 0; l < n; l++) {
                    //For Each Weight from the previous layer(h-1) to the current node i in layer (h)
                    //Weight Error
                    int pos = (l * j) + i;
                    weights[h - 1].set(pos, weights[h - 1].get(pos)
                            + (learningRate
                            * levels[h].get(i).error
                            * levels[h].get(i).output));
                }
                //Bias Error Term
                biases[h - 1].set(i, learningRate
                        * levels[h].get(i).error);

            }

        }
    }

    private double CalculateN(int level, int neuron) {
        double N = 0;
        if (level == 0) {
            //INPUT
            N = levels[level].get(neuron).output;
        } else {
            for (int i = 0; i < levels[level - 1].size(); i++) {
                N = N
                        + (weights[level - 1].get((i * levels[level].size()) + neuron)
                        * levels[level - 1].get(i).output);
            }

            N = N + (biases[level - 1].get(neuron));
        }
        return N;
    }

    private double CaculateSumDeltaInputs(int level, int neuron) {
        //sum of the products of the next layer's error with the weights from them to the current node 
        double sumDeltaInputs = 0;
        int m = levels[level + 1].size();
        for (int k = 0; k < m; k++) {
            //Add the product of its's error and the weight from the current neuron(in the current level) to that neuron(in the next level)
            sumDeltaInputs
                    += levels[level + 1].get(k).error
                    * weights[level].get((neuron * levels[level + 1].size()) + k);
        }
        return sumDeltaInputs;
    }

    private double Random() {
        return this.random.nextDouble() * ((1)) - 0.5;
    }

    public void Print() {
        //INPUT 
        System.out.print("INPUT[ ");
        for (int j = 0; j < levels[0].size(); j++) {
            System.out.print(levels[0].get(j).output + "    ");
        }
        System.out.print("]\n");

        //HIDDEN
        int i = 1;
        for (; i < (levels.length - 1); i++) {
            System.out.print("W   [");
            for (int j = 0; j < levels[i - 1].size(); j++) {
                System.out.print("{");
                for (int k = 0; k < levels[i].size(); k++) {
                    System.out.print(weights[i - 1].get((j * levels[i].size()) + k) + "   ");
                }
                System.out.print("}");
            }
            System.out.println("]");
            System.out.print("B   [ ");
            for (int j = 0; j < biases[i - 1].size(); j++) {
                System.out.print(biases[i - 1].get(j) + "   ");
            }
            System.out.print("]\n");
            System.out.print("HIDDEN #" + (i) + "[");
            for (int j = 0; j < levels[i].size(); j++) {
                System.out.print(levels[i].get(j).output + "    ");
            }
            System.out.print("]\n");
        }

        //OUTPUT
        System.out.print("W   [");
        for (int j = 0; j < numClasses; j++) {
            //for each node in the last output layer
            System.out.print("{");
            for (int k = 0; k < levels[i - 1].size(); k++) {
                //for each weight from the hidden node to each output node
                System.out.print(weights[i - 1].get((j * levels[i - 1].size()) + k) + "   ");
            }
            System.out.print("}");
        }
        System.out.println("]");
        System.out.print("B   [");
        for (int j = 0; j < levels[i].size(); j++) {
            System.out.print(biases[i - 1].get(j) + "   ");
        }
        System.out.print("]\n");
        System.out.print("OUTPUT [");
        for (int j = 0; j < levels[i].size(); j++) {
            System.out.print(levels[i].get(j).output + "    ");
        }
        System.out.print("]\n");

    }
}
