/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author rudo5
 */
public class Main {
    
    private static ArrayList<double[]> readInData(String pathname){
        ArrayList<double[]> data = new ArrayList();
        try{
            File file = new File(pathname);
            BufferedReader br = new BufferedReader(new FileReader(file));
            int row =0, col = 0;
            String line = null;
            Scanner sc;
            while((line = br.readLine()) != null && line.trim() != ""){
                col = 0;
                sc = new Scanner(line).useDelimiter(",");
                double[] array = new double[5];
                while(sc.hasNext() && col < 4)array[col++] = Double.parseDouble(sc.next().trim());
                String label = sc.next().trim();
                switch(label){
                    case "Iris-setosa" : array[col++] = 0;
                        break;
                    case "Iris-versicolor" : array[col++] = 1;
                        break;
                    case "Iris-virginica": array[col++] = 2;
                        break;
                }
                sc.close();
                data.add(array);
            }
            br.close();
            
        }catch(Exception e){
            System.err.println("Error: "+e.getMessage());
        }
        return data;
    }
    
    public static void main(String[] args) {
        
        int classes = 3;
        int numberHiddenLayers = 1;
        int numberNodesInHiddenLayers = 3;
        double learningRate = 0.04;
        int maximumEpochs = 50;
        String parameters   = "C:\\Users\\rudo5\\Documents\\Homework\\COS 314\\A\\03\\Iris Dataset\\parameters.txt";
        String train        = "C:\\Users\\rudo5\\Documents\\Homework\\COS 314\\A\\03\\Iris Dataset\\train.data";
        String test         = "C:\\Users\\rudo5\\Documents\\Homework\\COS 314\\A\\03\\Iris Dataset\\test.data";
        String validation   = "C:\\Users\\rudo5\\Documents\\Homework\\COS 314\\A\\03\\Iris Dataset\\iris.data";
        /*
        parameters  = args[0];
        train       = args[1];
        test        = args[2];
        validation  = args[3];
        
        try{
            File file           = new File(parameters);
            BufferedReader br   = new BufferedReader(new FileReader(file));
            String line         = br.readLine();
            Scanner sc          = new Scanner(line).useDelimiter(" ");
            classes                     = Integer.parseInt(sc.next().trim());
            numberHiddenLayers          = Integer.parseInt(sc.next().trim());
            numberNodesInHiddenLayers   = Integer.parseInt(sc.next().trim());
            learningRate                = Double.parseDouble(sc.next().trim());
            maximumEpochs               = Integer.parseInt(sc.next().trim());
            
            sc.close();
            br.close();
        }catch(Exception e){
            System.err.println("Error:  "+e.getMessage());
        }
        */
        
        ArrayList<double[]> trainData         = readInData(train);
        ArrayList<double[]> testData          = readInData(test);
        ArrayList<double[]> validationData    = readInData(validation);
        
        double [][] train_input         = new double[trainData.size()][4];
        double [][] test_input          = new double[testData.size()][4];
        double [][] validation_input    = new double[validationData.size()][4];
        
        double [] train_target          = new double[trainData.size()];
        double [] test_target           = new double[testData.size()];
        double [] validation_target     = new double[validationData.size()];
        
        
        for (int i = 0; i < trainData.size(); i++) {
            int j = 0;
            for (; j < trainData.get(i).length-1; j++) {
                train_input[i][j] = trainData.get(i)[j];
            }
            train_target[i] = trainData.get(i)[j];
        }
        
        for (int i = 0; i < testData.size(); i++) {
            int j =0;
            for(; j < testData.get(i).length-1; j++){
                test_input[i][j] = testData.get(i)[j];
            }
            test_target[i] = testData.get(i)[j];
        }
        
        for (int i = 0; i < validationData.size(); i++) {
            int j = 0;
            for (; j < validationData.get(i).length-1; j++) {
                validation_input[i][j] = validationData.get(i)[j];
            }
            validation_target[i] = validationData.get(i)[j];
        }
        
        
        

        
        
        NeuralNetwork nn = new NeuralNetwork(classes                        //classe
                                             ,numberHiddenLayers            //number hidden layers
                                             ,numberNodesInHiddenLayers     //number of nodes in a hidden layer
                                             ,learningRate                  //learning rate
                                             ,maximumEpochs                 //maximum epochs //50
                                            );
        
        //3,1,3, 0.05,6 -> 94.6     0.6*((levels[numberLayers - 1].get((int)Math.floor(t)).output) - (levels[numberLayers - 1].get(k).output))
        //3,1,3,0.05,8 -> 95        0.7*((levels[numberLayers - 1].get((int)Math.floor(t)).output) - (levels[numberLayers - 1].get(k).output))
        //3,1,3,0.05,8 -> 96        0.8*((levels[numberLayers - 1].get((int)Math.floor(t)).output) - (levels[numberLayers - 1].get(k).output))
        //3,1,3,0.04,13 -> 97.5     0.8*((levels[numberLayers - 1].get((int)Math.floor(t)).output) - (levels[numberLayers - 1].get(k).output))
        //nn.Print();
        nn.train(train_input, train_target, test_input, test_target, validation_input, validation_target);
    }
}
