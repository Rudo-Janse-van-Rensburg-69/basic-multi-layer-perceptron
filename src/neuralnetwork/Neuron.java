/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

/**
 *
 * @author rudo5
 */
public class Neuron {
    public double output, error;
    private final static int ACTIVATION = 0;
    public Neuron() {
        output = 0;
        error = 0;
    }
    
    public double F(double n){
        switch(ACTIVATION){
            case 0: Sigmoid(n);
                break;
            case 1: BipolarSigmoid(n);
                break;
        }
        return output;
    }
    
    
    public double F_Derivative(double n){
        double derivative = 0;
        switch(ACTIVATION){
            case 0: derivative = DerivativeSigmoid(n);
                break;
            case 1: derivative = DerivativeBipolarSigmoid(n);
                break;
        }
        return derivative;
    }
    
    
    
    private void Sigmoid(double n){
        output = 1/(1+Math.exp((-1)*n));
    }
    
    private double DerivativeSigmoid(Double n){
        return F(n)*(1-F(n));
    }
    
    private void BipolarSigmoid(double n){
        output = (2/(1+Math.exp((-1)*n)))-1;
    }
    
    private double DerivativeBipolarSigmoid(Double n){
        return (1/2)*(1+F(n))*(1-F(n));
    }
    
}

