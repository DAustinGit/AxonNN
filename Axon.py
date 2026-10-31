
#import numpy as np
import cupy as cp
import pandas as pd
rng = cp.random.default_rng()
max_itterations = 400
from typing import Any




class network_layer:
  def __init__(self, m:int, n:int,output_features:int=1,regularisation:str=None, biases_init:str="rand", Bvals:int=1, Wvals:int=1, weights_init:str="rand",  activation:str = "ReLu", lambda_param:float = 0.00001, stage:str="hidden"):
    rng = cp.random.default_rng()
    self.m = m
    self.n = n
    self.stage = stage
    if stage == "output":
      self.biases_n = output_features
    else:
      self.biases_n = n

    if biases_init == "rand":
      self.biases=rng.uniform(-0.1,0.1, size =(1, self.biases_n))
    if biases_init == "fill":
      self.biases=cp.full(Bvals,Bvals, size =(1, self.biases_n))
    if weights_init == "rand":
      self.weights=rng.standard_normal(size=(self.m, self.n)) * cp.sqrt(2.0 / m)
    if weights_init =="fill":
      self.weights = cp.full(Wvals,Wvals, size =(self.m, self.n))
    self.lambda_param = lambda_param
    self.output = cp.array([[]])
    self.input = cp.array([[]])
    self.regularisation = regularisation
    self.activation = activation
    self.activation_functions = {
      "RelU":{"forwards":self.reLu_forwards, "back_propergation":self.relu_backpropergation},
      "tanh":{"forwards":self.tanh_forwards, "back_propergation":self.tanh_backpropergation},
      "linear":{"forwards":self.linear_forwards, "back_propergation":self.linear_backpropergation}
    }

    #print (f"{self.stage} layer with output function {self.activation}, m = {self.m}, n = {self.n}, biases initialisastion {biases_init}, weights initialisation {weights_init} regularisation {regularisation}")
    self.activation_func = self.activation_functions[self.activation]["forwards"]
    self.back_prop = self.activation_functions[self.activation]["back_propergation"]
    if regularisation == "L1":
      self.L = 1
    elif regularisation =="L2":
      self.L = 2
    else:
      self.L = 0
  def reLu_forwards(self, input)->cp.array:
    #print (f"shape of activation input {input.shape}")
    A = cp.maximum(0, input)
    #print (f"shape of A {A.shape}")
    return  A
  def relu_backpropergation(self,input)->cp.array:
    #print(f"m = {self.m}")
    #print(f"n = {self.n}") 
    return cp.where(input > 0, 1, 0)
  def tanh_forwards(self, input)->cp.array:
    return cp.tanh(input)
  def tanh_backpropergation(self, input)->cp.array:
    return  1 - cp.power(self, input, 2)
  def linear_forwards(self, input)->cp.array:
    return input
  def linear_backpropergation(self, input)->cp.array:
    return  cp.ones_like(input)


  
  def forward (self, input):
    self.input = cp.array(input)
    #print (f"shape of input{self.input.shape}, shape of weights{self.weights.shape}")
    #print (f"shape of biases{self.biases.shape}")

    self.linear_output = cp.dot(self.input, self.weights) + self.biases                                          
    self.output = self.activation_func(self.linear_output)   
    self.weights_regularisation = cp.sum(self.weights**self.L)              
    #print (f"shape of output{self.output.shape}")
    #print (f"shape of linear output{self.linear_output.shape}")

    return (self.output)
  
  def back_propergation (self, learning_rate:float, upstream_gradient:cp.array):
    self.deriv = self.back_prop(self.linear_output)
  
    self.delta = upstream_gradient * self.deriv 
    #print (f"{self.stage} self.linear output = {self.linear_output.shape}self.deriv = {self.deriv.shape} self.input.T. shape: {self.input.T.shape}, delta shape: {self.delta.shape}")
    dW_base = cp.dot(self.input.T, self.delta)
    if self.regularisation == None:
      dW_total = dW_base
    elif self.regularisation == "L1":
      dW_total = dW_base + self.lambda_param * 2 * cp.sign(sum(self.weights))
    elif self.regularisation == "L2":
      dW_total = dW_base + 2*(self.lambda_param*cp.sum(self.weights))
    onward_gradient = cp.dot(self.delta, self.weights.T)
    self.weights = self.weights-(learning_rate*dW_total)
    self.biases = self.biases-(learning_rate*cp.sum(self.delta, axis=0, keepdims=True))      
    return onward_gradient


class regressor:
  def __init__(self, n_layers:int, batch_size:int, layers_dict:list[dict:any], regularisation:str=None):
    self.layers_dict=layers_dict
    self.n_layers = n_layers
    
    self.batch_size = batch_size
    if regularisation == None:
      self.reg = self.std_loss
    elif regularisation == "L1" or "L2":
      self.reg = self.regularisation_loss

    
  def regularisation_loss(self):
    layer_weights_sum = 0
    for i in range (self.n_layers):
      layer_weights_sum += self.layers_dict[i]["layer_obj"].L
    return self.loss_gradient+(self.L_lambda*layer_weights_sum )
  def std_loss(self):
    return self.loss_gradient
  def loss(self, y:cp.array, y_pred:cp.array):
    y = cp.asarray(y)
    self.loss_gradient = 2*(y_pred-y)/self.batch_size
    batch_MSE = cp.mean((y_pred-y)**2)
    return self.reg()

class network:
  def __init__ (self, layers:list[dict:any], learning_rate:float=1e-7):
    self.layers = layers
    self.n_layers = len(self.layers)
    self.global_learning_rate = learning_rate
    #print (f"n_layers= {self.n_layers}")
  def forward_pass(self,input) ->cp.array:
    feed = [input]
    for i in range(self.n_layers):
      output = self.layers[i]["layer_obj"].forward(feed[-1])
      feed.append(output)
      #print (f"shape of layer output = {output.shape}")
    self.output = feed [-1]
    return self.output

  def back_pass(self, loss_gradient:cp.array, y_pred:cp.array, y:cp.array):
    for i in range (self.n_layers-1,-1,-1):
      #print (f"loss grad type{type(loss_gradient)}")
      loss_gradient = self.layers[i]["layer_obj"].back_propergation(upstream_gradient=loss_gradient, learning_rate=self.global_learning_rate)
      


class model:
  def __init__ (self, model:list[dict:any],
                input_features:int,
                learning_rate:float=1e-4,
                output_features:int = 1, 
                regularisation:str=None, 
                learning_rate_schedualing:str = "none", 
                biases_init:str="rand", Bvals:int=1, 
                Wvals:int=1,
                weights_init:str="rand",  
                lambda_param:float = 0.00001,
                batch_size:int = 1
                ):
    self.model = model
    self.batch_size= batch_size
    self.global_learning_rate = learning_rate
    
    for i, layer in enumerate (self.model):
      if i == 0:
        layer_m = input_features
        layer_n = layer["hidden_features"]
        stage = "input"
      elif i == len(model)-1:
          layer_m = layer["hidden_features"]
          layer_n = output_features
          ouput_type = layer.get("n_layer", "regressor")
          stage = "output"
      else:
        layer_m = layer["hidden_features"]
        layer_n = model[i-1]["hidden_features"]
        stage = "hidden"
      activation_function = layer["activation_function"]
      
      p = layer.get("n_layer", 1)
      #type = layer.get("type", None)
      for t in range(p):

        layer["layer_obj"] = network_layer (
                                    m=layer_m, n=layer_n, 
                                    regularisation=regularisation, 
                                    biases_init=biases_init,Bvals=Bvals, 
                                    Wvals=Wvals, weights_init=weights_init,
                                    activation=activation_function, lambda_param=lambda_param,
                                    stage=stage

                                    )
    if ouput_type == "regressor":
      self.loss_function = regressor(n_layers=len(self.model), batch_size=self.batch_size, layers_dict=model, regularisation=regularisation)
    self.network = network(layers=self.model, learning_rate=self.global_learning_rate)
    print (f"model: {self.model}")
  
  
  def train (self, x:cp.array, y:cp.array, epochs):
    rows,_=x.shape
    x = cp.asarray(x)
    y = cp.asarray(y)
    for epoch in range (epochs):
      for i in range(0, rows, self.batch_size):
        y_pred = self.network.forward_pass(input=x[i:i+(self.batch_size)])
        loss = self.loss_function.loss(y=y[i:i+(self.batch_size)], y_pred=y_pred)
        mse = self.network.back_pass (loss_gradient=loss, y_pred=y_pred,y=y[i:i+(self.batch_size)])



    
if __name__=="__main__":
  dict = [{"activation_function":"RelU", "hidden_features":12},
          {"activation_function":"RelU", "hidden_features":12},
          {"activation_function":"RelU", "hidden_features":1},
          ]

  nn =model(model = dict, input_features=8, output_features=1 )










  

