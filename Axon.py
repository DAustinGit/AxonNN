
#import numpy as np
import cupy as cp
rng = cp.random.default_rng()
max_itterations = 400
class network_layer:
  def __init__(self, parent_instance, m:int, n:int, activation:str = "ReLu", stage:str="hidden"):
    self.parent_model = parent_instance 
    rng = cp.random.default_rng()
    self.m = m
    self.n = n
    self.stage = stage
    if stage == "output":
      self.biases_n = self.parent_model.output_features
    else:
      self.biases_n = self.n

    if self.parent_model.biases_init == "rand":
      self.biases=rng.uniform(-0.1,0.1, size =(1, self.biases_n))
    if self.parent_model.biases_init == "fill":
      self.biases=cp.full(self.parent_model.Bvals, size =(1, self.biases_n))
    if self.parent_model.weights_init == "rand":
      self.weights=rng.standard_normal(size=(self.m, self.n)) * cp.sqrt(2.0 / self.m)
    if self.parent_model.weights_init =="fill":
      self.weights = cp.full(self.parent_model.Wvals, size =(self.m, self.n))
    self.output = cp.array([[]])
    self.input = cp.array([[]])
    self.activation = activation
    self.activation_functions = {
      "RelU":{"forwards":self.reLu_forwards, "back_propergation":self.relu_backpropergation},
      "tanh":{"forwards":self.tanh_forwards, "back_propergation":self.tanh_backpropergation},
      "linear":{"forwards":self.linear_forwards, "back_propergation":self.linear_backpropergation}
    }
    
    self.activation_func = self.activation_functions[self.activation]["forwards"]
    self.back_prop = self.activation_functions[self.activation]["back_propergation"]
    print (f"stage {stage} activation = {self.activation_func}, back = {self.back_prop }")
    if self.parent_model.regularisation == "L1":
      self.L = 1
      self.bp_type = self.L1_back_prop
    elif self.parent_model.regularisation =="L2":
      self.L = 2
      self.bp_type = self.L2_back_prop
    else:
      self.L = 0
      self.bp_type = self.std_back_prop
  def reLu_forwards(self,)->cp.ndarray:
    A = cp.maximum(0, self.linear_output)
    return  A
  def relu_backpropergation(self)->cp.ndarray:
    return cp.where(self.linear_output > 0, 1, 0)
  def tanh_forwards(self)->cp.ndarray:
    return cp.tanh(self.linear_output)
  def tanh_backpropergation(self)->cp.ndarray:
    return  1 - cp.power(self.linear_output, 2)
  def linear_forwards(self)->cp.ndarray:
    return self.linear_output
  def linear_backpropergation(self)->cp.ndarray:
    return  cp.ones_like(self.linear_output)




  
  def forward (self, input):
    self.input = cp.array(input)
    self.linear_output = cp.dot(self.input, self.weights) + self.biases                                                        
    return self.activation_func()
  
  def L2_back_prop(self):
    return self.dW_base + 2*(self.parent_model.lambda_param*self.weights)
  def L1_back_prop(self):
    return self.dW_base + self.parent_model.lambda_param*cp.sign(self.weights)
  def std_back_prop(self):
    return self.dW_base
  
  def back_propergation (self, upstream_gradient:cp.ndarray)->cp.ndarray:
    self.deriv = self.back_prop()
    self.delta = upstream_gradient * self.deriv 
    self.dW_base = cp.dot(self.input.T, self.delta)
    dW_total = self.bp_type()
    self.weights = self.weights-(self.parent_model.learning_rate*dW_total)
    self.biases = self.biases-(self.parent_model.learning_rate*cp.sum(self.delta, axis=0, keepdims=True))      
    return cp.dot(self.delta, self.weights.T)


class regressor:
  def __init__(self, parent_instance):
      self.parent_model = parent_instance
  def loss(self, y:cp.ndarray, y_pred:cp.ndarray):
    y = cp.asarray(y)
    self.loss_gradient = 2*(y_pred-y)/self.parent_model.batch_size
    self.parent_model.sum_abselute_error += cp.sum((y_pred-y)**2)
    return self.loss_gradient

class network:
  def __init__(self, parent_instance):
    self.parent_model = parent_instance
  

  
  def forward_pass(self,input) ->cp.ndarray:
    feed = [input]
    for i in range(self.parent_model.n_layers):
      feed.append(self.parent_model.model[i]["layer_obj"].forward(feed[-1]))
      #output = self.parent_model.model[i]["layer_obj"].forward(feed[-1])
      #feed.append(output)     #print (f"shape of layer output = {output.shape}")
    return feed [-1]

  def back_pass(self, loss_gradient:cp.ndarray):
    for i in range (self.parent_model.n_layers-1,-1,-1):
      #print (f"loss grad type{type(loss_gradient)}")
      loss_gradient = self.parent_model.model[i]["layer_obj"].back_propergation(upstream_gradient=loss_gradient)
      


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
    self.input_features = input_features
    self.learning_rate = learning_rate
    self.output_features = output_features
    self.regularisation = regularisation
    self.learning_rate_schedualing = learning_rate_schedualing
    self.biases_init = biases_init
    self.weights_init = weights_init
    self.Wvals = Wvals
    self.weights_init = weights_init
    self.lambda_param = cp.array(lambda_param)
    self.batch_size = batch_size
    self.n_layers = len(self.model)
    self.analytics={}
    self.analytics["MSE"]=[]

    
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

        layer["layer_obj"] = network_layer (self,
                                    m=layer_m, n=layer_n, 
                                    activation=activation_function, 
                                    stage=stage
                                    )
    if ouput_type == "regressor":
      self.loss_function = regressor(self)
    self.network = network(self)
    print (f"model: {self.model}")
  
  
  def train (self, x:cp.array, y:cp.array, epochs):
    rows,_=x.shape
    x = cp.asarray(x)
    y = cp.asarray(y)
    for epoch in range (epochs):
      self.sum_abselute_error = 0
      for i in range(0, rows, self.batch_size):
        y_pred = self.network.forward_pass(input=x[i:i+(self.batch_size)])
        loss = self.loss_function.loss(y=y[i:i+(self.batch_size)], y_pred=y_pred)
        self.network.back_pass (loss_gradient=loss)
      self.analytics["MSE"].append(float(self.sum_abselute_error/rows))

  def predict(self, x):
    feed_out = cp.asarray(x)
    for i in range(self.n_layers):
      feed_out = self.model[i]["layer_obj"].forward(feed_out)
    return feed_out   
  
  def MSE (self):
    return (self.analytics["MSE"])

    
if __name__=="__main__":
  dict = [{"activation_function":"RelU", "hidden_features":12},
          {"activation_function":"RelU", "hidden_features":12},
          {"activation_function":"RelU", "hidden_features":1},
          ]

  nn =model(model = dict, input_features=8, output_features=1 )









  

