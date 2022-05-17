import random
import numpy

class Perceptron: 
  def __init__(self, training_dataframe):
    self.training_data = training_dataframe
    self.weights_vector = self._get_initial_weights_vector()

  def _get_initial_weights_vector(self):
    return [random.random() for _ in range(35)]

  def _calculate_prediction(self, input_vector):
    prediction = numpy.matmul(input_vector, self.weights_vector)
    if (prediction > 1): return 1
    if (prediction < 0): return 0
    return prediction

  def train(self, alpha: int = 1, iterations: int = 10):
    for _ in range(iterations):
      for goal_prediction, input_vector in self.training_data:
        prediction = self._calculate_prediction(input_vector)
        delta = prediction - goal_prediction 
        weight_delta = [delta * alpha * x for x in input_vector] 
        self.weights_vector = numpy.subtract(self.weights_vector, weight_delta) 

  def predict(self, input_vector):
    return self._calculate_prediction(input_vector)
