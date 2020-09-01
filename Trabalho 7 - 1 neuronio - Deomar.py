#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#Função de ativação sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))
#Derivada da função de ativação
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

#Criação da rede neural
class NeuralNet: #Cria classe da rede neural
    def __init__(self,x,y): #Variáveis de inicialização
        self.data = x #Entradas
        self.w = np.random.random(2) #Pesos w
        self.b = np.random.random(1) #Peso b
        self.y_train = y #Treino
    def feedforward(self): #Método feedforward
        self.z = np.dot(self.w, self.data.T) + self.b #Variável z de saída
        self.output = sigmoid(self.z) #Saída
    def backpropagation(self, taxa): #Método backpropagation
        self.d_w = np.dot(2*(self.y_train - self.output)*sigmoid_derivative(self.data), self.data.T) #Derivada em relação a w
        self.d_b = 2*(self.y_train - self.output)*sigmoid_derivative(self.b) #Derivada em relação a b
        self.w += taxa*self.d_w #Atualização dos pesos w
        self.b += taxa*self.d_b #Atualização do pesos b
    def loss(self): #Função erro/perda
        return np.power(self.y_train - self.output, 2) #Retorna o erro
    def saida(self): #Saída
        return self.output

#Entrada do dataset e instanciação
x = np.array([0,1]) #Dados de treino
y = np.array([1]) #Saídas desejadas do treino
taxa = 0.01 #Taxa de aprendizagem
epochs = 5000 #Iterações
neural = NeuralNet(x,y) #Instancia objeto da rede

#Treinamento da rede
aux = [] #Lista auxiliar
y_saida = [] #Lista de saída
for i in range(epochs): #Loop de iterações
    neural.feedforward() #Executa feedforward
    neural.backpropagation(taxa) #Executa backpropagation
    aux.append(neural.loss()) #Guarda função perda
    y_saida.append(neural.saida()) #Guarda saída

#Plot do erro
plt.xlabel('Epochs') #Nome eixo x
plt.ylabel('Função erro') #Nome eixo y
plt.plot(aux) #Plota da função erro com epochs
#plt.savefig("Erro_1_neuronio.png", dpi=150) #Salva figura

