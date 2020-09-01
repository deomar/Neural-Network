#!/usr/bin/env python
# coding: utf-8

#Bibliotecas utilizadas
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#Função de ativação sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))
#Derivada da função de ativação
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

#Implementação da rede neural 4 neurônios e 2 camadas
class NeuralNet: #Classe da rede
    def __init__(self,x,y): #Variáveis de inicialização (pesos e construção da saída)
        self.data = x #Entrada
        self.w = np.random.randn(2,3) #Pesos da camada 1
        self.u = np.random.randn(3,1) #Pesos da camada 2 (última camada)
        self.y_train = y #Saída
        self.output = np.zeros(self.y_train.shape) #Começa com saída zerada
    def feedforward(self): #Cálculo das saídas dos neurônios
        self.h = sigmoid(np.dot(self.data, self.w)) #Saídas dos neurônios da camada 1
        self.output = sigmoid(np.dot(self.h, self.u)) #Saída da rede
    def backpropagation(self, taxa): #Cálculo das derivadas parciais que minimizam o erro
        self.dldsy = 2*(self.y_train - self.output)*sigmoid_derivative(self.output) #Derivada que aparece em todas as camadas
        self.du = np.dot(self.h.T, self.dldsy) #Derivada da função erro em função dos pesos da segunda camada
        self.dw = np.dot(self.data.T, (np.dot(self.dldsy, self.u.T)*sigmoid_derivative(self.h))) #Derivada da função erro
        #em função do pesos da camada 1
        self.w += taxa*self.dw #Passo que minimiza os pesos da camada 1
        self.u += taxa*self.du #Passo que minimiza os pesos da camada 2
    def loss(self): #Cálculo do erro (MSE)
        return np.power(self.y_train - self.output, 2).sum() #Soma para cada entrada
    def saida(self): #Saída final do conjunto treinado
        return self.output

#Entrada do dataset e instanciação
x = np.array([[0,0],[0,1],[1,0],[1,1]]) #Entrada da porta XOR
y = np.array([[0],[1],[1],[0]]) #Saídas da porta XOR
taxa = 0.1 #Taxa de aprendizagem
epochs = 40000 #Número de iterações
neural = NeuralNet(x,y) #Instancia o objeto da rede neural
erro = [] #Lista vazio para os erros

#Treinamento da rede
for i in range(epochs): #Faz loop de iterações
    neural.feedforward() #Executa o método feedforward
    neural.backpropagation(taxa) #Executa o método backpropagation
    erro.append(neural.loss()) #Guarda o erro
print(neural.saida()) #Printa saídas calculadas pela rede

#Plot do erro
plt.xlabel('Epochs') #Nome eixo x
plt.ylabel('Função erro') #Nome eixo y
plt.plot(erro) #Plota o erro em função dos epochs
#plt.savefig("Erro_4_neuronios_XOR.png", dpi=150) #Salva figura

