#!/usr/bin/env python

import tkinter as tk
import numpy as np
import csv



def sigmoid(value):
    return 1.0 / ( 1.0+np.exp(-value))

def sigmoid_prime(value):
    return sigmoid(value) * (1 - sigmoid(value))

def softmax(value):
    return np.exp(value) / (np.sum(np.exp(value), axis=0))




with open('W1.csv') as csvfile:
    rows = csv.reader(csvfile)
    W1_orig = list(zip(*rows))
with open('W2.csv') as csvfile:
    rows = csv.reader(csvfile)
    W2_orig = list(zip(*rows))
with open('b1.csv') as csvfile:
    rows = csv.reader(csvfile)
    b1_orig = list(zip(*rows))
with open('b2.csv') as csvfile:
    rows = csv.reader(csvfile)
    b2_orig = list(zip(*rows))

W1 = np.array(W1_orig).T
W2 = np.array(W2_orig).T
b1 = np.array(b1_orig).T
b2 = np.array(b2_orig).T

W1 = [[float(j) for j in i] for i in W1]
W2 = [[float(j) for j in i] for i in W2]
b1 = [[float(j) for j in i] for i in b1]
b2 = [[float(j) for j in i] for i in b2]






class Drawer(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=280, height=280, bg = "black", cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.button_print = tk.Button(self, text = "Submit Number", command = self.submit_number)
        self.button_print.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text="redraw", command=self.draw_example)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)

    def clear_all(self):
        self.canvas.delete("all")
        self.points_recorded = []

    def submit_number(self):     #####Make this better.
        out = []
        for x in self.points_recorded:
            out.append(x//10)
        formated = [0]*784
        for y in range(0,len(out),2):
            formated[(out[y+1]-1)*28+out[y]] = 255
            formated[(out[y + 1] - 2) * 28 + out[y]] = 255
            formated[(out[y + 1] - 0) * 28 + out[y]] = 255
            formated[(out[y  +1] - 1) * 28 + out[y]+1] = 255
            formated[(out[y  +1] - 1) * 28 + out[y] -1] = 255
        self.draw_predict(formated)


    def draw_example(self):
        out = []
        for x in self.points_recorded:
            out.append(x//10)
        for x in range (0,len(out)-3,2):
            self.canvas.create_line(out[x], out[x+1], out[x+2], out[x+3], fill="yellow")

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y


    def draw_from_where_you_are(self, event):
        self.x = event.x
        self.y = event.y
        self.canvas.create_line(self.previous_x, self.previous_y,self.x, self.y,fill="yellow",width=3)
        self.points_recorded.append(self.x)
        self.points_recorded.append(self.y)
        self.previous_x = self.x
        self.previous_y = self.y

    def draw_predict(self, input):

        Z1 = np.matmul(W1, input) + b1
        A1 = sigmoid(Z1)
        Z2 = np.matmul(W2, A1) + b2
        A2 = softmax(Z2)
        prediction = np.argmax(A2, axis=0)[0]

        probability = A2[prediction] * 100
        probability = probability.astype(float)[0]

        print(prediction)
        print(probability)


root = Drawer()
root.mainloop()