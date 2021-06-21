import pygame
import mnist_loader
import random
import numpy as np

class displayWindow(object):
    def __init__(self, S_WIDTH):
        self.BLUE = [0, 0, 255]
        self.RED = [255, 0, 0]
        self.S_WIDTH = S_WIDTH
        self.LINE_WIDTH = 2
        self.CIRCLE_WIDTH = 10
        self.win = pygame.display.set_mode((S_WIDTH, S_WIDTH))
        pygame.display.set_caption('visual neural network')
        self.THRESHHOLD = 20
        self.showCount = 0

    def drawNeuralNet(self, layers, weights, biases):
        for k in range(len(layers)-1):
            for i in range(layers[k]):
                for j in range(layers[k+1]):
                    if (layers[k]<self.THRESHHOLD or ((i>=(layers[k]/2-self.THRESHHOLD/2) and i<=(layers[k]/2+self.THRESHHOLD/2))) 
                    and (layers[k+1]<self.THRESHHOLD or ((j>=(layers[k+1]/2-self.THRESHHOLD/2) and j<=(layers[k+1]/2+self.THRESHHOLD/2))))):
                        x = weights[k][j-1][i-1]
                        r = 0
                        b = 0
                        if x>0:
                            if x>2: x=2
                            b = int((255*abs(x))/2)
                        else:
                            if x<-2: x = -2
                            r = int((255*abs(x))/2)
                        start = []
                        end = []
                        if layers[k]<self.THRESHHOLD:
                            start = [k*(self.S_WIDTH/len(layers))+.5*(self.S_WIDTH/len(layers)), 
                            (i)*(self.S_WIDTH/layers[k])+.5*(self.S_WIDTH/layers[k])]
                        elif (((i>=(layers[k]/2-self.THRESHHOLD/2) and i<(layers[k]/2+self.THRESHHOLD/2)))):
                            start = [k*(self.S_WIDTH/len(layers))+.5*(self.S_WIDTH/len(layers)), 
                            (i-(layers[k]/2-self.THRESHHOLD/2))*(self.S_WIDTH/(self.THRESHHOLD))+.5*(self.S_WIDTH/self.THRESHHOLD)]
                        if layers[k+1]<self.THRESHHOLD:
                            end = [(k+1)*(self.S_WIDTH/len(layers))+.5*(self.S_WIDTH/len(layers)), 
                            (j)*(self.S_WIDTH/layers[k+1])+.5*(self.S_WIDTH/layers[k+1])]
                        elif (((j>=(layers[k+1]/2-self.THRESHHOLD/2) and j<(layers[k+1]/2+self.THRESHHOLD/2)))):
                            end = [(k+1)*(self.S_WIDTH/len(layers))+.5*(self.S_WIDTH/len(layers)), 
                            (j-(layers[k+1]/2-self.THRESHHOLD/2))*(self.S_WIDTH/(self.THRESHHOLD))+.5*(self.S_WIDTH/self.THRESHHOLD)]
                        if (start != [] and end != []):
                            pygame.draw.line(self.win, [r, 0, b], 
                                start, end, width = self.LINE_WIDTH)
        for k in range(len(layers)):
            for i in range(layers[k]):
                r = 0
                b = 0
                g = 0
                if k>0:
                    x = biases[k-1][i]
                    if x>0:
                        if x>1: x=1
                        b = int((255*abs(x)))
                    else:
                        if x<-1: x = -1
                        r = int((255*abs(x)))
                else:
                    r = 100
                    b = 100
                    g = 200
                if (layers[k]<=self.THRESHHOLD):
                    pygame.draw.circle(self.win, [r, g, b], 
                        [k*(self.S_WIDTH/len(layers))+.5*(self.S_WIDTH/len(layers)), 
                        i*(self.S_WIDTH/layers[k])+.5*(self.S_WIDTH/layers[k])], self.CIRCLE_WIDTH)
                else:
                    if (((i>=(layers[k]/2-self.THRESHHOLD/2) and i<(layers[k]/2+self.THRESHHOLD/2)))):
                        pygame.draw.circle(self.win, [r, g, b], 
                            [k*(self.S_WIDTH/len(layers))+.5*(self.S_WIDTH/len(layers)), 
                            (i-(layers[k]/2-self.THRESHHOLD/2))*(self.S_WIDTH/(self.THRESHHOLD))+.5*(self.S_WIDTH/self.THRESHHOLD)], self.CIRCLE_WIDTH)
                    
        pygame.display.update()
    
    def drawExNum(self, ex, output, keep=False):
        IMAGE_SIZE = 2
        posy = ((self.showCount-self.showCount%5)/5)*(IMAGE_SIZE*28)
        posx = (self.showCount%5)*(IMAGE_SIZE*28*3)
        
        pygame.font.init()
        myfont = pygame.font.SysFont('freesansbold.ttf', IMAGE_SIZE*25)
        textsurface = myfont.render('-> '+str(output), False, (255, 255, 255))
        self.win.blit(textsurface,(posx+20+IMAGE_SIZE*28,posy))

        pygame.draw.rect(self.win, [255, 255, 255], (posx, posy, 28*IMAGE_SIZE, 28*IMAGE_SIZE))
        for i in range(28):
            for j in range(28):
                pygame.draw.rect(self.win, [int(ex[(28*i)+j][0]*255), int(ex[(28*i)+j][0]*255), int(ex[(28*i)+j][0]*255)], (posx+(IMAGE_SIZE*j), posy+(IMAGE_SIZE*i), IMAGE_SIZE, IMAGE_SIZE))
        pygame.display.update()
        self.showCount+=1
        if keep:
            run = True
            while run:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
