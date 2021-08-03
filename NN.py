import numpy as np
import random
import helper as h

class Net:
    
    def __init__(self):
        
        self.weights=[] #list of weight matrices in order from left to right
        self.biases=[]
        
        #default parameters values
        self.epochs=100
        self.bsize=2
        self.eta=1
        self.acti='sig'
        self.metrics=True
        #======================
    
    def printNet(self):
        print ("Weights:",self.weights)
        print("Biases:",self.biases)

    def saveNet(self,f):
        file = open(f, "w")
        file.write('==============Weights and biases saved from NN.py============'+"\n\n")

        file.write("Weights"+"\n")
        for w in self.weights:
            file.write("==="+"\n")
            ws=w.shape
            for i in range(ws[0]):
                for j in range(ws[1]):
                    file.write(str(w[i][j])+" ")
                file.write("\n")
            

        file.write("Biases"+"\n")
        for b in self.biases:
            file.write("==="+"\n")
            bs=b.shape
            for i in range(bs[0]):
                for j in range(bs[1]):
                    file.write(str(b[i][j])+" ")
                file.write("\n")
            

        file.close()

    def resumeNet(self,f):
       
        with open(f, 'r') as file:
            file.readline()
            file.readline()
            file.readline()
            file.readline()
            i=0
            j=0
            k=0
            wbtrigg=0

            for line in file:
                #print(line)
                k=0
                if (line=="\n"):
                    break
                if (line=="===\n"):
                    i+=1
                    j=0
                    k=0
                    continue
                if (line=="Biases\n"):    
                    i=-1
                    j=0
                    k=0
                    wbtrigg=1
                    continue

                for val in line.strip().split():
                    
                    if (wbtrigg==0):
                        self.weights[i][j][k]=float(val)
                        k+=1
                    if (wbtrigg==1):
                        self.biases[i][j][k]=float(val)
                        k+=1
                j+=1
                        

        file.close()

    def addLayer(self,size):
        sweights=np.random.uniform(low=-1,high=1,size=(size[1],size[0]))
        sbiases= np.random.uniform(low=-1,high=1,size=(size[1],1))
        self.weights.append(sweights)
        self.biases.append(sbiases)
        self.wnum=len(self.weights) #total number of weight matrices
        #print (self.wieghts)

    def setParam(self,epochs=200,bsize=2,eta=1,acti='sig',metrics=True):
        # eta=learning rate, bsize = minibatch size
        self.epochs=epochs
        self.bsize=bsize
        self.eta=eta
        self.acti=acti
        self.metrics=metrics

    def updateBch(self,bch):
        #bch = mini batch
        for (x,y) in bch:
           
            err=0
            gradW=[np.zeros(w.shape) for w in self.weights] #init zero weight matrices 
            gradB=[np.zeros(b.shape) for b in self.biases]
            lz,la=self.feedF(x)
            pred=la[-1] #predicted output
            dif=pred-y
            err+=np.mean(dif**2)
            dgradW,dgradB=self.backP(x,y,lz,la)
            gradW = [nw+dnw for nw, dnw in zip(gradW, dgradW)]
            gradB = [nb+dnb for nb, dnb in zip(gradB, dgradB)]
            #grad+=self.backP(x,y,lz,la) #summing up gradients of weight matrices from all elmnts of batch
            #print(grad)

        for i in range(self.wnum):  #updating weights & biases (no of weights = no of biases)
            self.weights[i]-=(self.eta/self.bsize)*gradW[i]
            self.biases[i]-=(self.eta/self.bsize)*gradB[i]

        error=err/(2*self.bsize)
        return error

    def backP(self,x,y,lz,la):  #Backpropogate
        #x,y,lz= input layer, output layer, feedforwarded layers list
        lnum=len(la) # number of layers
        
        gradW=[np.zeros(w.shape) for w in self.weights] #init zero list of gradient weights
        gradB=[np.zeros(b.shape) for b in self.biases]
        pred=la[-1] #predicted output
        
        dif=pred-y
        delta=dif*h.dsig(lz[-1])
        
        gradW[-1]=np.dot(delta,la[-2].T)
        gradB[-1]=delta  #backprop of just output layer
        
        
        for n in range (2,lnum):  
            delta= np.dot(self.weights[-n+1].T, delta) * h.dsig(lz[-n])
            #print(la[-n-1],delta)
            gradB[-n]=delta
            gradW[-n]= np.dot(delta,la[-n-1].T)
        return gradW,gradB
        
    def feedF(self,x):  #Feedforward
        #x= input layer
        
        lz=[]   #list of neuron layers before activations excluding input layer
        la=[x]   #list of activated neuron layers including input layer
        for j in range (self.wnum): #feedforward

            """ for regression (output without sigmoid)
            if (j!=self.wnum-1):
                z=np.dot(x,self.weights[j])
                x=h.sig(z)     
            else:
                x=np.dot(x,self.weights[j])
                z=x
            """
            z=np.dot(self.weights[j],x)+self.biases[j]
            
            x=h.sig(z)

            lz.append(z)
            la.append(x)

        return lz,la

    def evaluate(self,ipl):
        laArr=[]
        ipl=[np.reshape(ipl[i],(len(ipl[i]),1)) for i in range(len(ipl))]
        for l in ipl:
            lz,la=self.feedF(l)
            laArr.append(la[-1])
        print(laArr)

    def trainNet(self,ipl,opl):
        # ipl,opl = input and output layers as np arrays
        tnum=np.shape(opl)[0]  # num of training samples
        ipl=[np.reshape(ipl[i],(len(ipl[i]),1)) for i in range(len(ipl))]
        opl=[np.reshape(opl[i],(len(opl[i]),1)) for i in range(len(opl))]
        
        train_data=[]
        for i in range(tnum):
            iol=(ipl[i],opl[i])
            train_data.append(iol)
        
        for j in range (self.epochs):
            random.shuffle(train_data)
            
            batches = [train_data[k:k+self.bsize] for k in range(0, tnum, self.bsize)]
            
            error=0
            for bch in batches:
                error+=self.updateBch(bch)
                    
            error=error/self.bsize
            if (self.metrics):
                print ("Epoch {0} of".format(j),self.epochs," : Error=",error)
            
