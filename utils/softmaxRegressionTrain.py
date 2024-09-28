import numpy as np
import matplotlib.pyplot as plt

def forecastFunction(theta,x):
    m=np.shape(x)[0]
    n=np.shape(x)[1]
    k=np.shape(theta)[0]
    a=np.exp((x*(theta.T)))
    sum=a*np.mat(np.ones((k,1)))
    for i in range(m):
        a[i]=a[i]/sum[i]
    return a

def errCost(fc,y):
    m=np.shape(fc)[0]
    sumCost=0.0
    for i in range(m):
        if (fc[i,y[i]])>0:
            sumCost+=np.log(fc[i,y[i]])
        else:
            sumCost+=0
    return sumCost/m

def loopGradientDescent(alpha,x,y,k,cnt):
    m=np.shape(x)[0]
    n=np.shape(x)[1]
    theta=np.mat(np.ones((k,n)))
    sumCost_list = []
    for i in range(cnt):
        fc=forecastFunction(theta,x)
        sumCost=errCost(fc,y)
        fc=(-1)*fc
        for j in range(m):
            fc[j,y[j]]+=1
        theta=theta+(alpha/m)*(((x.T)*fc).T)
        if i%200==0:
            print(+str(i)+":")
            print("ERROR:"+str(sumCost))
            print("theta:")
            print(theta)
            sumCost_l = abs(sumCost.tolist()[0][0])
            sumCost_list.append(sumCost_l)
    np.savetxt("%sumCost.csv" , sumCost_list, delimiter=",")

    Iterations = list(range(1,cnt+1))
    data_dict = {}
    for i, j in zip(Iterations, sumCost_list):
        data_dict[i] = j
    plt.title("loss function")
    plt.xlabel("iteration times / 200")
    plt.ylabel("loss value")
    x = [i for i in data_dict.keys()]
    y = [i for i in data_dict.values()]
    plt.plot(x, y, label="loss")
    plt.legend()
    plt.show()

    return theta



def loadData(fileName):
    f=open(fileName)
    x=[]
    y=[]
    for line in f.readlines():
        tmpX=[]
        tmpX.append(1)
        lines=line.strip().split("\t")
        for i in range(len(lines)-1):
            tmpX.append(float(lines[i]))
        y.append(int(float(lines[-1])))
        x.append(tmpX)
    f.close()
    return np.mat(x),np.mat(y).T,len(set(y))

def saveModel(fileName,theta):
    f=open(fileName,"w")
    m=np.shape(theta)[0]
    k=np.shape(theta)[1]
    for i in range(m):
        tmpT=[]
        for j in range(k):
            tmpT.append(str(theta[i,j]))
        f.write("\t".join(tmpT)+"\n")
    f.close()

if __name__=='__main__':
    x,y,k=loadData("traindata_and_testdata")
    theta=loopGradientDescent(0.04,x,y,k,10000)
    saveModel("modelData",theta)
    
