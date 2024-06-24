#!/usr/bin/env python
# coding: utf-8

# In[9]:


from matplotlib.pyplot import scatter,show,title,xlabel,ylabel,xlim,plot,yscale,legend,hist,ylim
from numpy import *
from scipy.optimize import curve_fit
from scipy.stats import norm,poisson


# In[10]:


#Definimos una función que nos calcule el hamiltoniano H=J_{z_1}+J_{z_2}+\lambda/J(J_{x_1}J_{x_2})

def matriz(J,lamb):
    #Tenemos una matriz H=H0+W donde W=Jx1*Jx2 que va a ser nuestra perturbación. Calculamos H0 y W por separado
    dim=int((2*J+1)**2)
    
    
    #Definimos los vectores m1 y m2 que van de J a -J
    mj=[];M=[]; m=J
    
    mj=arange(-J,J+1)
    dimM=len(mj)
    
    #Calculamos el vector M que irá en la diagonal
    
    for j in range(dimM):
        for k in range(dimM):
            M.append(mj[k]+mj[j])
        
    #Calculamos H0
    
    H0=zeros([dim,dim])
    
    for i in range(dim):
        H0[i,i]=M[i]
        
        
        
    #Calculamos los vectores J+ y J-
    Jmas=[];Jmenos=[]
    
    for i in range(dimM):
        Jmas.append(sqrt(J*(J+1)-mj[i]*(mj[i]+1)))
        Jmenos.append(sqrt(J*(J+1)-mj[i]*(mj[i]-1)))
    
    #El término perturbativo estará compuesto de 4 matrices: W1, W2, W3 y W4
    W=zeros([dim,dim]); W1=[]; W2=[]; W3=[]; W4=[]
    
    for i in range(dimM):
        for j in range(dimM):
            for k in range(dimM):
                for l in range(dimM):
                    if (mj[i]==mj[k]+1) and (mj[j]==mj[l]+1):
                        W1.append(Jmas[k]*Jmas[l])
                        W2.append(0)
                        W3.append(0)
                        W4.append(0)
                        
                    elif (mj[i]==mj[k]-1) and (mj[j]==mj[l]+1):
                        W2.append(Jmenos[k]*Jmas[l])
                        W1.append(0)
                        W3.append(0)
                        W4.append(0)
                        
                    elif (mj[i]==mj[k]+1) and (mj[j]==mj[l]-1):
                        W3.append(Jmas[k]*Jmenos[l])
                        W1.append(0)
                        W2.append(0)
                        W4.append(0)
                            
                    elif (mj[i]==mj[k]-1) and (mj[j]==mj[l]-1):
                        W4.append(Jmenos[k]*Jmenos[l])
                        W1.append(0)
                        W2.append(0)
                        W3.append(0)
                            
                    else:
                        W1.append(0)
                        W2.append(0)
                        W3.append(0)
                        W4.append(0)

    k=0
    for i in range(dim):
        for j in range(dim):
            W[i,j]=W1[k]+W2[k]+W3[k]+W4[k]
            k+=1
            
    return H0+lamb*W/(4*J),H0


# In[11]:


#Definimos una función que calcule el valor esperado del observable en la base del hamiltoniano

def valoresperado(H0,H):
    
    autovalores, autovectores = linalg.eig(H) #Función que calcula los autovalores y autovectores
    
    X=0; Y=0; O=[];
    
    for i in range(len(autovectores)):
        #Empezamos con el primer autovector normalizado
        autovector_norm=autovectores[:,i]/linalg.norm(autovectores[:,i])
        
        for j in range(len(H0)):
            for k in range(len(H0)):
                X+=autovector_norm[k]*H0[k,j]
            
            Y+=X*autovector_norm[j]
            X=0
        O.append(Y)
        Y=0
    return autovalores, O


# In[12]:


#calculamos la matriz Jz1Jz2

def observable2(J):
    #Tenemos una matriz H=H0+W donde W=Jz1*Jz2 que va a ser nuestra perturbación. Calculamos H0 y W por separado
    dim=int((2*J+1)**2)
    
    
    #Definimos los vectores m1 y m2 que van de J a -J
    mj=[];M=[]; m=J
    
    mj=arange(-J,J+1)
    dimM=len(mj)
    
    #Calculamos el vector M que irá en la diagonal
    
    for j in range(dimM):
        for k in range(dimM):
            M.append(mj[k]*mj[j])
        
    
    O2=zeros([dim,dim])
    
    for i in range(dim):
        O2[i,i]=M[i]
    return O2


# In[13]:


#Ejemplo de como calcular el valor esperado y autovalores del hamiltoniano para un J y lambda datos

H,H0=matriz(40,5)
O2=observable2(30)

autovalores,O=valoresperado(O2,H)

K=zeros([len(autovalores),2])
    
for i in range(len(autovalores)):
    K[i,0]=autovalores[i]
    K[i,1]=O[i]
        
savetxt('datosJz1Jz2505.txt', K)


# In[ ]:


#Representamos el valor esperado en función de los autovalores E_i, lo ajustamos a una recta, representamos el ruido y su histograma

datos_ordenados3 = sorted(loadtxt("datosJz1Jz2405.txt"), key=lambda x: x[0]) #key nos deja ordenar según una función. 
#Con Lambda podemos definir dicha función. 
from scipy.optimize import curve_fit

#A continuación ajustamos los datos de [0,50] a una recta

#Primero tenemos que ordenar los datos

# Ordenar la lista de datos en función de las energías (columna 0)

datos_ordenados2 = sorted(loadtxt("datosJz1Jz2255.txt"), key=lambda x: x[0]) #key nos deja ordenar según una función. 
#Con Lambda podemos definir dicha función. 

P=array(datos_ordenados2)#La funcion de antes te devuelve una lista y dentro el array. Usamos esta funcion para que nos devuelva directamente el array

#Lo guardamos en un fichero de texto para agilizar trámites

savetxt('datosordenados255.txt', P)

#Ahora hacemos el ajuste. Para ello vamos a usar la función curve_fit. Definimos primero la función de una recta

def f(x,a,b):
    return a*x+b

#Cogemos los datos que nos interesan


fit,error=curve_fit(f,P[1350:2000,0],P[1350:2000,1]) #Cogemos los datos que nos interesan. En este caso, los datos en el intervalo |E_i|<50

#fit nos va a devolver el mejor valor para a y b, mientras que error nos va a devolver el error que se comete

a,b=fit

#Representamos la curva 
plot(P[1350:2000,0],f(P[1350:2000,0],a,b),label="a*x+b",c="red")
scatter(P[:,0],P[:,1],0.5)
xlabel("$E_i$")
ylabel(r'$\langle E_i| J_{z_1}\cdot J_{z_2} |E_i\rangle$')
#title("$\lambda=5$, J=25")
legend(["Ajuste lineal"])
title("J=25 $\lambda=5$")
show()

#Ahora restamos y-y_fit de forma que

y_nuevo=P[1350:2000,1]-f(P[1350:2000,0],a,b)

plot(P[1350:2000,0],y_nuevo,linewidth=0.5)
xlabel("$E_i$")
ylabel("$\~y=y-y_{fit}$")

show()

n=len(y_nuevo)
x = linspace(-110, 110, n)

mu, std = norm.fit(y_nuevo)
#Calcular la PDF (función de densidad de probabilidad) de la distribución ajustada
p = norm.pdf(x, mu, std)

# Graficar la curva de la distribución ajustada
plot(x, p, 'm', linewidth=2)
legend(["$P(\~y)$"])

hist(y_nuevo,density=True, bins=100)

print(mu,std**2)
show()



# In[ ]:


#Ahora calculamos si hay degeneración.

x=[];y=[]
for i in range(len(L)-2):
    i+=1
    x.append(L[i,0])
    y.append(min(L[i+1,0]-L[i,0],L[i,0]-L[i-1,0])) #Cogemos el mínimo entre la diferencia E_{i+1}-E_i y E_i-E_{i-1}


scatter(x,y,0.5)
yscale('log')
xlabel("$E_i$")
ylabel("logaritmo de min $(E_{i+1}-E_{i},E_i-E_{i-1})$")
show()
scatter(x,y,0.5)
xlabel("$E_i$")
ylabel("min $(E_{i+1}-E_{i},E_i-E_{i-1})$")

