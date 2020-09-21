#!/usr/bin/env python
# coding: utf-8

#  
#                                  
#                                              @Zamith@
#                             
#                                        
#                                        
# Esse trabalho foi desenvolvido utilizando como ferramenta para análise dos dados a linguagem de programacao PYTHON pelas diversas caracteristicas positivas relacionas
# a Estatistica e exploração de dados
#                                        
#                           

#                                             PARTE I
#                                       
#                                     APRESENTAÇÃO DOS DADOS

# In[30]:


import pandas as pd # importando Data Frame onde vou utilizar meu aquivo .txt
import matplotlib.pyplot as plt # lib importantíssima para plotagem dos 
 #gráficos da atividade
import numpy as np  # lib matemática do python
import statsmodels.api as sm # lib estatistica
from scipy import stats # outra poderosa lib estatistica

get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv("/Users/thiagozamith/Desktop/dados2.txt",index_col=0) # amostrando 
                                                                     #os dados do arquivo
df.head()


# In[31]:


df.isnull().values.any()
# Verificando tem algum dado faltando ou dado como "NaN" 
#, retorna False,sem dados faltando...


# In[32]:


df_=df.iloc[1:4321,[9,10]] 

# gerando um Dataframe  com duas variaveis de nossa escolha, 6 meses amostragem 
# 30 dias x 24h x 6 meses = 4320h
df_.head()
     


#                                           PARTE II 
# 
#                       INVESTIGACAO E INTERPRETAÇÃO ESTATÍSTICA DOS DADOS

#                                 Gráfico Temporal Inicial

# In[33]:


df_.plot(label="Grafico Temporal",logy=True,figsize=(30,20),fontsize=15)

                                Gráfico Temporal - Filtro 01
                            
Comentários:
Primeiramente faremos uma primeira filtragem de dados,excluindo dos dados valores absurdos do ponto de vista físico.
# In[34]:


#FILTRO INICIAL= CONDICAO T> -273.15C E Ur < 0 , ABSURDOS
df_=df_[np.logical_and(df_["Dewp"] > -273.15 , df_["Humi"] > 0)] 
# Nao existe Temperatura menor que o Zero Absoluto e Umidade Relativa negativa.


# In[35]:


df_.plot(figsize=(50,20),fontsize=24) 
# acerca dessa plotagem, podemos reparar a presenca de outliers, 
#isto é dados que fogem d normalidade

                                          GRAFICO HISTOGRAMA
                                    
                                              
                                 
n=numero de amostras >>> 30 ,considerando a distribuicao normal entao podemos normalizar a curva e atribuindo um filtro para nosso desvio padrão , a fim de eliminar os outliers observados no grafico anterior.
# In[36]:


df_.hist() # Histograma dessas amostras em 6 meses
#Observe que a presenca que mesmo com valores que 
#fogem da normalidade a Var "Humi" ja apresenta caracteristica normal
#diferente da Temperatura do ponto.de.orvalho...


#                                   GRÁFICO 'BOXPLOT'
#                                     
#                                   
# 
# 
# 

# In[37]:


df_.plot(kind="box",logy=False,figsize=(15,10),subplots=True) # BOX PLOT GRAPHIC'S 


#                               Variaveis Estatísticas da Amostra
#                       
#                               Media,Moda,Mediana e Desvio Padrão
#                       
#                                                 

# In[38]:


media=df_.mean() #media
media=pd.DataFrame(media)
print(media)


# In[39]:


moda=df_.mode() #moda
moda=moda.transpose()
print(moda)


# In[40]:


mediana=df_.median() #mediana
mediana


# In[41]:


d_p=df_.std() #desvio padrao
d_p


# In[42]:


#Filtrando Outliers
# Regra Empirica " 68–95–99.7 " ---> 

std_dev = 3
df_ = df_[(np.abs(stats.zscore(df_)) < float(std_dev)).all(axis=1)] #


# In[43]:


df_.plot(figsize=(35,12),fontsize=15) #amostrando meus dados sem a presença de Outliers


# In[44]:


df_.plot(kind="box",logy=False,figsize=(15,10),subplots=True)


# In[45]:


media=df_.mean() #media
media=pd.DataFrame(media)
print(media)


# In[46]:


moda=df_.mode() # moda
moda=pd.DataFrame(moda)
moda=moda.transpose()
print(moda)


# In[47]:


d_p=df_.std() #desvio padrao
d_p


# In[48]:


mediana=df_.median() #mediana
mediana


# In[49]:


df_.hist() 
# observe que uma vez extraido os outliers as variaveis apresentam distribuição normal


#                              GRÁFICO DE DISPERSÃO
#                                 
#                                                

# In[50]:




plt.scatter(df_["Dewp"] , df_["Humi"],color="blue")
plt.title('Dispersao') 
plt.xlabel('Ponto de Orvalho') 
plt.ylabel('Umidade Relativa') 
plt.show()


#                                 REGRESSÃO LINEAR
#                                 
#                                               

# In[52]:


from sklearn.linear_model import LinearRegression  #Linear Regresion
      #importando as libs estatisticas  necessarias 
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


# In[53]:


x=df_[["Dewp"]]
x=np.array(x)
y=df_[["Humi"]]
y=np.array(y)


# In[54]:


# criar modelo linear e otimizar
lm_model = LinearRegression()
lm_model.fit(x, y)
# extrair coeficientes
alfa = lm_model.coef_         # Caso ajustassemos para Yˆ=βˆ + β1 * x # uma reta #
beta = lm_model.intercept_


# In[55]:


print("alfa =" + str(alfa) +  "e beta = "+ str(beta))


# In[56]:


plt.scatter(x,y,s=20,color="blue")
plt.plot(x,(x * alfa + beta), color='r')
plt.title('Regressao Linear') 
plt.xlabel('Ponto de Orvalho') 
plt.ylabel('Umidade Relativa') 
plt.show()


#                                      TABELA OLS 01
#                                      
# Comentários:Pode-se concluir que temos uma correlação positiva entre as duas variáveis...

# In[57]:


# é necessário adicionar uma constante a matriz X
x_sm = sm.add_constant(x)
# OLS vem de Ordinary Least Squares e o método fit irá treinar o modelo
results = sm.OLS(y, x_sm).fit()
# mostrando as estatísticas do modelo
results.summary()


#                                  REGRESSÃO NÃO LINEAR
#                              
#                                                 

# In[111]:


from sklearn.preprocessing import PolynomialFeatures # importando um modelo nao linear 
  
poly = PolynomialFeatures(degree =5) 
X_poly = poly.fit_transform(x) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 


# In[112]:




plt.scatter(x, y, color = 'blue') 

plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red') 
plt.title('Regressao Nao Linear') 
plt.xlabel('Ponto de Orvalho') 
plt.ylabel('Umidade Relativa') 
plt.show()


# In[113]:


alfa2=lin2.coef_
beta2=lin2.intercept_
print("alfa2 = " + str(alfa2) + "  e beta2 = " + str(beta2))


#                                  TABELA OLS 02

# In[122]:


x_sm2 = sm.add_constant(X_poly)
results = sm.OLS(y, x_sm2).fit()
# mostrando as estatísticas do modelo
results.summary()


#                              GRÁFICO DENSIDADE KERNEL 
#                               
#                                       
#                                      
# 

# In[121]:


import seaborn as sns # Grafico variado


sns.kdeplot(df_ , shade=False, cut=0.5,cmap="Purples_d",label= "kernel")  

plt.show()
# Densidade de Kernel

