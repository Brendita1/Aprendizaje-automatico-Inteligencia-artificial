import os
import copy

def Diccionario():
	nodos = {

            'A': {'B':3, 'C':2 },
         	'B': {'A':7, 'E':4},
            'C': {'A':7, 'D':1},
			'D': {'F':9, 'C':2},
			'E': {'B':3,'F':9,'G':5},
			'F': {'D':1,'E':4},
			'G': {'E':4},

	}
	return nodos

def pintar(actual,siguiente,grafo):
	lista=[]
	lista_colores=['rojo','verde','azul','negro','amarillo']
	for i in lista_colores:
		for j in grafo[i]:
			lista_colores[i]=grafo[j]
		if(actual == siguiente):
			lista_colores[i+1]= grafo[j]
	return lista_colores



def Recorrido(grafo, nodo):
	
	actual = [nodo]
	nivel = 0
	niveles= {nodo: nivel}	
	while len(actual) > 0:
		nivel += 1
		despues = []
		for i in actual:
			for j in grafo[i]:
				if j not in niveles:
					niveles[j] = nivel
					despues.append(j)
		actual = despues
	print(lista_colores)	
	return niveles
nodo=[]
actual = [nodo]
siguiente = []
listaux= Diccionario()
lista_colores=['rojo','verde','azul','negro','amarillo']
lista = Diccionario()
print(lista)
print("\n BFS: ", Recorrido(lista,'A'))
pintar(actual,siguiente,listaux)
 