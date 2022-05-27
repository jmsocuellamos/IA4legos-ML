# IA4legos-ML

Proyecto sobre aprendizaje automático con Python

## PARTE 1: INTRODUCCIÓN AL APRENDIZAJE AUTOMÁTICO		

1. Introducción a los algoritmos de aprendizaje automático: cuaderno01

> Contenidos: ¿Qué es el prendizaje automático? Aplicaciones. Tareas del aprendizaje automático. Etapas y escenarios de aprendizaje (Supervisado y no supervisado). Descripción de modelos de aprendizaje (Regresion, clasificación, reducción de la dimensión y agrupación).	

2. Scikit-learn: cuaderno02, cuaderno03, cuaderno04

> Contenidos: Introducción a Scikil-learn. Módulos para problemas de Machine Learning. Preparación de los datos en un problema de ML: preprocesado (separación de muestras), estandarización, normalización, codificación, tratamiento de valores pérdidos, y generación de tendencias polinómicas. Selección del modelo (medidas de error y búsqueda de los hyperparámetros del modelo). Pipelines.		

## PARTE 2: APRENDIZAJE SUPERVISADO					

Introducción a los modelos de aprendizaje automático supervisado		1	Modelos de regresión, modelos de clasificación y modelos de regresión y clasificación. Bancos de datos de trabajo.		
Modelos Lineales de regresión para outputs numéricos (RLS, RLM, ANCOVA)		4			
			Formulación del modelo. Preprocesado de los datos y análisis exploratorio inicial.	1	
			Ajuste del modelo. Evaluación del modelo y predicción.	1	
			Extensiones de los modelos de regresión: modelos polinomicos y suavizados. Regularización del modelo: Ridge regression (regularización L2), y Lasso regression (regularización L1). 	1	
			Casos prácticos	1	
Modelos Lineales de regresión para outputs categóricos (regresión logística y multinomial)		4			
			Formulación del modelo para variables clasificadoras dicotómicas. Preprocesado de los datos y análisis exploratorio inicial.	1	
			Ajuste del modelo. Evaluación del modelo y predicción.  Matriz de confusión y curvas ROC.	1	
			Modelo para variables clasificadoras politómicas. Mapa de clasificación. Modelo General.	1	
			Casos prácticos	1	
Modelo de clasificación: NAIVE BAYES		2			
			¿Qué son? ¿Cómo trabajan? ¿Cuando debemos usarlos y cuando no? Clasificadores Bayes para dos grupos	1	
			Clasificadores Bayes para múltiples grupos.	1	
Modelo de clasificación: K-VECINO MÁS CERCANO		2			
			¿Qué son? ¿Cómo trabajan? ¿Cuando debemos usarlos y cuando no? Algoritmos en KNN. Nuestra primera clasificación con KNN	1	
			KNN en aplicaciones prácticas	1	
Modelo de clasificación y regresión: SUPPORT VECTOR MACHINE		2			
			¿Qué son? ¿Cómo trabajan? ¿Cuando debemos usarlos y cuando no? Preprocesado de los datos y clasificador lineal. Evaluación de la clasificación.	1	
			Clasificadores no lineales.	1	
			SVM para regresión	1	
Modelo de clasificación y regresión: DECISION TREE		3			
			Marco teórico de los árdoles de decisión ¿Cuando debemos usarlos y cuando no? Construcción del árbol de decisión. Nuestro primer árbol de decisión para: a) problemas de clasificación,  b) problemas de regresión. 	1	
			Árboles de decisión para problemas de clasificación.	1	
			Árboles de decisión para problemas de regresión.	1	
Modelo de clasificación y regresión: RANDOM FOREST		3			
			Marco teórico de los bosques aleatorios. Tipos de clasificadores y ejemplos. ¿Cuando debemos usarlos y cuando no? Construcción del árbol de decisión. Nuestro primer árbol de decisión para: a) problemas de clasificación, b) problemas de regresión	1	
			Bosques aleatorios en problemas de clasificación	1	
			Bosques aleatorios en problemas de regresión	1	
## PARTE 3: APRENDIZAJE NO SUPERVISADO					
Modelos de reducción de la dimensión: COMPONENTES PRINCIPALES		3			
			Marco teórico de componentes principales. CP aleatorizadas y CP incrementales. Nuestro primer modelo de componenets principales: preparación de los datos, extracción de las componentes, variabilidad explicada y representación gráfica. 	1	
			Aplicación de las CP al reconocimiento de digitos y de caras.	1	
			Combinación de las CP con otras técnicas de aprendizaje automático para clasificación.	1	
Modelos de reducción de la dimensión: ANÁLISIS DISCRIMINANTE		2			
			Introducción al análisis discriminante. Análisis discriminante lineal. Aprendizaje en LDA . Mejorando la función discriminante lineal.	1	
			Extensiones del LDA: quadratic discriminant analysis (QDA), flexible discriminant analysis (FDA), y regularized discriminat analysis (RDA)	1	
					
Modelos de reducción de la dimensión: APRENDIZAJE MÚLTIPLE (MANIFOLD LEARNING)		1			
			Ver colab del planificador	1	
Modelos de agrupación: K-MEDIAS Y DBSCAN		2			
			Introducción a los procedimientos de cluster. El algoritmo EM para resolver el problema de K-medias. Dificultades con el algoritmo EM. Análisis (preprocesado, selección de centroides, estimación del número de clusters -elbow method-, minibatch kmeans y evaluación de la solución). Ejemplos de aplicación del algoritmo de K-medias	1	
			Algoritmo DBSCAN	1	
Modelos de agrupación: AGRUPACIÓN JERÁRQUICA		1	Introducción a la agruoación jerárquica. Distancias entre sujetos y entre grupos. Dendograma. Preprocesado y estimación de la agrupación. Selección del número de grupos. Ejemplos de aplicación.	1	
Modelos de agrupación: MODELOS DE MIXTURAS GAUSSINAS		1	Motivación de los modelos de mixturas gaussianas. Generalizando el algoritmo EM. Selección de la matriz de covarianzas. Estimación de densidades mediante MG. Aplicaciones.	1	

## PARTE 4: AMPLIACIONES					

SISTEMAS DE RECOMENDACIÓN		2			
			Introducción a los sistemas de recomendación. Tipos de recomendadores: simple, basados en contenido, filtrado colaborativo. Ejemplos demonstrativos de cada reomendador. Ventajas y desventajas de cada recomendador. Evaluando el sistema de recomendación.	1	
			Aplicaciones de los sistemas de recomendación a diferentes bancos de datos.	1	
MODELOS DE APRENDIZAJE INTEGRADOS EN APLICACIONES WEB		1			
			Introducción. Características de la web a tener en cuenta. Generación de nuestro modelo de ML. Integración de la solución en una aplicación web (librerías Flask y Pickle)	1	
					
WEB SCRAPING					
					
AUTOMATIZACIÓN DE DOCUMENTOS					
					
PROCESADO DE LENGUAJE NATURAL y ANÁLISIS SENTIMENTAL		4			
			Introducción a los NPL. Módulos RE, NLTK, spaCy y TextBlob. Procesado de cadenas de texto I.	1	
			Procesado de cadenas de texto II. Módulo Re.	1	
			Análisis avanzado de textos. Preprocesado de datos :Codificación de textos, secuencias y vectorización de textos. Procesamiento de NLP. Análisis de textos y clasificación (similaridad y/o agrupación).	1	
			Datos para el análisis sentimental y su relación con el NLP. Diccionarios de palabras. Procesamiento inicial. Análisis de polaridad y clasificación sentimental.	1	
SERIES TEMPORALES		3			
			Introducción a las series temporales. Preprocesado de datos. Análisis de tendencia y estacionalidad. 	1	
			Modelos básicos en el análisis de series temporales. Diagnóstico, predicción y medidas de error de predicción.	1	
			Series temporales multivaraintes	1	
APRENDIZAJE REFORZADO					
		1	Introducción al aprendizaje reforzado.	1	
