# IA4legos-ML

Proyecto sobre aprendizaje automático con Python

## PARTE 1: INTRODUCCIÓN AL APRENDIZAJE AUTOMÁTICO		

1. Introducción a los algoritmos de aprendizaje automático: cuaderno01

> ¿Qué es el prendizaje automático? Aplicaciones. Tareas del aprendizaje automático. Etapas y escenarios de aprendizaje (Supervisado y no supervisado). Descripción de modelos de aprendizaje (Regresion, clasificación, reducción de la dimensión y agrupación).	

2. Scikit-learn: cuaderno02, cuaderno03, cuaderno04

> Introducción a Scikil-learn. Módulos para problemas de Machine Learning. Preparación de los datos en un problema de ML: preprocesado (separación de muestras), estandarización, normalización, codificación, tratamiento de valores pérdidos, y generación de tendencias polinómicas. Selección del modelo (medidas de error y búsqueda de los hyperparámetros del modelo). Pipelines.		

## PARTE 2: APRENDIZAJE SUPERVISADO					

3. Introducción a los modelos de aprendizaje automático supervisado: cuaderno05

> Modelos de regresión, modelos de clasificación y modelos de regresión y clasificación. Bancos de datos de trabajo.		

4. Modelos Lineales de regresión para outputs numéricos (RLS, RLM, ANCOVA): cuaderno06, cuaderno07, cuaderno08, cuaderno09
					
> Formulación del modelo. Preprocesado de los datos y análisis exploratorio inicial. Ajuste del modelo. Evaluación del modelo y predicción. Extensiones de los modelos de regresión: modelos polinomicos y suavizados. Regularización del modelo: Ridge regression (regularización L2), y Lasso regression (regularización L1). Casos prácticos.

5. Modelos Lineales de regresión para outputs categóricos (regresión logística y multinomial): cuaderno10, cuaderno11, cuaderno12, cuaderno13
				
> Formulación del modelo para variables clasificadoras dicotómicas. Preprocesado de los datos y análisis exploratorio inicial. Ajuste del modelo. Evaluación del modelo y predicción.  Matriz de confusión y curvas ROC. Modelo para variables clasificadoras politómicas. Mapa de clasificación. Modelo General. Casos prácticos.
	
6. Modelo de clasificación. NAIVE BAYES: cuaderno14, cuaderno15
		
> ¿Qué son? ¿Cómo trabajan? ¿Cuando debemos usarlos y cuando no? Clasificadores Bayes para dos grupos.	Clasificadores Bayes para múltiples grupos.
	
7. Modelo de clasificación. K-VECINO MÁS CERCANO: cuaderno16, cuaderno17
	
> ¿Qué son? ¿Cómo trabajan? ¿Cuando debemos usarlos y cuando no? Algoritmos en KNN. Nuestra primera clasificación con KNN. KNN en aplicaciones prácticas.
	
8. Modelo de clasificación y regresión. SUPPORT VECTOR MACHINE: cuaderno18, cuaderno19, cuaderno20
		
> ¿Qué son? ¿Cómo trabajan? ¿Cuando debemos usarlos y cuando no? Preprocesado de los datos y clasificador lineal. Evaluación de la clasificación. Clasificadores no lineales. SVM para regresión.
	
9. Modelo de clasificación y regresión. DECISION TREE: cuaderno21, cuaderno22, cuaderno23
	
> Marco teórico de los árdoles de decisión ¿Cuando debemos usarlos y cuando no? Construcción del árbol de decisión. Nuestro primer árbol de decisión para: a) problemas de clasificación,  b) problemas de regresión. Árboles de decisión para problemas de clasificación. Árboles de decisión para problemas de regresión.

10. Modelo de clasificación y regresión. RANDOM FOREST: cuaderno24, cuaderno25, cuaderno26
		
> Marco teórico de los bosques aleatorios. Tipos de clasificadores y ejemplos. ¿Cuando debemos usarlos y cuando no? Construcción del árbol de decisión. Nuestro primer árbol de decisión para: a) problemas de clasificación, b) problemas de regresión. Bosques aleatorios en problemas de clasificación. Bosques aleatorios en problemas de regresión
	
## PARTE 3: APRENDIZAJE NO SUPERVISADO					

11. Modelos de reducción de la dimensión. COMPONENTES PRINCIPALES: cuaderno27, cuaderno28, cuaderno29
		
> Marco teórico de componentes principales. CP aleatorizadas y CP incrementales. Nuestro primer modelo de componentes principales: preparación de los datos, extracción de las componentes, variabilidad explicada y representación gráfica. Aplicación de las CP al reconocimiento de digitos y de caras.Combinación de las CP con otras técnicas de aprendizaje automático para clasificación.


12. Modelos de reducción de la dimensión. ANÁLISIS DISCRIMINANTE: cuaderno30, cuaderno31
 
> Introducción al análisis discriminante. Análisis discriminante lineal. Aprendizaje en LDA . Mejorando la función discriminante lineal. Extensiones del LDA: quadratic discriminant analysis (QDA), flexible discriminant analysis (FDA), y regularized discriminat analysis (RDA).	
					
13. Modelos de reducción de la dimensión. APRENDIZAJE MÚLTIPLE (MANIFOLD LEARNING): cuaderno32
		
> Ver colab del planificador
	
14. Modelos de agrupación. K-MEDIAS Y DBSCAN: cuaderno33, cuaderno34
			
> Introducción a los procedimientos de cluster. El algoritmo EM para resolver el problema de K-medias. Dificultades con el algoritmo EM. Análisis (preprocesado, selección de centroides, estimación del número de clusters -elbow method-, minibatch kmeans y evaluación de la solución). Ejemplos de aplicación del algoritmo de K-medias. Algoritmo DBSCAN.
	
15 Modelos de agrupación. AGRUPACIÓN JERÁRQUICA: cuaderno35

> Introducción a la agruoación jerárquica. Distancias entre sujetos y entre grupos. Dendograma. Preprocesado y estimación de la agrupación. Selección del número de grupos. Ejemplos de aplicación.

16. Modelos de agrupación. MODELOS DE MIXTURAS GAUSSINAS: cuaderno36 

> Motivación de los modelos de mixturas gaussianas. Generalizando el algoritmo EM. Selección de la matriz de covarianzas. Estimación de densidades mediante MG. Aplicaciones.		

## PARTE 4: AMPLIACIONES					

17. Sistemas de recomendación: cuaderno37, cuaderno38
			
> Introducción a los sistemas de recomendación. Tipos de recomendadores: simple, basados en contenido, filtrado colaborativo. Ejemplos demonstrativos de cada reomendador. Ventajas y desventajas de cada recomendador. Evaluando el sistema de recomendación.	Aplicaciones de los sistemas de recomendación a diferentes bancos de datos.	

18. Modelos de aprendizaje interados en aplicaciones web: cuaderno 39

> Introducción. Características de la web a tener en cuenta. Generación de nuestro modelo de ML. Integración de la solución en una aplicación web (librerías Flask y Pickle).	
					
19. Web scraping					
					
20. Automatización de documentos					
					
21. Procesado de lenguaje natural y análisis sentimental: 4			
			
> Introducción a los NPL. Módulos RE, NLTK, spaCy y TextBlob. Procesado de cadenas de texto I. Procesado de cadenas de texto II. Módulo Re. Análisis avanzado de textos. Preprocesado de datos :Codificación de textos, secuencias y vectorización de textos. Procesamiento de NLP. Análisis de textos y clasificación (similaridad y/o agrupación). Datos para el análisis sentimental y su relación con el NLP. Diccionarios de palabras. Procesamiento inicial. Análisis de polaridad y clasificación sentimental.
	
22. Series temporales: 3			

> Introducción a las series temporales. Preprocesado de datos. Análisis de tendencia y estacionalidad. Modelos básicos en el análisis de series temporales. Diagnóstico, predicción y medidas de error de predicción. Series temporales multivaraintes.
	
23. Aprendizaje reforzado: 1	
