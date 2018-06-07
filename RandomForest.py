# Databricks notebook source exported at Thu, 1 Dec 2016 20:11:29 UTC
from __future__ import print_function

import sys

import pandas as pd
import numpy as np

from pyspark import SparkContext, SQLContext

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

from pyspark.ml.feature import Word2Vec


# COMMAND ----------

data = sqlContext.read.json("/FileStore/tables/y75o9asp1480616538486/train__1_-3038d.json").select("cuisine","id","ingredients")

# COMMAND ----------

data.printSchema()
labelIndexer = StringIndexer(inputCol="cuisine", outputCol="indexedLabel").fit(data)

# COMMAND ----------



word2Vec = Word2Vec(vectorSize=50, inputCol="ingredients", outputCol="result")
model = word2Vec.fit(data)
result = model.transform(data)


# COMMAND ----------

res = labelIndexer.transform(result)

# COMMAND ----------

display(res)

# COMMAND ----------

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="result")

# COMMAND ----------

(trainingData, testingData) = res.randomSplit([0.8,0.2])

# COMMAND ----------

rfModel = rf.fit(trainingData)

# COMMAND ----------

predictions = rfModel.transform(testingData)

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

