from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import os, git

def load_data(data_path="data/iris.csv", train_split=0.8):
    # create Spark Context
    spark = SparkSession(sparkContext=SparkContext.getOrCreate())

    # get the path to the data
    base_path = git.Repo('.', search_parent_directories=True).working_tree_dir
    path = os.path.join(base_path, data_path)
    
    # load the dataset
    data = spark.read.csv(path, header=True, inferSchema=True)

    # extract features and target variable
    preprocessed_data = data.rdd.map(lambda x: Row(features=Vectors.dense(x[:-1]), species=x[-1])).toDF()
    
    # encode labels
    stringindexer = StringIndexer(inputCol='species', outputCol='label')
    pipeline = Pipeline(stages=[stringindexer])
    labeled_data = pipeline.fit(preprocessed_data).transform(preprocessed_data)
    return labeled_data.randomSplit([train_split, 1-train_split], seed=1234)

def train(train_data):
    # train the model
    nb = NaiveBayes(featuresCol="features", labelCol="label")
    return nb.fit(train_data)

def test(model, test_data):
    # test the model
    pred_test = model.transform(test_data)

    # evaluate the model
    evaluator = MulticlassClassificationEvaluator(metricName='accuracy')

    print(f"Accuracy: {evaluator.setMetricName('accuracy').evaluate(pred_test)}")
    print(f"F1 score: {evaluator.setMetricName('f1').evaluate(pred_test)}")
    print(f"Weighted precision: {evaluator.setMetricName('weightedPrecision').evaluate(pred_test)}")
    print(f"Weighted recall: {evaluator.setMetricName('weightedRecall').evaluate(pred_test)}")

def main():
    # load the data
    train_data, test_data = load_data()

    # train the model
    naivebayes_model = train(train_data)

    # test the model
    test(naivebayes_model, train_data, test_data)

if __name__ == "__main__":
    main()