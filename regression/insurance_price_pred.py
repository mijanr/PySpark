from pyspark import SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
import git

class InsurancePricePred:
    def __init__(self):
        # spark session & context
        self.spark = SparkSession.builder.appName("insurance_price_pred").getOrCreate()
        self.sc = SparkContext.getOrCreate()
        # data path
        self.base_path = git.Repo('.', search_parent_directories=True).working_tree_dir
        self.data_path = self.base_path + "/data/insurance.csv"

    def load_data(self, train_size=0.8):
        # load data
        self.df = self.spark.read.csv(self.data_path, header=True, inferSchema=True)
        
        # encode ['sex', 'smoker', 'region'] to numeric
        encode_cols = ['sex', 'smoker', 'region']
        indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(self.df) for column in encode_cols]
        pipeline = Pipeline(stages=indexers)
        self.df = pipeline.fit(self.df).transform(self.df)
        
        # drop original columns
        self.df = self.df.drop(*encode_cols)

        # take charges to the last column
        self.df = self.df.select([c for c in self.df.columns if c != 'charges'] + ['charges'])

        # extract features and label
        preprocessed_data = self.df.rdd.map(lambda row: [Vectors.dense(row[0:-1]), row[-1]]).toDF(['features', 'label'])

        ## split data
        train, test = preprocessed_data.randomSplit([train_size, 1-train_size], seed=42)

        return train, test

    def train(self, train, model_name='linear_regression'):
        if model_name == 'linear_regression':
            model = LinearRegression(featuresCol='features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
        elif model_name == 'random_forest':
            model = RandomForestRegressor(featuresCol='features', labelCol='label')
        elif model_name == 'gbt':
            model = GBTRegressor(featuresCol='features', labelCol='label', maxIter=10)
        else:
            raise ValueError('model_name should be one of linear_regression, random_forest, gbt')
        
        # train
        model = model.fit(train)

        return model
    
    def evaluate(self, model, test):
        # evaluate
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')
        rmse = evaluator.evaluate(predictions)
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        return rmse, r2
    
    def predict(self, model, data):
        predictions = model.transform(data)
        return predictions
    

if __name__ == "__main__":
    # initialize class
    ip = InsurancePricePred()

    # load the data
    train, test = ip.load_data()

    # train the model
    model = ip.train(train, model_name='gbt') # linear_regression, random_forest, gbt
    
    # evaluate the model
    rmse, r2 = ip.evaluate(model, test)
    print('rmse: ', rmse)
    print('r2: ', r2)

    # kill spark session
    try:
        ip.spark.stop()
    except:
        pass
    
