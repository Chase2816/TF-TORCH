package com.lkyooo.test;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class WekaModelTest {

    public static void main(String[] args) throws Exception {
        Instances instances = ConverterUtils.DataSource.read("/Users/Administrator/Dev/Projects/research-projects/tfserving-maskrcnn/model/num.train.csv");
        instances.setClassIndex(instances.numAttributes() - 1);
        MultilayerPerceptron multilayerPerceptron = (MultilayerPerceptron) weka.core.SerializationHelper.read("/Users/Administrator/Dev/Projects/research-projects/tfserving-maskrcnn/model/weka-house-trained.model");
        //        multilayerPerceptron.buildClassifier(instances); //再次训练
        Evaluation evaluation = new Evaluation(instances);
        double[] predictions = evaluation.evaluateModel(multilayerPerceptron, instances);
        for (double prediction : predictions) {
            System.out.println(prediction);
        }
    }
}
