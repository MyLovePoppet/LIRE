package net.semanticmetadata.lire.imageanalysis.features.global.cnn;

import net.semanticmetadata.lire.builders.DocumentBuilder;
import net.semanticmetadata.lire.imageanalysis.features.GlobalFeature;
import net.semanticmetadata.lire.imageanalysis.features.LireFeature;
import net.semanticmetadata.lire.utils.MetricsUtils;
import net.semanticmetadata.lire.utils.SerializationUtils;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.model.AlexNet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.IOException;

public class CNN implements GlobalFeature {
    private final static MultiLayerNetwork multiLayerNetWork;
    private final static Java2DNativeImageLoader nativeImageLoader;
    private final static int endLayer = 11;
    private double[] feature;

    static {
        multiLayerNetWork = AlexNet.builder()
                .numClasses(1000)
                .cacheMode(CacheMode.DEVICE)
                .build().init();
        nativeImageLoader = new Java2DNativeImageLoader(227, 227, 3);
    }

    @Override
    public void extract(BufferedImage image) {
        INDArray inputArray = null;
        try {
            inputArray = nativeImageLoader.asMatrix(image);
        } catch (IOException e) {
            e.printStackTrace();
        }
        INDArray featureArray = multiLayerNetWork.activateSelectedLayers(0, endLayer, inputArray);
        this.feature = featureArray.toDoubleVector();
    }

    @Override
    public String getFeatureName() {
        return "CNN";
    }

    @Override
    public String getFieldName() {
        return DocumentBuilder.FIELD_NAME_CNN;
    }

    @Override
    public byte[] getByteArrayRepresentation() {
        return SerializationUtils.toByteArray(feature);
    }

    @Override
    public void setByteArrayRepresentation(byte[] featureData) {
        this.feature = SerializationUtils.toDoubleArray(featureData);
    }

    @Override
    public void setByteArrayRepresentation(byte[] featureData, int offset, int length) {
        this.feature = SerializationUtils.toDoubleArray(featureData, offset, length);
    }

    @Override
    public double getDistance(LireFeature f) {
        if (!(f instanceof CNN)) throw new UnsupportedOperationException("Wrong descriptor.");
        return MetricsUtils.distL2(feature, ((CNN) f).feature);
    }

    @Override
    public double[] getFeatureVector() {
        return feature;
    }
}
