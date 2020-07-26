package net.semanticmetadata.lire.imageanalysis.features;


import net.semanticmetadata.lire.imageanalysis.features.global.ACCID;
import net.semanticmetadata.lire.imageanalysis.features.global.cnn.CNN;
import net.semanticmetadata.lire.utils.FileUtils;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;

public class TestCNN {
    public static void main(String[] args) throws IOException {
        ArrayList<File> imageFiles = FileUtils.getAllImageFiles(new File("src/test/resources/images"), true);
        CNN cnn = new CNN();
        for (Iterator<File> iterator = imageFiles.iterator(); iterator.hasNext(); ) {
            File nextImage = iterator.next();
            cnn.extract(ImageIO.read(nextImage));
            System.out.println(nextImage.getName() + ": " + Arrays.toString(cnn.getFeatureVector()));
        }
    }
}
