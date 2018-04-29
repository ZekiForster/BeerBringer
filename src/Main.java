
// Imports the Google Cloud client library

import com.google.cloud.vision.v1.*;
import com.google.cloud.vision.v1.Feature.Type;
import com.google.gson.*;
import com.google.protobuf.ByteString;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Main {
    public static void main(String... args) throws Exception {
        // Instantiates a client
        try (ImageAnnotatorClient vision = ImageAnnotatorClient.create()) {

            // The path to the image file to annotate
            String fileName = args[0];

            // Reads the image file into memory
            Path path = Paths.get(fileName);
            byte[] data = Files.readAllBytes(path);
            ByteString imgBytes = ByteString.copyFrom(data);

            // Builds the image annotation request
            List<AnnotateImageRequest> requests = new ArrayList<>();
            Image img = Image.newBuilder().setContent(imgBytes).build();
            Feature feat = Feature.newBuilder().setType(Type.FACE_DETECTION).build();
            AnnotateImageRequest request = AnnotateImageRequest.newBuilder()
                    .addFeatures(feat)
                    .setImage(img)
                    .build();
            requests.add(request);

            // Performs label detection on the image file
            BatchAnnotateImagesResponse response = vision.batchAnnotateImages(requests);
            List<AnnotateImageResponse> responses = response.getResponsesList();

            //Create a list that holds the values for the unhappy faces
            JsonArray unhappiness = new JsonArray();

            for (AnnotateImageResponse res : responses) {
                if (res.hasError()) {
                    System.out.printf("Error: %s\n", res.getError().getMessage());
                    return;
                }

                //Transfer the sorrow and the face box coords into a json array
                for (FaceAnnotation annotation : res.getFaceAnnotationsList()) {
                    //System.out.println("S: " + annotation.getSorrowLikelihoodValue());
                    //System.out.println("J: "+ annotation.getJoyLikelihoodValue());
                    //System.out.println("A: "+annotation.getAngerLikelihoodValue());
                    //System.out.println("Su: "+annotation.getSurpriseLikelihoodValue());
                    JsonObject person = new JsonObject();
                    JsonPrimitive element = new JsonPrimitive(annotation.getSorrowLikelihoodValue());
                    person.add("Sorrow", element);
                    element = new JsonPrimitive(annotation.getJoyLikelihoodValue());
                    person.add("Joy", element);
                    JsonArray vertices = new JsonArray();
                    List<Vertex> verticesList = annotation.getBoundingPoly().getVerticesList();

                    for(Vertex vert : verticesList){
                        JsonArray vertex = new JsonArray();
                        vertex.add(vert.getX());
                        vertex.add(vert.getY());
                        vertices.add(vertex);
                    }
                    person.add("Vertices", vertices);
                    unhappiness.add(person);
                }
            }



            //Now if we have the saddest person
            for(JsonElement pers : unhappiness){
                if (!pers.isJsonNull()){
                    //Calculate the central pixel
                    JsonArray vertTemp = pers.getAsJsonObject().get("Vertices").getAsJsonArray();
                    JsonArray vertTL = vertTemp.get(0).getAsJsonArray();
                    JsonArray vertBR = vertTemp.get(2).getAsJsonArray();
                    JsonArray midPoint = new JsonArray();
                    midPoint.add((vertBR.get(0).getAsInt()+vertTL.get(0).getAsInt())/2);
                    midPoint.add((vertBR.get(1).getAsInt()+vertTL.get(1).getAsInt())/2);
                    pers.getAsJsonObject().add("MidPoint", midPoint);
                }
            }
            List<JsonObject> list = new ArrayList<>();

            for(JsonElement el : unhappiness){
                list.add(el.getAsJsonObject());
            }

            Collections.sort(list, new Comparator<JsonObject>() {
                @Override
                public int compare(JsonObject o1, JsonObject o2) {
                    if(o1.get("Sorrow").getAsFloat() > o2.get("Sorrow").getAsFloat()){
                        return -1;
                    }else if(o1.get("Sorrow").getAsFloat() < o2.get("Sorrow").getAsFloat()){
                        return 1;
                    }else{
                        return 0;
                    }
                }
            });

            Collections.sort(list, new Comparator<JsonObject>() {
                @Override
                public int compare(JsonObject o1, JsonObject o2) {
                    if(o1.get("Joy").getAsFloat() > o2.get("Joy").getAsFloat()){
                        return 1;
                    }else if(o1.get("Joy").getAsFloat() < o2.get("Joy").getAsFloat()){
                        return -1;
                    }else{
                        return 0;
                    }
                }
            });

            if(!list.isEmpty()){
                System.out.println(list);
            }else{
                System.out.println("No Faces Detected!");
            }



        }
    }
}