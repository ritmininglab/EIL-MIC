import java.util.*;
import java.io.*;


import java.io.File;
import java.io.IOException;
import java.util.List;


public class RemoveCommentsAllFiles{
   public static void main(String[] args) throws IOException {
   
      /**
      int length = "abcdbe".indexOf("bc")+1;
      int end = length+"bc".length()-1;
      System.out.println("Starting pos: "+length+", Ending pos: "+end);
      
      File folder = new File("beforejava/");
      File[] listOfFiles = folder.listFiles();
   
      for (int i = 0; i < listOfFiles.length; i++) {
         if (listOfFiles[i].isFile()) {
            System.out.println("File " + listOfFiles[i].getName());
         } else if (listOfFiles[i].isDirectory()) {
            System.out.println("Directory " + listOfFiles[i].getName());
         }
      }
      **/
      
      String foldername = "beforejava/";
      File folder = new File(foldername);
      File[] listOfFiles = folder.listFiles();
   
      for (int i = 0; i < listOfFiles.length; i++) {
         if (listOfFiles[i].isFile()) {
            System.out.println("File " + listOfFiles[i].getName());
         } else if (listOfFiles[i].isDirectory()) {
            System.out.println("Directory " + listOfFiles[i].getName());
         }
      
         String data = "";
         String filename = listOfFiles[i].getName();
         String outputfilename = ""+filename;
      
         try {
            Scanner scan = new Scanner(new File(foldername+filename));
            while(scan.hasNextLine()) {
               String currentline = scan.nextLine();
               int pos = currentline.indexOf("#");
               if (pos<0){ // this line does not contain #
                  data = data + currentline + "\n";
               }else if(pos>0){ // this line contains # later
                  currentline = currentline.substring(0,pos);
               //System.out.println(pos+":"+currentline);
                  if(currentline.trim().length()>0){ // this line contains only space before #
                     data = data + currentline + "\n";
                  }
               }
            
            }
         //System.out.println(data);
         }
         catch(Exception e) {
            System.out.println(e.getMessage());
         }
      
      
         try{
            File fout = new File(outputfilename);
            FileOutputStream fos = new FileOutputStream(fout);
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
            bw.write(data);
            bw.close();   
         } catch (IOException e){
         //e.printStackTrace();
         }
      
      
      }
   
   
   }
}