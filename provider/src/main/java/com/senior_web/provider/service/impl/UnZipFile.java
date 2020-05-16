package com.senior_web.provider.service.impl;


import java.io.*;
import java.nio.charset.Charset;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;


public class UnZipFile {


    /**
     * 解压到指定目录
     * @param zipPath
     * @param descDir
     */
    public static String unZipFiles(String zipPath, String descDir) throws IOException {
        return unZipFiles(new File(zipPath), descDir);
    }

    /**
     * 解压文件到指定目录
     * 解压后的文件名，和之前一致
     * @param zipFile	待解压的zip文件
     * @param descDir 	指定目录
     */
    @SuppressWarnings("rawtypes")
    public static String unZipFiles(File zipFile, String descDir) throws IOException {

        ZipFile zip = new ZipFile(zipFile,Charset.forName("GBK"));//解决中文文件夹乱码
        String name = zip.getName().substring(zip.getName().lastIndexOf('\\')+1, zip.getName().lastIndexOf('.'));

        File pathFile = new File(descDir+File.separator+name);
        if (!pathFile.exists()) {
            pathFile.mkdirs();
        }

        for (Enumeration<? extends ZipEntry> entries = zip.entries(); entries.hasMoreElements();) {
            ZipEntry entry = (ZipEntry) entries.nextElement();
            String zipEntryName = entry.getName();
            InputStream in = zip.getInputStream(entry);
            String outPath = (descDir +File.separator+ name +"/"+ zipEntryName).replaceAll("\\*", "/");

            // 判断路径是否存在,不存在则创建文件路径
            File file = new File(outPath.substring(0, outPath.lastIndexOf('/')));
            if (!file.exists()) {
                file.mkdirs();
            }
            // 判断文件全路径是否为文件夹,如果是上面已经上传,不需要解压
            if (new File(outPath).isDirectory()) {
                continue;
            }
            // 输出文件路径信息
//			System.out.println(outPath);

            FileOutputStream out = new FileOutputStream(outPath);
            byte[] buf1 = new byte[1024];
            int len;
            while ((len = in.read(buf1)) > 0) {
                out.write(buf1, 0, len);
            }
            in.close();
            out.close();
        }
        System.out.println("******************解压完毕********************");
        return descDir+File.separator+name;
    }

    //测试

    public static void main(String[] args) {
//        try {
//            unZipFiles(new File("E:/Study/Java.zip"), "E:/Study/abc/");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        File tempFile = new File("E:/Study/abc/abc.txt");
        try{
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(byteArrayOutputStream);
            System.out.println(byteArrayOutputStream.size());
            oos.writeObject(tempFile);
            oos.flush();
            System.out.println(byteArrayOutputStream.size());
            oos.flush();
        }catch (IOException e){
            e.printStackTrace();
        }


    }

}
