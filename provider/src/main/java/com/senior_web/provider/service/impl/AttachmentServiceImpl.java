package com.senior_web.provider.service.impl;

import com.alibaba.dubbo.config.annotation.Service;
import com.senior_web.common.service.AttachmentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import java.io.*;
import java.util.HashMap;
import java.util.Map;

@Service(version = "1.0.0")
public class AttachmentServiceImpl implements AttachmentService, Serializable {
    @Autowired
    private Environment env;

    public static String protempfile = "protempfile";

    public Map<String, String> ckEditorUploadImage(byte[] bytes,String originName){


        String originalName = originName;


        //读取文件内容
        Object obj = null;
        try {
            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);
            ObjectInputStream ois = new ObjectInputStream(byteArrayInputStream);
            obj = (File)ois.readObject();




        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        if(obj==null || "".equals(originName.trim())) {
            return generateResult(false, "#");
        }



        //过去资源文件绝对路径
        File resourcefile = new File("tmp_file");
        String projectRealPath = resourcefile.getAbsolutePath();


        //创建子文件夹 protempfile
        String realPath = projectRealPath+File.separator+protempfile;
        File imageDir = new File(realPath);
        if(!imageDir.exists()) {//文件夹
            imageDir.mkdirs();
        }

        //获取本地路径
        boolean fileStoreTag = false;

        // generate file name
        String localFileName = System.currentTimeMillis() + "-" + originalName;
        String localFilePath = realPath + File.separator + localFileName;
        try {
            ((File) obj).renameTo(new File(localFilePath));

        } catch (Exception e) {
            System.out.println("文件创建失败");
            e.printStackTrace();

            // log here
        }


        System.out.println("provider 本地路径: " + localFilePath);

        // log here -

        Map<String,String> acceptDataStatus = new HashMap<String,String>();
        acceptDataStatus.put("localFilePath",localFilePath);

        return acceptDataStatus;
    }

    private Map<String, String> generateResult(boolean uploaded, String relativeUrl){
        Map<String, String> result = new HashMap<String, String>();
        result.put("uploaded", uploaded + "");
        result.put("url", relativeUrl);

        return result;
    }





    /**
     * 将文件写入指定的路径
     * @param file File 对象
     * @param path 磁盘路径
     */
    public  boolean WriteObject(File file,String path) {

        ObjectOutputStream oos =null;
        try {
            // 把对象写入到文件中，使用ObjectOutputStream

            oos = new ObjectOutputStream(
                    new FileOutputStream(path));
            oos.writeObject(file);
            // 把对象写入到文件中
            System.out.println("写入文件完毕！");
            return true;
        } catch (IOException e) {
            System.out.println(e.getMessage() + "错误！");
            return false;
        }finally{
            try {
                oos.close();//关闭输出流
            } catch (IOException e) {
                System.out.println("关闭文件出错");
            }

        }
    }





}
