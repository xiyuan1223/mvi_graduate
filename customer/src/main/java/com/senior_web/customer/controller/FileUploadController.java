package com.senior_web.customer.controller;

import com.alibaba.dubbo.config.annotation.Reference;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.senior_web.common.service.AttachmentService;
import com.senior_web.common.service.DataFlowControl;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.Resource;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.util.Map;
import java.util.UUID;

@Controller
@RequestMapping("/file")
public class FileUploadController implements Serializable {
    @Resource
    @Reference(version = "1.0.0")
    AttachmentService attachmentService;

    @Resource
    @Reference(version = "1.0.0")
    DataFlowControl dataFlowControl;

    @RequestMapping(value = "/uploadImage", method = RequestMethod.POST)
    @ResponseBody
    public void receiveImage(@RequestPart("file") MultipartFile file, HttpServletRequest request, HttpServletResponse response) throws IOException {
        String vicePath = "custempfile";
        //传到项目所在目录同级目录下
        File resourcefile = new File("tmp_file");
        String projectRealPath = resourcefile.getAbsolutePath();
//        String projectRealPath = env.getProperty("upload.path");
        // get the real path to store received images

        //存放文件的本地文件夹
        String realPath = projectRealPath+File.separator+vicePath;
        File imageDir = new File(realPath);
        if(!imageDir.exists()) {//文件夹
            imageDir.mkdirs();
        }
        try {
            String fileName = file.getOriginalFilename();
            String localFileName = System.currentTimeMillis() + "-" + fileName;
            String suffix = fileName.substring(fileName.lastIndexOf("."));
            //根据前缀和后缀创建一个空文件
            localFileName = realPath+File.separator+localFileName;
            File tempFile = new File(localFileName);
            file.transferTo(tempFile);
            System.out.println("custom写入文件："+localFileName);

            //对象流的传输需要序列化，不能直接getBytes,而应该使用ObjectOutputStrem.write()写入
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(byteArrayOutputStream);

//            file.transferTo(tempFile);

            oos.writeObject(tempFile);

            oos.flush();

            Map<String, String> map = dataFlowControl.dataFlow(byteArrayOutputStream.toByteArray(), file.getOriginalFilename());
            PrintWriter out = response.getWriter();
            String CKEditorFuncNum = request.getParameter("CKEditorFuncNum");
            String imgUrl = map.get("url");
            System.out.println(imgUrl + "上传图片路径");
            out.println("<script type=\"text/javascript\">");
            out.println("window.parent.CKEDITOR.tools.callFunction("
                    + CKEditorFuncNum + ",'" + imgUrl + "','')");
            out.println("</script>");
            deleteFile(tempFile);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void deleteFile(File... files) {
        for (File file : files) {
            if (file.exists()) {
                file.delete();
            }
        }
    }


}