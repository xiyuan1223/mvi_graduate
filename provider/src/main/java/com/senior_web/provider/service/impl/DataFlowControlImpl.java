package com.senior_web.provider.service.impl;

import com.alibaba.dubbo.config.annotation.Reference;
import com.senior_web.common.service.AttachmentService;
import com.senior_web.common.service.DataFlowControl;
import javax.annotation.Resource;
import java.io.*;
import java.util.HashMap;
import java.util.Map;
import com.senior_web.provider.service.impl.UnZipFile;

/**
 * 1.接收zip文件 2.解压缩 3.用test_data_preprocess处理 4.送入模型
 */


public class DataFlowControlImpl implements DataFlowControl{
    @Resource
    @Reference(version = "1.0.0")
    AttachmentService attachmentService;

    public Map<String, String> dataFlow(byte[] bytes, String originName) {

        /**
         * ctRealPath tmp_file子目录，用于存放解压后的文件
         * projectRealPath 资源存放目录tmp_file的绝对路径
         * 临时文件夹为ctRealPath(存放解压缩文件)，terminal_path(存放预处理后的文件) 都需要在最后删除
         */
        String ctFilePath = "ctpath";
        File resourcefile = new File("tmp_file");
        String projectRealPath = resourcefile.getAbsolutePath();
        String ctRealPath = projectRealPath+File.separator+ctFilePath;


        //在tmp_file下创建一个空的ctRealPath
        File ctRealPathDir = new File(ctRealPath);
        if(!ctRealPathDir.exists()) {
            ctRealPathDir.mkdirs();
        }
        else{
            //清空当前目录
            deleteDir(ctRealPathDir);
            ctRealPathDir.mkdirs();
        }



        /*************************获取zip文件并解压缩**********************************************************************/

        //接收文件以zip格式存储
        Map<String, String> acceptDataStatus = attachmentService.ckEditorUploadImage(bytes, originName);
        //获取zip文件的存储路径
        String zipCtFilePath = acceptDataStatus.get("localFilePath");
        String zipDoneFilePath = "";
        //解压缩到ctRealPath
        try{

            zipDoneFilePath = UnZipFile.unZipFiles(zipCtFilePath,ctRealPath);
        }catch (IOException e){
            e.printStackTrace();
            System.out.println("解压缩文件失败");

        }
        //解压成功删除压缩包
//        deleteDir(zipCtFilePath);

        /*************************调用test_data_preprocess进行处理**********************************************************************/
        //调用test_data_preprocess 脚本进行处理
        Process test_data_preprocess_proc;

        String ct_root_path = zipDoneFilePath;
        String terminal_path =  "python"+File.separator+"test_data";
        File terminalPathDir = new File(terminal_path);
        terminal_path = terminalPathDir.getAbsolutePath();



        if(!terminalPathDir.exists()) {
            terminalPathDir.mkdirs();
        }
        else{
            //清空当前目录
            deleteDir(terminalPathDir);
            terminalPathDir.mkdirs();

        }

        try{
            test_data_preprocess_proc = Runtime.getRuntime().exec("python python/tools/test_data_preprocess.py "+" --ct_root_path="
                    +ct_root_path+" --terminal_path="+terminal_path);// 执行py文件
        }
        catch (IOException e){
            System.out.println("test_data_preprocess 文件执行失败");
            e.printStackTrace();
        }




        /*************************调用网络模型计算并获取结果进行处理**********************************************************************/
        Process model_proc;
        String line="";
        String reStr = "";
        try{

            model_proc = Runtime.getRuntime().exec("python python/online_test.py --load_model_path=checkpoints/resnet34_1203_03_05_14.pth --test_data=test_data");// 执行py文件

            model_proc.waitFor();
            InputStreamReader ir = new InputStreamReader(model_proc.getInputStream());
            BufferedReader in = new BufferedReader(ir);
            while ((line = in.readLine()) != null) {
                reStr = line;
            }
            in.close();
            ir.close();
            model_proc.waitFor();

            System.out.println("模型输出结果");


        }
        catch (IOException e){
            System.out.println("test_data_preprocess 文件执行失败");
            e.printStackTrace();
        }catch (InterruptedException ite){
            System.out.println("从模型获取结果失败");
            ite.printStackTrace();
        }

        return acceptDataStatus;
    }

    private Map<String, String> generateResult(boolean uploaded, String relativeUrl) {
        Map<String, String> result = new HashMap<String, String>();
        result.put("uploaded", uploaded + "");
        result.put("url", relativeUrl);

        return result;
    }
    private static boolean deleteDir(File dir) {
        if (dir.isDirectory()) {
            String[] children = dir.list();
            //递归删除目录中的子目录下
            for (int i=0; i<children.length; i++) {
                boolean success = deleteDir(new File(dir, children[i]));
                if (!success) {
                    return false;
                }
            }
        }
        // 目录此时为空，可以删除
        return dir.delete();
    }
    private static boolean deleteDir(String filePath){
        File dir = new File(filePath);
        if (dir.isDirectory()) {
            String[] children = dir.list();
            //递归删除目录中的子目录下
            for (int i=0; i<children.length; i++) {
                boolean success = deleteDir(new File(dir, children[i]));
                if (!success) {
                    return false;
                }
            }
        }
        // 目录此时为空，可以删除
        return dir.delete();
    }
}
