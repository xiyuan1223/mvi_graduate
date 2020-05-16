package com.senior_web.common.service;

import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Map;

public interface AttachmentService  {

    public Map<String, String> ckEditorUploadImage(byte[] bytes,String originName);

}


