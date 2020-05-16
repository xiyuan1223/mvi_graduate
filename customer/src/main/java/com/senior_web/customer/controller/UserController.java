package com.senior_web.customer.controller;

import com.alibaba.dubbo.config.annotation.Reference;
import com.senior_web.common.domin.User;
import com.senior_web.common.service.UserService;
import org.json.JSONException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.json.JSONObject;
import javax.annotation.Resource;
import java.util.List;
import java.util.Map;

@Controller
@RequestMapping("/lr")
public class UserController {
    private static final Logger LOGGER = LoggerFactory.getLogger(UserController.class);

    @Resource
    @Reference(version = "1.0.0")
    private UserService userService;

    @GetMapping("getUser")
    public User user(){
        System.out.println("进来了！！！！！！！！！！！！！！！！");
        System.out.println(userService.hashCode()+"!!!!!!!!!!!!!!!!");
        return userService.findUser();
    }

    @RequestMapping(value = "/login",method = RequestMethod.POST,produces="application/json;charset=UTF-8")
    @ResponseBody
    public String login(@RequestBody Map<String,Object> map) throws JSONException {
        String status = "";
        String username = (String) map.get("username");
        String password= (String) map.get("password");

        User user= userService.getUserByName(username);

        if(user==null || !(user.getPassword().equals(password))){
            status= "2";
            LOGGER.info("用户尝试登录失败: >> " + username);
        }

        else if(user.getPassword().equals(password)){
            status = "1";
            LOGGER.info("用户登录成功 >> " + username);

        }
        else{
            status="2";
            System.out.println("默认status2");


        }
        System.out.println(user);

        //数据库查询操作
        //状态码：1->登录成功；2->用户名和密码错误
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("status",status);
        System.out.println("status"+":"+status);


        return jsonObject.toString();

    }
    @RequestMapping(value = "/register",method = RequestMethod.POST,produces="application/json;charset=UTF-8")
    @ResponseBody
    public Object register(@RequestBody Map<String,Object> map) throws JSONException {
        String status = "2";
        String username = (String) map.get("username");
        String password= (String) map.get("password");
        System.out.println(username);
        System.out.println(password);
        // 用户名重复验证
        User ticket_user = userService.getUserByName(username);
        System.out.println(ticket_user);
        if(ticket_user!=null){
            status= "2";
        }
        else {
            User user = new User();
            user.setName(username);
            user.setPassword(password);
            userService.saveUser(user);
            //数据库查询操作
            //状态码：1->注册成功；2->用户名已存在
            status = "1";
        }
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("status",status);
        return jsonObject.toString();
    }


    @RequestMapping(value = "/tj",method = RequestMethod.POST,produces="application/json;charset=UTF-8")
    @ResponseBody
    public Object testJson(@RequestBody Map<String,Object> map) throws JSONException {
        System.out.println(map.get("username"));
        System.out.println(map.get("password"));
        System.out.println(map.get("openid"));
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("a","a");
        jsonObject.put("tttt","tttt");
        jsonObject.put("bbbbb","bbbbb");
        System.out.println(jsonObject.getString("a"));
        return jsonObject.toString();

    }
    @RequestMapping(value="/getUserById",method = RequestMethod.GET,produces="application/json;charset=UTF-8")
    public User getUserById(){
        System.out.println("进来了！！！！！！！！！！！！！！！！");
        System.out.println(userService.hashCode()+"!!!!!!!!!!!!!!!!");
        int id = 3;
        return userService.getUserById(id);
    }

    @GetMapping("saveUser")
    public void saveUser(){
        System.out.println("进来了！！！！！！！！！！！！！！！！");
        System.out.println(userService.hashCode()+"!!!!!!!!!!!!!!!!");
        User user = new User();
        user.setName("赵洋");

        int i = userService.saveUser(user);
        System.out.println("是否保存成功？-------------"+i);
    }

    @GetMapping("updateUser")
    public void updateUser(){
        System.out.println("进来了！！！！！！！！！！！！！！！！");
        System.out.println(userService.hashCode()+"!!!!!!!!!!!!!!!!");
        User user = new User();
        user.setUserId(1);
        user.setName("赵洋");

        int i = userService.updateUser(user);
        System.out.println("是否保存成功？-------------"+i);
    }

    @GetMapping("deleteUser")
    public void deleteUser(){
        System.out.println("进来了！！！！！！！！！！！！！！！！");
        System.out.println(userService.hashCode()+"!!!!!!!!!!!!!!!!");
        int id = 3;
        int i = userService.deleteUser(id);
        System.out.println("是否保存成功？-------------"+i);
    }

    @GetMapping("findList")
    public List<User> findList(){
        System.out.println("进来了！！！！！！！！！！！！！！！！");
        System.out.println(userService.hashCode()+"!!!!!!!!!!!!!!!!");
        User user = new User();
//        user.setUserId(1);
        return userService.findList(user);
    }
}
