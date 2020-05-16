package com.senior_web.common.domin;

import java.io.Serializable;
import java.util.PrimitiveIterator;

public class User implements Serializable{
    private Integer userId;

    private  String name;





    private String password;



    public Integer getUserId() {
        return userId;
    }

    public void setUserId(Integer userId) {
        this.userId = userId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }



    @Override
    public String toString() {
        return "User{" +
                "userId=" + userId +
                ", name=" + name +
                ", password=" + password+'}';
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}
