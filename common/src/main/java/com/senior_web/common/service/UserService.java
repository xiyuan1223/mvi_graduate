package com.senior_web.common.service;

import com.senior_web.common.domin.User;

import java.util.List;

public interface UserService {

    User getUserById(int id);
    User getUserByName(String name);
    List<User> findList(User user);
    int saveUser(User user);

    User findUser();
    int updateUser(User user);
    int deleteUser(int id);
}
