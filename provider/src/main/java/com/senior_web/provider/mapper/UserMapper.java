package com.senior_web.provider.mapper;

import com.senior_web.common.domin.User;
import org.apache.ibatis.annotations.*;

import java.util.List;

//import java.util.List;

@Mapper
public interface UserMapper {
    User getOne(int id);


    User getUserByName(String name);
    List<User> findList(User user);
    int saveUser(User user);

    User getUserById(int id);

    int updateUser(User user);
    User findUser();
    int deleteUser(int id);
//    @Results(id = "userMap", value = {
//            @Result(column = "userId", property = "userId"),
//            @Result(column = "name", property = "name"),
//            @Result(column = "age", property = "age"),
//            @Result(column = "sex", property = "sex")})
//    @Select("SELECT * FROM u_user")
//    List<User> getAll();
//
//    @Select("SELECT * FROM u_user t WHERE t.userId = #{userId}")
//    @ResultMap("userMap")
//    User getOne(Long userId);
}
