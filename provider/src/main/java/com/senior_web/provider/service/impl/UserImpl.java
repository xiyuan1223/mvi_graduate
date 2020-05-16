package com.senior_web.provider.service.impl;

import com.alibaba.dubbo.config.annotation.Service;
import com.senior_web.common.domin.User;
import com.senior_web.common.service.UserService;
import com.senior_web.provider.mapper.UserMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import javax.annotation.Resource;
import java.util.List;
import java.util.concurrent.TimeUnit;


@Service(version = "1.0.0")
public class UserImpl implements UserService {

    private static final Logger LOGGER = LoggerFactory.getLogger(UserImpl.class);

    @Resource
    private UserMapper userMapper;

    @Autowired
    private RedisTemplate redisTemplate;

    @Override
    public User findUser() {
//        User user = new User();
//        user.setAge(12);
//        user.setId(1);
//        user.setName("zhangsan");
//        user.setSex("nan");
//        return user;
        return userMapper.getOne(1);
//        return userMapper.getOne(1l);
    }

    /**
     * 获取user逻辑：
     * 如果缓存存在，从缓存中获取user信息
     * 如果缓存不存在，从 DB 中获取user信息，然后插入缓存
     */
    @Override
    public User getUserById(int id) {
        //从缓存中获取城市信息
        String key = ""+id;
        ValueOperations<String,User> operations = redisTemplate.opsForValue();

        //缓存存在
        boolean hasKey = redisTemplate.hasKey(key);
        User u = operations.get(key);
        System.out.println("是否有缓存："+hasKey+"  缓存中的值是："+u);
        if(hasKey){
            User user = operations.get(key);
            LOGGER.info("UserImpl.updateUser() : 从缓存中获取了user >> " + user.toString());
            return user;
        }
//        从数据库中获取user数据
        User user = userMapper.getUserById(id);

        //插入缓存
//        operations.set(key, user, 4, TimeUnit.HOURS);
//        LOGGER.info("UserImpl.findUserById() :user插入缓存 >> " + user.toString());
        return user;
    }




    public int saveUser(User user){



        return userMapper.saveUser(user);
    }

    /**
     * 更新user逻辑：
     * 如果缓存存在，从缓存中删除user信息
     * 如果缓存不存在，不操作
     */
    public int updateUser(User user){
        int ret = userMapper.updateUser(user);

        //缓存存在，删除缓存
        String key = "user_" + user.getUserId();
        boolean haskey = redisTemplate.hasKey(key);
        if (haskey){
            redisTemplate.delete(key);
            LOGGER.info("UserImmpl.updateUser() : 从缓存中删除user >> " + user.toString());
        }
        return ret;
    }


    public User getUserByName(String name){
        //从缓存中获取城市信息
        String key = ""+name;
        ValueOperations<String,User> operations = redisTemplate.opsForValue();

        //缓存存在
        boolean hasKey = redisTemplate.hasKey(key);
        User u = operations.get(key);
        System.out.println("是否有缓存："+hasKey+"  缓存中的值是："+u);
        if(hasKey&&u!=null){
            User user = operations.get(key);
            if(user!=null)
                LOGGER.info("UserImpl.findUserByName() : 从缓存中获取了user >> " + user.toString());
            return user;
        }
        else {
//        从数据库中获取user数据
            User user = userMapper.getUserByName(name);
//            插入缓存
        operations.set(name, user, 4, TimeUnit.HOURS);
        if(user!=null)
            LOGGER.info("UserImpl.findUserByName() :user插入缓存 >> " + user.toString());

            return user;
        }

    }

    /**
     * 删除user逻辑：
     * 如果缓存存在，从缓存中删除user信息
     * 如果缓存不存在，不操作
     */
    public int deleteUser(int id){
        int ret = userMapper.deleteUser(id);

        //缓存存在，删除缓存
        String key = "user_" + id;
        System.out.println("key的值为： " +key);
        boolean haskey = redisTemplate.hasKey(key);
        ValueOperations<String,User> operations = redisTemplate.opsForValue();
        User u = operations.get(key);
        System.out.println("是否有缓存："+haskey+"  缓存中的值是："+u);
        if (haskey){
            LOGGER.info("UserImmpl.updateUser() : 从缓存中删除user >> " + operations.get(key));
            redisTemplate.delete(key);
        }
        return ret;
    }

    @Override
    public List<User> findList(User user) {
        return userMapper.findList(user);
    }
}
