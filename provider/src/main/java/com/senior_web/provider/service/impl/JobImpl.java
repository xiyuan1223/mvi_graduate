package com.senior_web.provider.service.impl;

import com.alibaba.dubbo.config.annotation.Service;
import com.senior_web.common.domin.Job;

import com.senior_web.common.service.JobService;
import com.senior_web.provider.mapper.JobMapper;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import javax.annotation.Resource;
import java.util.List;
import java.util.concurrent.TimeUnit;


@Service(version = "1.0.0")
public class JobImpl implements JobService {
    private static final Logger LOGGER = LoggerFactory.getLogger(JobImpl.class);

    @Resource
    private JobMapper jobMapper;

    @Autowired
    private RedisTemplate redisTemplate;



    /**
     * 获取job逻辑：
     * 如果缓存存在，从缓存中获取job信息
     * 如果缓存不存在，从 DB 中获取job信息，然后插入缓存
     */
    @Override
    public Job getJobById(int id) {
        //从缓存中获取城市信息
        String key = "id";
        ValueOperations<String,Job> operations = redisTemplate.opsForValue();

        //缓存存在
        boolean hasKey = redisTemplate.hasKey(key);
        Job j = operations.get(key);
        System.out.println("是否有缓存："+hasKey+"  缓存中的值是："+j);
        if(hasKey){
            Job job = operations.get(key);
            LOGGER.info("JobImpl.updateJob() : 从缓存中获取了job >> " + job.toString());
            return job;
        }
        //从数据库中获取job数据
        Job job = jobMapper.getJobById(id);

        //插入缓存
        operations.set(key, job, 4, TimeUnit.HOURS);
        LOGGER.info("JobImpl.findJobById() :job插入缓存 >> " + job.toString());
        return job;
    }
    @Override
    public List<Job> findList(int startId,int stopId) {
//        //从缓存中获取城市信息
//        String key = "id";
//        ValueOperations<String,Job> operations = redisTemplate.opsForValue();
//
//        //缓存存在
//        boolean hasKey = redisTemplate.hasKey(key);
//        Job j = operations.get(key);
//        System.out.println("是否有缓存："+hasKey+"  缓存中的值是："+j);
//        if(hasKey){
//            Job job = operations.get(key);
//            LOGGER.info("JobImpl.updateJob() : 从缓存中获取了job >> " + job.toString());
//            return job;
//        }
//        //从数据库中获取job数据
        return  jobMapper.findList(startId,stopId);

        //插入缓存
//        operations.set(key, job, 4, TimeUnit.HOURS);
//        LOGGER.info("JobImpl.findJobById() :job插入缓存 >> " + job.toString());
        
    }

//    @Override
//    public List<Job> findSet(int start_id,int stop_id) {
//        //从缓存中获取城市信息
//        String key = "id";
//        ValueOperations<String,Job> operations = redisTemplate.opsForValue();
//
//        //缓存存在
//        boolean hasKey = redisTemplate.hasKey(key);
//        Job j = operations.get(key);
//        System.out.println("是否有缓存："+hasKey+"  缓存中的值是："+j);
//        if(hasKey){
//            Job job = operations.get(key);
//            LOGGER.info("JobImpl.updateJob() : 从缓存中获取了job >> " + job.toString());
//            return job;
//        }
//        //从数据库中获取job数据
//        Job job = jobMapper.getJobbyId(id);
//
//        //插入缓存
//        operations.set(key, job, 4, TimeUnit.HOURS);
//        LOGGER.info("JobImpl.findJobById() :job插入缓存 >> " + job.toString());
//        return job;
//    }


//    @Override
////    public List<Job> findAll() {
////        //从缓存中获取城市信息
////        String key = "id";
////        ValueOperations<String,Job> operations = redisTemplate.opsForValue();
////
////        //缓存存在
////        boolean hasKey = redisTemplate.hasKey(key);
////        Job j = operations.get(key);
////        System.out.println("是否有缓存："+hasKey+"  缓存中的值是："+j);
////        if(hasKey){
////            Job job = operations.get(key);
////            LOGGER.info("JobImpl.updateJob() : 从缓存中获取了job >> " + job.toString());
////            return job;
////        }
////        //从数据库中获取job数据
////        Job job = jobMapper.getJobById(id);
////
////        //插入缓存
////        operations.set(key, job, 4, TimeUnit.HOURS);
////        LOGGER.info("JobImpl.findJobById() :job插入缓存 >> " + job.toString());
////        return job;
////    }

    /**
     * 更新job逻辑：
     * 如果缓存存在，从缓存中删除job信息
     * 如果缓存不存在，不操作
     */
    @Override
    public int updateJob(Job job){
        int ret = jobMapper.updateJob(job);

        //缓存存在，删除缓存
        String key = job.getId().toString();
        boolean haskey = redisTemplate.hasKey(key);
        if (haskey){
            redisTemplate.delete(key);
            LOGGER.info("jobImmpl.updatejob() : 从缓存中删除job >> " + job.toString());
        }
        return ret;
    }

    /**
     * 获取job总数
     */

    @Override
    public int getJobTotal(){
        return jobMapper.getJobTotal();
    }


    /**
     * 获取分页信息
     */
    @Override
    public List<Job> getJobByPage(int offset,int pageSize){
        return jobMapper.getJobByPage(offset, pageSize);
    }


    /**
     * 删除job逻辑：
     * 如果缓存存在，从缓存中删除job信息
     * 如果缓存不存在，不操作
     */
    @Override
    public int deleteJob(int id){
        int ret = jobMapper.deleteJob(id);

        //缓存存在，删除缓存
        String key = ""+id;
        System.out.println("key的值为： " +key);
        boolean haskey = redisTemplate.hasKey(key);
        ValueOperations<String,Job> operations = redisTemplate.opsForValue();
        Job j = operations.get(key);
        System.out.println("是否有缓存："+haskey+"  缓存中的值是："+j);
        if (haskey){
            LOGGER.info("jobImmpl.updatejob() : 从缓存中删除job >> " + operations.get(key));
            redisTemplate.delete(key);
        }
        return ret;
    }


}
