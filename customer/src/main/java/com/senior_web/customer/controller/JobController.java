package com.senior_web.customer.controller;


import com.alibaba.dubbo.config.annotation.Reference;
import com.senior_web.common.domin.Job;

import com.senior_web.common.service.JobService;

import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
public class JobController{
    @Resource
    @Reference(version = "1.0.0")
    private JobService jobService;

    @GetMapping("getJobById")
    public Job getUserById(){
        System.out.println("job controller hit ");

        int id = 6;
        return jobService.getJobById(id);
    }

    @GetMapping("getJobTotal")
    public int getJobTotal(){
        return jobService.getJobTotal();
    }


    @RequestMapping(value = "/getJobByPage",method = RequestMethod.POST,produces="application/json;charset=UTF-8")
    @ResponseBody
    public Map<String,Object> getPage(@RequestParam("page")int page,@RequestParam("pageSize")int pageSize){
        Map<String ,Object> map = new HashMap<String,Object>();
        int offset = pageSize*page;
        List<Job> jobs = jobService.getJobByPage(offset,pageSize);
        map.put("jobs",jobs);
        return map;
    }


    @GetMapping("findList")
    public List<Job> findList(){
        System.out.println("job controller hit ");

        int startId = 6;
        int stopId = 18000;
        List<Job> jobs = jobService.findList(startId, stopId);
        System.out.println(jobs.size());
        return jobs;
    }


//    @GetMapping("updateUser")
//    public void updateUser(){
//        System.out.println("进来了！！！！！！！！！！！！！！！！");
//        System.out.println(jobService.hashCode()+"!!!!!!!!!!!!!!!!");
//        Job job = new Job();
//        int i = jobService.updateJob(job);
//        System.out.println("是否保存成功？-------------"+i);
//    }

    @GetMapping("deleteJob")
    public void deleteUser(){
        System.out.println("进来了！！！！！！！！！！！！！！！！");
        System.out.println(jobService.hashCode()+"!!!!!!!!!!!!!!!!");
        int id = 3;
        int i = jobService.deleteJob(id);
        System.out.println("是否保存成功？-------------"+i);
    }
}
