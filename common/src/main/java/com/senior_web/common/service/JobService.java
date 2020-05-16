package com.senior_web.common.service;

import com.senior_web.common.domin.Job;

import java.util.List;
import java.util.Map;

public interface JobService {

    Job getJobById(int id);
//    List<Job> findSet(int start_id,int stop_id);
        List<Job> findList(int startId,int stopId);
    int getJobTotal();
    List<Job> getJobByPage(int offset,int pageSize);
    int updateJob(Job user);
    int deleteJob(int id);
}

