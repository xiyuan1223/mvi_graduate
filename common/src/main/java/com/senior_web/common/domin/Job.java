package com.senior_web.common.domin;

import java.io.Serializable;
import java.math.BigDecimal;

public class Job implements Serializable {
    private Integer id;
    private String position;
    private String area;
    private String company;
    private BigDecimal salary;



    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getPosition() {
        return position;
    }

    public void setPosition(String position) {
        this.position = position;
    }

    public String getArea() {
        return area;
    }

    public void setArea(String area) {
        this.area = area;
    }

    public String getCompany() {
        return company;
    }

    public void setCompany(String company) {
        this.company = company;
    }



    public BigDecimal getSalary(){
        return salary;
    }

    public void setSalary(BigDecimal salary) {
        this.salary = salary;
    }


    @Override
    public String toString() {
        return "Job{" +
                "id=" + id +
                ", position=" + position +
                ", company='" + company + '\'' +
                ", are='" + area + '\'' +
                ", salary='" + salary + '\'' +
                '}';
    }


}
