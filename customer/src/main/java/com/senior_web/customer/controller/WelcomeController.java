package com.senior_web.customer.controller;


import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/")
public class WelcomeController {
    @RequestMapping("login")
    public String sayHello() {
        return "login";
    }

    @RequestMapping("charts")
    public String chartjs() {
        return "charts";
    }
    @RequestMapping("index")
    public String index() {
        return "index";
    }
    @RequestMapping("page")
    public String page() {
        return "page";
    }
    @RequestMapping("geo")
    public String geo() {
        return "geo";
    }
    @RequestMapping("uploadFile")
    public String uploadFile() {
        return "uploadFile";
    }

}




