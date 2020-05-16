// 直方图
var myChart = echarts.init(document.getElementById('histogram'));
var histogram_option = {
    title: {
        text: '各地区职位数量统计'
    },
    tooltip: {},
    legend: {
        data:['销量']
    },
    toolbox: {
        feature: {
            //saveAsImage: {}//保存图表
        }
    },
    xAxis: {
        data: ["上海","广州","深圳","北京","武汉","成都","西安"]
    },
    yAxis: {},
    series: [{
        name: '职位数量',
        type: 'bar',
        data: [46787, 45099, 38738, 34425, 21130,14855,9525]
    }]
};
myChart.setOption(histogram_option);



// 饼图
var myChart = echarts.init(document.getElementById('pie_chart'));
var pie_option = {
    title:{
        text: "不同薪资水平占比"
    },
    series : [
        {
            name: '薪资水平',
            type: 'pie',
            radius: '55%',
            //roseType: 'angle',
            data:[
                {value:60505, name:'10000-15000'},
                {value:15476, name:'15000-20000'},
                {value:9881, name:'20000-30000'},
                {value:1850, name:'30000-40000'},
                {value:556, name:'40000-50000'},
                {value:573, name:'50000+'},
                {value:227530, name:'5000-10000'},
                {value:63127, name:'<5000'},
            ]
        }
    ]
};
// 使用刚指定的配置项和数据显示图表。
myChart.setOption(pie_option);

//折线图
var myChart = echarts.init(document.getElementById('line_chart'));
line_option = {
    title: {
        text: '平均薪资在不同地区的分布'
    },
    tooltip: {
        trigger: 'axis'
    },
    legend: {
        data:['Java工程师','人事专员','会计','办公室文员','销售代表']
    },
    grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
    },
    xAxis: {
        type: 'category',
        boundaryGap: false,
        data: ['上海','北京','广州','深圳','杭州',]
    },
    yAxis: {
        type: 'value'
    },
    series: [
        {
            name:'Java工程师',
            type:'line',
            stack: '',
            data:[14882, 14612, 10996, 14137, 14706]
        },
        {
            name:'人事专员',
            type:'line',
            stack: '',
            data:[6550, 6434, 5281, 5850, 5764]
        },
        {
            name:'会计',
            type:'line',
            stack: '',
            data:[7327, 7531, 5575, 6459, 6478]
        },
        {
            name:'办公室文员',
            type:'line',
            stack: '',
            data:[5067, 5481, 4074, 11833, 4706]
        },
        {
            name:'销售代表',
            type:'line',
            stack: '',
            data:[8825, 9164, 8123, 9172, 9465]
        }
    ]
};
myChart.setOption(line_option);