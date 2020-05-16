"use strict";
function getTotal(){
	$.ajax({
		type : "GET",
		url : "/getJobTotal",
		dataType : "json",
		success : function(data) {
			console.log(data);

		}
	});
}
function initUI(page, pageSize) {
	$.ajax({
		type: "POST",
		url: "/getJobByPage?page=" + page + "&pageSize=" + pageSize,
		dataType: "json",
		success: function (data) {
			// 拼接html(这个部分根据自己的需求去实现)
			var list = data.jobs;
			var htm = '';
			console.log(page, pageSize, list.length)
			for (var i = 0; i < list.length; i++) {
				htm+= "<ol class=\"rightList\">"
				htm += "<li>" +
					"<span>ID:" + list[i].id + "</span>" +
					"<span><i class=\"iconfont icon-arrow-up-solid\"></i></span>" +
					"</li>"
				htm += "<li>" +
					"<span>职位:" + list[i].position + "</span>" +
					"<span><i class=\"iconfont icon-arrow-down-solid\"></i></span>" +
					"</li>"
				htm += "<li>" +
					"<span>地区:" + list[i].area + "</span>" +
					"<span><i class=\"iconfont icon-minus\"></i></span>" +
					"</li>"
				htm += "<li>\n" +
					"<span>公司:" + list[i].company + "</span>" +
					"<span><i class=\"iconfont icon-arrow-up-solid\"></i></span>" +
					"</li>"
				htm += "<li>" +
					"<span>薪水:" + list[i].salary + "</span>" +
					"<span><i class=\"iconfont icon-minus\">" +
					"</li>"

				htm+= "</ol><br>"

				console.log("in for loop")
				document.getElementsByClassName('data-list')[0].innerHTML = htm;
			}
		}
	})
	pagination({
		cur: page,
		total: 190445,
		len: 5,
		targetId: 'pagination',
		callback: function(total) {
			var oPages = document.getElementsByClassName('page-index');
			for(var i = 0; i < oPages.length; i++) {
				oPages[i].onclick=function() {
					initUI(this.getAttribute('data-index'), 5);
				}
			}
			var goPage = document.getElementById('go-search');
			goPage.onclick = function() {
				var index = document.getElementById('yeshu').value;
				if(!index || (+index > total) || (+index < 1)) {
					return;
				}
				initUI(index, 5);
			}
		}
	});
}
initUI(1,5);
console.log(total)



