catalog显示标题是大小一样，先修改成只显示一级和二级标题。
footer.html文件220行
a = P.find('h1,h2,h3,h4,h5,h6');修改为
a = P.find('h1,h2');
