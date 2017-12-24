---
layout:       post
title:        "博客的搭建"
subtitle:     "------记录搭建博客的流程"
date:         2017-12-23 22:00:00
author:       "Jelliy"
header-img:   "img/home-bg-ai2.jpeg"
header-mask:  0.3
catalog:      true
multilingual: false
tags:
    - Linux
    - Blog
    - Notes
---


## 平台的选择

> Github Pages是Github 的静态页面托管服务，利用Jekyll这一类静态页面生成工具，可以简单快捷地搭建自己的网站。

搭建的主要步骤:

* 创建一个存储库
* git客户端
* 克隆存储库
* 添加一个index.html文件
* 添加，提交和推送
* 启动浏览器并转到https://username.github.io

具体详情见官方网站 [GitHub Pages](https://pages.github.com/) 
也可以参考github的帮助文档[GitHub Help](https://help.github.com/)

此外Github Pages有一些限制：

* 仓库存储的所有文件不能超过 1 GB
* 页面的带宽限制是低于每月 100 GB 或是每月 100,000 次请求。
* 每小时最多只能部署 10 个静态网站。


## 博客模板选择

github上模板有很多，考虑到中文字体排版和目录的需求选择了[黄玄的博客](https://github.com/Huxpro/huxpro.github.io)作为模板。

其它的一些模板

> * https://github.com/poole/lanyon
> * https://github.com/daattali/beautiful-jekyll
> * https://github.com/barryclark/jekyll-now
> * https://github.com/renyuanz/leonids

fork之后，需要更改的内容网上教程很多，readme中也有介绍。有点需要注意的是md文件开始的yaml文件头**multilingual**必须有，否则catalog不能加载出来，浏览器控制台(F12)会报错。以前没接触过网页，这原因找了好久。


## MarkDown

使用Mardown写博客，并把文件放入_post下。
用markdown可以避开HTML，转而使用更加直观的Markdown语法。如果不熟悉Markdown语法也没关系，可以参见这份[Markdown语法说明](http://wowubuntu.com/markdown/)

[Cmd Markdown 在线版本](https://www.zybuluo.com/mdeditor)

## Jekyll

> Jekyll是一种简单的、适用于博客的、静态网站生成引擎。它使用一个模板目录作为网站布局的基础框架，支持Markdown、Textile等标记语言的解析，提供了模板、变量、插件等功能，最终生成一个完整的静态Web站点。说白了就是，只要安装Jekyll的规范和结构，不用写html，就可以生成网站。

jekyll本地环境搭建,见官网https://jekyllrb.com/docs/quickstart/

本地运行jekyll服务: jekyll serve
在浏览器打开http://127.0.0.1:4000，可以预览效果

## Git的常用命令

> Git是一种分布式版本控制系统，由大名鼎鼎的Linux操作系统创建人Linus Torvalds编写，当时的目的是用来管理Linux内核的源码。
> Github是全球知名的使用Git系统的一个免费远程仓库（Repository），倡导开源理念，若要将自己的代码仓库私有则需付费。

4.Git基本配置
配置本机git的两个重要信息，user.name和user.email,中终端输入如下命令即可设置
git config --global user.name "Your Name"
git config --global user.email "email@example.com"

克隆代码

1. git clone <版本库的网址> <本地目录名>

将服务器代码更新本地

1. git status（查看本地分支文件信息，确保更新时不产生冲突）
2. git checkout -- [file name] (若文件有修改，可以还原到最初状态; 若文件需要更新到服务器上，应该先merge到服务器，再更新到本地)
3. git branch (查看当前分支情况)
4. git checkout [remote branch] (若分支为本地分支，则需切换到服务器的远程分支)
5. git pull

将本地代码更新服务器

1. git add.
2. git commit -am "修改说明"
3. git push origin master



## 工具

Pandoc

## 参考

[如何在github上搭建自己的博客](https://www.cnblogs.com/EX32/p/4479712.html)
[通过GitHub Pages建立个人站点（详细步骤](https://www.cnblogs.com/purediy/archive/2013/03/07/2948892.html)
[Ubuntu14.04+Jekyll+Github Pages搭建静态博客](https://www.cnblogs.com/mo-wang/p/5115266.html)
[Jekyll 博客搭建笔记 1](https://segmentfault.com/a/1190000011629270)

