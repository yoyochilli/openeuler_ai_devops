# openEuler 软件包管理

## DNF 基础操作
DNF 是 openEuler 默认的软件包管理器。
要安装软件，请使用命令：`dnf install <package_name>`。

## 常见问题排查
### 依赖冲突报错
当你在安装 nginx 时遇到依赖冲突（Dependency Resolution Errors），通常是因为本地缓存过期。
解决方法：首先运行 `dnf clean all` 清除缓存，然后再运行 `dnf makecache` 重新生成缓存。