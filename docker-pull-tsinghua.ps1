# 使用清华大学镜像源拉取 Docker 镜像

Write-Host "配置 Docker 使用清华大学镜像加速..." -ForegroundColor Green
Write-Host "请在 Docker Desktop 设置中手动配置镜像加速器：" -ForegroundColor Yellow
Write-Host "Settings -> Docker Engine -> 添加以下配置：" -ForegroundColor Yellow
Write-Host ""
Write-Host '  "registry-mirrors": [' -ForegroundColor Cyan
Write-Host '    "https://docker.mirrors.ustc.edu.cn",' -ForegroundColor Cyan
Write-Host '    "https://hub-mirror.c.163.com",' -ForegroundColor Cyan
Write-Host '    "https://mirror.baidubce.com"' -ForegroundColor Cyan
Write-Host '  ]' -ForegroundColor Cyan
Write-Host ""
Write-Host "按任意键继续..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host "`n拉取 PyTorch 基础镜像（使用与其他成功项目一致的版本）..." -ForegroundColor Green
Write-Host "注意：请先在 Docker Desktop 中配置镜像加速器！" -ForegroundColor Yellow
docker pull pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n基础镜像拉取成功！" -ForegroundColor Green
    Write-Host "现在可以构建项目镜像：" -ForegroundColor Green
    Write-Host '  docker-compose -f docker-compose.gpu.yml build' -ForegroundColor Cyan
    Write-Host "`n或直接启动：" -ForegroundColor Green
    Write-Host '  docker-compose -f docker-compose.gpu.yml up -d' -ForegroundColor Cyan
} else {
    Write-Host "`n镜像拉取失败，请检查：" -ForegroundColor Red
    Write-Host "1. Docker Desktop 是否正在运行" -ForegroundColor Yellow
    Write-Host "2. 是否已配置镜像加速器" -ForegroundColor Yellow
    Write-Host "3. 网络连接是否正常" -ForegroundColor Yellow
}


