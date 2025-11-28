#!/usr/bin/env python3
"""
分布式Worker服务器
"""

import asyncio
import random
import requests
import socket
import threading
import time
import uvicorn
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
from fastapi import FastAPI

from internbootcamp.utils.load_tool_from_config import load_tool_from_config
from .models import WorkerRegistrationData, CreateInput
from .utils import get_external_ip, find_available_port, find_available_port_range, is_port_available


class DistributedWorkerServer:
    """分布式Worker服务器，可独立部署并注册到Master"""
    
    def __init__(self, tools_config: List[Dict], host: str, port: int, worker_id: str, 
                 master_url: Optional[str] = None, log_file: str = None):
        self.tools_config = tools_config
        self.host = host
        self.port = port
        self.worker_id = worker_id
        self.master_url = master_url
        self.log_file = log_file
        self.app = FastAPI(title=f"Distributed Worker Server {worker_id}")
        self.tools = {}
        self.tool_names = []
        self.is_registered = False
        self.heartbeat_thread = None
        self.stop_heartbeat = False
        self.registration_thread = None
        
        self._load_tools()
        self._setup_routes()
    
    def _log(self, message: str):
        """统一的日志记录方法"""
        log_line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(log_line)
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_line + '\n')
            except Exception as e:
                print(f"警告：写入日志文件失败: {e}")

    def _load_tools(self):
        """加载并实例化配置文件中的所有工具"""
        self._log(f"🔧 Worker {self.worker_id} 加载工具...")
        for tool_config in self.tools_config:
            try:
                _,_,tool_instance = load_tool_from_config(tool_config)
                tool_name = tool_instance.__class__.__name__
                self.tools[tool_name] = tool_instance
                self.tool_names.append(tool_name)
                self._log(f"  - ✅ Worker {self.worker_id} 已加载: {tool_name}")
            except Exception as e:
                self._log(f"  - ❌ Worker {self.worker_id} 加载工具失败 {tool_config.get('class_name', 'N/A')}: {e}")
                import traceback
                traceback.print_exc()
    
    def _setup_routes(self):
        """为所有加载的工具设置API路由"""
        self._log(f"🔗 Worker {self.worker_id} 设置路由...")
        
        @self.app.get("/health", tags=["Worker"])
        async def health_check():
            """健康检查端点"""
            return {
                "status": "ok", 
                "worker_id": self.worker_id,
                "tools": self.tool_names,
                "is_registered": self.is_registered,
                "master_url": self.master_url
            }

        @self.app.post("/register_to_master", tags=["Worker"])
        async def register_to_master():
            """手动注册到Master的端点"""
            if self.master_url:
                success = await self._register_to_master()
                return {"success": success, "registered": self.is_registered}
            return {"success": False, "error": "No master URL configured"}
        
        for tool_name, tool_instance in self.tools.items():
            self._create_tool_endpoints(tool_name, tool_instance)
        self._log(f"  - ✅ Worker {self.worker_id} 路由设置完成")

    def _create_tool_endpoints(self, tool_name: str, tool_instance: any):
        """为单个工具创建端点"""
        import traceback
        
        @self.app.post(f"/{tool_name}/create", tags=[tool_name])
        async def create_endpoint(input_data: CreateInput):
            input_dict = input_data.model_dump()
            self._log(f"[DEBUG] Worker {self.worker_id} {tool_name} 创建输入: {input_dict}")
            try:
                result = await tool_instance.create(input_dict["instance_id"], input_dict["identity"])
                self._log(f"[DEBUG] Worker {self.worker_id} {tool_name} 创建返回: {result}")
                return {"success": True, "result": result}
            except Exception as e:
                self._log(f"[DEBUG][ERROR] Worker {self.worker_id} {tool_name} 创建异常: {traceback.format_exc()}")
                return {"success": False, "error": str(e)}

        @self.app.post(f"/{tool_name}/execute", tags=[tool_name])
        async def execute_endpoint(input_data: dict):
            self._log(f"[DEBUG] Worker {self.worker_id} {tool_name} 执行输入: {input_data}")
            instance_id = input_data.pop("instance_id", None)
            try:
                output = await tool_instance.execute(instance_id=instance_id, parameters=input_data)
                self._log(f"[DEBUG] Worker {self.worker_id} {tool_name} 执行输出: {output}")
                return output
            except Exception as e:
                self._log(f"[DEBUG][ERROR] Worker {self.worker_id} {tool_name} 执行异常: {traceback.format_exc()}")
                raise e

        @self.app.post(f"/{tool_name}/release", tags=[tool_name])
        async def release_endpoint(input_data: dict):
            self._log(f"[DEBUG] Worker {self.worker_id} {tool_name} 释放输入: {input_data}")
            instance_id = input_data.pop("instance_id", None)
            try:
                result = await tool_instance.release(instance_id)
                self._log(f"[DEBUG] Worker {self.worker_id} {tool_name} 释放返回: {result}")
                return {"success": True, "result": result}
            except Exception as e:
                self._log(f"[DEBUG][ERROR] Worker {self.worker_id} {tool_name} 释放异常: {traceback.format_exc()}")
                return {"success": False, "error": str(e)}

        @self.app.post(f"/{tool_name}/calc_reward", tags=[tool_name])
        async def calc_reward_endpoint(input_data: dict):
            self._log(f"[DEBUG] Worker {self.worker_id} {tool_name} 计算奖励输入: {input_data}")
            instance_id = input_data.pop("instance_id", None)
            try:
                result = await tool_instance.calc_reward(instance_id)
                self._log(f"[DEBUG] Worker {self.worker_id} {tool_name} 计算奖励返回: {result}")
                return result
            except Exception as e:
                self._log(f"[DEBUG][ERROR] Worker {self.worker_id} {tool_name} 计算奖励异常: {traceback.format_exc()}")
                return {"success": False, "error": str(e)}

    def _prepare_registration_data(self) -> WorkerRegistrationData:
        """准备注册数据"""
        worker_url = f"http://{get_external_ip()}:{self.port}"
        return WorkerRegistrationData(
            worker_id=self.worker_id,
            worker_url=worker_url,
            tools=self.tool_names,
            host_info={
                "hostname": socket.gethostname(),
                "ip": get_external_ip(),
                "port": self.port
            }
        )

    async def _register_to_master(self) -> bool:
        """向Master服务器注册自己"""
        if not self.master_url:
            return False
            
        registration_data = self._prepare_registration_data()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.master_url}/register_worker",
                    json=registration_data.model_dump()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success"):
                            self.is_registered = True
                            self._log(f"✅ Worker {self.worker_id} 成功注册到Master: {self.master_url}")
                            return True
                    
                    error_text = await response.text()
                    self._log(f"❌ Worker注册失败: {response.status} - {error_text}")
                    return False
        except Exception as e:
            self._log(f"❌ Worker注册异常: {e}")
            return False

    def _start_registration_process(self):
        """启动注册流程（在服务器启动后调用）"""
        if not self.master_url:
            self._log(f"⚠️  未配置master_url，跳过注册")
            return
            
        self._log(f"🔗 准备注册到Master: {self.master_url}")
        
        def register_async():
            time.sleep(2)  # 等待服务器启动完成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                success = loop.run_until_complete(self._register_to_master())
                if success:
                    # 启动心跳
                    self.start_heartbeat()
                else:
                    self._log(f"❌ Worker {self.worker_id} 注册失败")
            finally:
                loop.close()
        
        self.registration_thread = threading.Thread(target=register_async, daemon=True)
        self.registration_thread.start()

    def start_heartbeat(self, interval: int = 30):
        """启动心跳线程"""
        def heartbeat_loop():
            while not self.stop_heartbeat:
                if self.master_url and self.is_registered:
                    try:
                        # 发送心跳到Master
                        response = requests.post(
                            f"{self.master_url}/worker_heartbeat",
                            json={
                                "worker_id": self.worker_id,
                                "status": "alive",
                            },
                            timeout=5
                        )
                        if response.status_code != 200:
                            self._log(f"⚠️  心跳失败: {response.status_code}")
                    except Exception as e:
                        self._log(f"⚠️  心跳异常: {e}")
                
                time.sleep(interval)

        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def run(self):
        """启动Worker服务器"""
        # self._log(f"🚀 启动分布式Worker服务器 {self.worker_id} 于 http://{self.host}:{self.port}")
        # self._log(f"📖 已注册工具: {self.tool_names}")
        
        # 改进的启动逻辑，支持多种端口分配策略
        retry = 0
        max_retries = 20  # 增加最大重试次数
        
        while retry < max_retries:
            try:
                # 策略1：优先使用基于worker_id的端口范围分配
                if is_port_available(self.host, self.port):
                    self._log(f"✅ Worker {self.worker_id} 使用端口: {self.port}")
                elif retry < 10:
                    try:
                        self.port = find_available_port_range(self.host, self.worker_id, self.port)
                        self._log(f"✅ Worker {self.worker_id} 使用范围分配端口: {self.port}")
                    except Exception as e:
                        # 如果范围分配失败，回退到传统方法
                        self.port = find_available_port(self.host, self.port, randomize=True)
                        self._log(f"✅ Worker {self.worker_id} 使用随机分配端口: {self.port}")
                else:
                    # 策略2：随机端口分配（减少冲突）
                    base_port = self.port + random.randint(100, 2000)
                    self.port = find_available_port(self.host, base_port, base_port + 1000, randomize=True)
                    self._log(f"✅ Worker {self.worker_id} 使用随机高位端口: {self.port}")
                
                
                self._log(f"✅ Worker {self.worker_id} 服务器启动成功，监听端口: {self.port}")
                
                # 在服务器启动后启动注册流程
                self._start_registration_process()
                
                uvicorn.run(self.app, host=self.host, port=self.port, log_config=None)
                break
                
            except Exception as e:
                self._log(f"❌ Worker {self.worker_id} 服务器启动失败 (尝试 {retry + 1}/{max_retries}): {e}")
                retry += 1
                
                # 指数退避策略，添加随机延迟避免雷鸣群
                if retry < max_retries:
                    delay = min(2 ** retry + random.uniform(0, 1), 10)  # 最大延迟10秒
                    self._log(f"⏰ Worker {self.worker_id} 等待 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
        
        if retry >= max_retries:
            error_msg = f"❌ Worker {self.worker_id} 服务器启动失败，重试次数已达上限 ({max_retries})"
            self._log(error_msg)
            raise RuntimeError(error_msg)