#!/usr/bin/env python3
"""
分布式Master服务器
"""
import os
import random
import threading
import time
import uvicorn
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import aiohttp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .models import WorkerRegistrationData, CreateInput
from .utils import extract_tool_names_from_config


class DistributedMasterServer:
    """分布式Master服务器，支持动态Worker注册和工具发现"""
    
    def __init__(self, host: str, port: int, tools_config: List[Dict] = None, log_file: str = None):
        self.host = host
        self.port = port
        self.log_file = log_file
        self.app = FastAPI(title="Distributed Master Server")
        
        # 动态工具发现
        self.available_tools = {}  # tool_name -> [worker_ids]
        self.tool_routes_created = set()  # 已创建路由的工具名称
        
        # Worker管理
        self.workers: Dict[str, Dict] = {}  # worker_id -> worker_info
        self.instance_worker_mapping = {}  # instance_id -> worker_id
        self.worker_last_heartbeat = {}  # worker_id -> timestamp
        self.health_check_thread = None
        self.stop_health_check = False
        
        # 兼容原有接口：如果提供了tools_config，预创建路由
        if tools_config:
            self.tools_config = tools_config
            self.tool_names = extract_tool_names_from_config(tools_config)
            self._log(f"🔧 Master预配置工具: {self.tool_names}")
            for tool_name in self.tool_names:
                self._create_tool_routes(tool_name)
                self.tool_routes_created.add(tool_name)
        else:
            self.tools_config = []
            self.tool_names = []
            self._log("🔧 Master启用动态工具发现模式")
        
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

    def _setup_routes(self):
        """设置Master服务器的路由"""
        self._log("🔗 分布式Master服务器设置路由...")
        
        @self.app.get("/", response_class=HTMLResponse, tags=["Master"])
        async def dashboard():
            """可视化仪表板"""
            return self._generate_dashboard_html()
        
        @self.app.get("/health", tags=["Master"])
        async def health_check():
            """健康检查端点（JSON格式）"""
            return {
                "status": "ok",
                "tools": self.tool_names,
                "registered_workers": len(self.workers),
                "workers": {
                    worker_id: {
                        "url": info["worker_url"],
                        "tools": info["tools"],
                        "last_heartbeat": self.worker_last_heartbeat.get(worker_id, "never"),
                        "status": "alive" if self._is_worker_healthy(worker_id) else "dead"
                    }
                    for worker_id, info in self.workers.items()
                },
                "instance_mappings": len(self.instance_worker_mapping)
            }

        @self.app.post("/register_worker", tags=["Master"])
        async def register_worker(registration_data: WorkerRegistrationData):
            """注册新的Worker"""
            worker_id = registration_data.worker_id
            worker_url = registration_data.worker_url
            
            # 验证Worker是否可达
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{worker_url}/health", timeout=10) as response:
                        if response.status != 200:
                            return {"success": False, "error": f"Worker health check failed: {response.status}"}
            except Exception as e:
                return {"success": False, "error": f"Cannot reach worker: {e}"}
            
            # 注册Worker
            self.workers[worker_id] = {
                "worker_url": worker_url,
                "tools": registration_data.tools,
                "host_info": registration_data.host_info,
                "registered_at": datetime.now().isoformat()
            }
            self.worker_last_heartbeat[worker_id] = time.time()
            
            # 动态工具发现：为新工具创建路由
            new_tools = []
            for tool_name in registration_data.tools:
                # 添加工具到可用工具列表
                if tool_name not in self.available_tools:
                    self.available_tools[tool_name] = []
                self.available_tools[tool_name].append(worker_id)
                
                # 如果是新工具且未创建路由，创建路由
                if tool_name not in self.tool_routes_created:
                    self._create_tool_routes(tool_name)
                    self.tool_routes_created.add(tool_name)
                    new_tools.append(tool_name)
                    # 更新全局工具名称列表
                    if tool_name not in self.tool_names:
                        self.tool_names.append(tool_name)
            
            self._log(f"✅ 新Worker注册成功: {worker_id} at {worker_url}")
            self._log(f"   工具: {registration_data.tools}")
            if new_tools:
                self._log(f"   🔧 新发现工具: {new_tools} (已自动创建路由)")
            self._log(f"   主机信息: {registration_data.host_info}")
            
            return {"success": True, "message": f"Worker {worker_id} registered successfully"}

        @self.app.post("/worker_heartbeat", tags=["Master"])
        async def worker_heartbeat(heartbeat_data: dict):
            """接收Worker心跳"""
            worker_id = heartbeat_data.get("worker_id")
            if worker_id in self.workers:
                self.worker_last_heartbeat[worker_id] = time.time()
                # 更新Worker状态信息
                if "instance_count" in heartbeat_data:
                    self.workers[worker_id]["instance_count"] = heartbeat_data["instance_count"]
                return {"success": True}
            return {"success": False, "error": "Worker not registered"}

        @self.app.post("/unregister_worker", tags=["Master"])
        async def unregister_worker(data: dict):
            """注销Worker"""
            worker_id = data.get("worker_id")
            if worker_id in self.workers:
                # 清理该Worker上的所有实例映射
                instances_to_remove = [
                    instance_id for instance_id, mapped_worker_id in self.instance_worker_mapping.items()
                    if mapped_worker_id == worker_id
                ]
                for instance_id in instances_to_remove:
                    del self.instance_worker_mapping[instance_id]
                
                # 清理工具映射
                worker_tools = self.workers[worker_id].get("tools", [])
                for tool_name in worker_tools:
                    if tool_name in self.available_tools:
                        if worker_id in self.available_tools[tool_name]:
                            self.available_tools[tool_name].remove(worker_id)
                        # 如果没有Worker提供此工具，可以考虑移除路由（可选）
                        if not self.available_tools[tool_name]:
                            self._log(f"⚠️  工具 {tool_name} 无可用Worker")
                
                # 移除Worker
                del self.workers[worker_id]
                if worker_id in self.worker_last_heartbeat:
                    del self.worker_last_heartbeat[worker_id]
                
                self._log(f"✅ Worker注销成功: {worker_id}")
                self._log(f"   已清理工具映射: {worker_tools}")
                return {"success": True, "message": f"Worker {worker_id} unregistered"}
            return {"success": False, "error": "Worker not found"}
        
        # 为每个工具创建路由
        for tool_name in self.tool_names:
            self._create_tool_routes(tool_name)
        
        self._log("  - ✅ 分布式Master服务器路由设置完成")

    def _create_tool_routes(self, tool_name: str):
        """为单个工具创建Master路由"""

        @self.app.post(f"/{tool_name}/create", tags=[tool_name])
        async def create_endpoint(input_data: CreateInput):
            input_dict = input_data.model_dump()
            instance_id = input_dict.get("instance_id")
            
            if not instance_id:
                return {"success": False, "error": "instance_id is required"}
            
            # 选择健康的Worker（负载均衡）
            # 使用动态工具映射
            tool_workers = self.available_tools.get(tool_name, [])
            available_workers = [
                worker_id for worker_id in tool_workers
                if self._is_worker_healthy(worker_id)
            ]
            
            if not available_workers:
                return {"success": False, "error": f"No healthy workers available for tool {tool_name}"}
            
            # 简单的负载均衡：先找出最少实例数
            min_instance_count = min(self._get_worker_instance_count(worker_id) for worker_id in available_workers)
            # self._log(f"[MASTER] {tool_name} 最少实例数: {min_instance_count}")
            # 从实例数最少的Workers中随机选择一个
            worker_id = random.choice([worker_id for worker_id in available_workers if self._get_worker_instance_count(worker_id) == min_instance_count])
            
            worker_url = self.workers[worker_id]["worker_url"]
            
            # 建立映射关系
            self.instance_worker_mapping[instance_id] = worker_id
            instance_count = self._get_worker_instance_count(worker_id)
            
            self._log(f"[MASTER] {tool_name} 创建请求路由到 {worker_id} ({worker_url}) [instances: {instance_count}]")
            
            # 转发请求到选中的worker
            result = await self._forward_request(worker_url, f"/{tool_name}/create", input_dict)
            
            if not result.get("success", False):
                self._log(f"[MASTER] {tool_name} 创建请求失败: {result}")
                # 如果创建失败，清理映射
                self.instance_worker_mapping.pop(instance_id, None)
            
            return result

        @self.app.post(f"/{tool_name}/execute", tags=[tool_name])
        async def execute_endpoint(input_data: dict):
            instance_id = input_data.get("instance_id")
            
            if not instance_id:
                return {"success": False, "error": "instance_id is required"}
            
            # 根据instance_id路由到对应的worker
            worker_id = self.instance_worker_mapping.get(instance_id)
            if not worker_id or worker_id not in self.workers:
                return {"success": False, "error": f"No worker found for instance_id: {instance_id}"}
            
            if not self._is_worker_healthy(worker_id):
                return {"success": False, "error": f"Worker {worker_id} is not healthy"}
            
            worker_url = self.workers[worker_id]["worker_url"]
            instance_count = self._get_worker_instance_count(worker_id)
            all_worker_instances = {w_id: self._get_worker_instance_count(w_id) for w_id in self.workers.keys()}
            self._log(f"[MASTER] {tool_name} 执行请求路由到 {worker_id} ({worker_url}) [instances: {instance_count}] [all worker instances: {all_worker_instances}]")
            
            # 转发请求
            return await self._forward_request(worker_url, f"/{tool_name}/execute", input_data)

        @self.app.post(f"/{tool_name}/release", tags=[tool_name])
        async def release_endpoint(input_data: dict):
            instance_id = input_data.get("instance_id")
            
            if not instance_id:
                return {"success": False, "error": "instance_id is required"}
            
            # 根据instance_id路由到对应的worker
            worker_id = self.instance_worker_mapping.get(instance_id)
            if not worker_id or worker_id not in self.workers:
                # 实例映射不存在或Worker已失效，直接清理映射
                self.instance_worker_mapping.pop(instance_id, None)
                return {"success": False, "error": f"No worker found for instance_id: {instance_id}"}
            
            worker_url = self.workers[worker_id]["worker_url"]
            instance_count = self._get_worker_instance_count(worker_id)
            all_worker_instances = {w_id: self._get_worker_instance_count(w_id) for w_id in self.workers.keys()}
            self._log(f"[MASTER] {tool_name} 释放请求路由到 {worker_id} ({worker_url}) [instances: {instance_count}] [all worker instances: {all_worker_instances}]")
            
            # 转发请求
            result = await self._forward_request(worker_url, f"/{tool_name}/release", input_data)
            
            # 无论释放成功还是失败，都清理映射（防止累积）
            # 如果Worker已经不存在该实例，我们也应该清理Master的映射
            self.instance_worker_mapping.pop(instance_id, None)
            self._log(f"[MASTER] {tool_name} 实例映射已清理: {instance_id} (release result: {result.get('success', False)})")
            
            return result

        @self.app.post(f"/{tool_name}/calc_reward", tags=[tool_name])
        async def calc_reward_endpoint(input_data: dict):
            instance_id = input_data.get("instance_id")
            
            if not instance_id:
                return {"success": False, "error": "instance_id is required"}
            
            # 根据instance_id路由到对应的worker
            worker_id = self.instance_worker_mapping.get(instance_id)
            if not worker_id or worker_id not in self.workers:
                return {"success": False, "error": f"No worker found for instance_id: {instance_id}"}
            
            if not self._is_worker_healthy(worker_id):
                return {"success": False, "error": f"Worker {worker_id} is not healthy"}
            
            worker_url = self.workers[worker_id]["worker_url"]
            instance_count = self._get_worker_instance_count(worker_id)
            self._log(f"[MASTER] {tool_name} 计算奖励请求路由到 {worker_id} ({worker_url}) [instances: {instance_count}]")
            
            # 转发请求
            return await self._forward_request(worker_url, f"/{tool_name}/calc_reward", input_data)

    def _get_worker_instance_count(self, worker_id: str) -> int:
        """获取指定worker映射的instance数量"""
        return sum(1 for mapped_worker_id in self.instance_worker_mapping.values() 
                  if mapped_worker_id == worker_id)

    async def _forward_request(self, worker_url: str, path: str, data: dict) -> dict:
        """转发请求到指定的worker"""
        full_url = f"{worker_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(full_url, json=data, timeout=None) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Worker returned {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}

    def _is_worker_healthy(self, worker_id: str, timeout: int = 60) -> bool:
        """检查Worker是否健康（基于心跳时间）"""
        if worker_id not in self.worker_last_heartbeat:
            return False
        
        last_heartbeat = self.worker_last_heartbeat[worker_id]
        return (time.time() - last_heartbeat) < timeout

    def _load_dashboard_template(self) -> str:
        """加载仪表板HTML模板"""
        template_path = Path(__file__).parent / "dashboard_template.txt"
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self._log(f"⚠️  加载仪表板模板失败: {e}")
            return "<html><body><h1>仪表板模板加载失败</h1></body></html>"
    
    def _generate_dashboard_html(self) -> str:
        """生成优雅的仪表板HTML页面"""
        # 统计信息
        total_workers = len(self.workers)
        alive_workers = sum(1 for worker_id in self.workers if self._is_worker_healthy(worker_id))
        total_instances = len(self.instance_worker_mapping)
        total_tools = len(self.tool_names)
        
        # 生成Worker卡片HTML
        workers_html = ""
        for worker_id, info in self.workers.items():
            is_healthy = self._is_worker_healthy(worker_id)
            status_class = "status-alive" if is_healthy else "status-dead"
            status_text = "🟢 在线" if is_healthy else "🔴 离线"
            
            last_heartbeat = self.worker_last_heartbeat.get(worker_id, 0)
            if last_heartbeat == 0:
                heartbeat_text = "从未"
            else:
                seconds_ago = int(time.time() - last_heartbeat)
                heartbeat_text = f"{seconds_ago}秒前"
            
            instance_count = self._get_worker_instance_count(worker_id)
            tools_list = ", ".join(info.get("tools", []))
            host_info = info.get("host_info", {})
            
            workers_html += f"""
            <div class="worker-card {status_class}">
                <div class="worker-header">
                    <h3>{worker_id}</h3>
                    <span class="status-badge">{status_text}</span>
                </div>
                <div class="worker-info">
                    <p><strong>🌐 URL:</strong> <code>{info['worker_url']}</code></p>
                    <p><strong>🔧 工具:</strong> <span class="tools-list">{tools_list}</span></p>
                    <p><strong>📦 活跃实例:</strong> <span class="instance-count">{instance_count}</span></p>
                    <p><strong>💓 最后心跳:</strong> {heartbeat_text}</p>
                    <p><strong>🖥️  主机:</strong> {host_info.get('hostname', 'N/A')} ({host_info.get('ip', 'N/A')})</p>
                    <p><strong>📅 注册时间:</strong> {info.get('registered_at', 'N/A')}</p>
                </div>
            </div>
            """
        
        if not workers_html:
            workers_html = '<div class="no-workers">暂无已注册的Worker节点</div>'
        
        # 生成工具列表HTML
        tools_html = ""
        for tool_name in sorted(self.tool_names):
            tool_workers = self.available_tools.get(tool_name, [])
            healthy_workers = [w for w in tool_workers if self._is_worker_healthy(w)]
            worker_count = len(healthy_workers)
            status_class = "tool-available" if worker_count > 0 else "tool-unavailable"
            
            tools_html += f"""
            <div class="tool-item {status_class}">
                <span class="tool-name">{tool_name}</span>
                <span class="tool-workers">{worker_count} 个Worker可用</span>
            </div>
            """
        
        if not tools_html:
            tools_html = '<div class="no-tools">暂无可用工具</div>'
        
        # 加载模板并填充数据
        template = self._load_dashboard_template()
        html = template.format(
            alive_workers=alive_workers,
            total_workers=total_workers,
            total_tools=total_tools,
            total_instances=total_instances,
            workers_html=workers_html,
            tools_html=tools_html,
            master_host=self.host,
            master_port=self.port,
            update_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html

    def start_health_monitor(self, check_interval: int = 5):
        """启动Worker健康监控"""
        def health_monitor_loop():
            while not self.stop_health_check:
                dead_workers = []
                for worker_id in list(self.workers.keys()):
                    if not self._is_worker_healthy(worker_id):
                        dead_workers.append(worker_id)
                
                # 清理死掉的Worker
                for worker_id in dead_workers:
                    self._log(f"⚠️  检测到Worker {worker_id} 死亡，正在清理...")
                    
                    # 清理实例映射
                    instances_to_remove = [
                        instance_id for instance_id, mapped_worker_id in self.instance_worker_mapping.items()
                        if mapped_worker_id == worker_id
                    ]
                    for instance_id in instances_to_remove:
                        del self.instance_worker_mapping[instance_id]
                        self._log(f"  - 清理实例映射: {instance_id}")
                    
                    # 移除Worker
                    if worker_id in self.workers:
                        del self.workers[worker_id]
                    if worker_id in self.worker_last_heartbeat:
                        del self.worker_last_heartbeat[worker_id]
                    
                    self._log(f"  - ✅ Worker {worker_id} 已清理")
                
                time.sleep(check_interval)

        self.health_check_thread = threading.Thread(target=health_monitor_loop, daemon=True)
        self.health_check_thread.start()
        self._log("✅ Worker健康监控已启动")

    def run(self):
        """启动Master服务器"""
        self._log(f"🚀 启动分布式Master服务器于 http://{self.host}:{self.port}")
        self._log(f"📖 支持工具: {self.tool_names}")
        self._log(f"🔗 等待Worker注册...")
        
        # 启动健康监控
        self.start_health_monitor()
        
        uvicorn.run(self.app, host=self.host, port=self.port, log_config=None) 