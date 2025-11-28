import os
from re import S
import openai
import yaml
import importlib
import json
import asyncio
import httpx
import csv

from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Callable, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from internbootcamp.utils.format_time_now import format_time_now
from internbootcamp.utils.load_tool_from_config import load_tool_from_config
from internbootcamp.utils.load_interaction_from_config import load_interaction_from_config
from internbootcamp.utils.load_class_from_str import load_class_from_string
from internbootcamp.src.base_tool import BaseTool
from internbootcamp.src.base_interaction import BaseInteraction
from internbootcamp.src.base_reward_calculator import BaseRewardCalculator
import jsonlines
from PIL import Image
from internbootcamp.src.img2base64 import encode_image_file_to_base64

def load_dataset(dataset_path, dataset=None):
    """
    加载数据集，支持 JSON、JSONL 和 Parquet 文件格式，并始终返回 list。
    
    参数:
        dataset_path (str): 数据集文件路径。
        dataset (list, optional): 如果提供了 dataset，则直接返回。
    
    返回:
        list: 加载的数据集，统一为列表格式。
    """
    if dataset_path and not dataset:
        # 获取文件扩展名
        _, ext = os.path.splitext(dataset_path)
        ext = ext.lower()  # 统一转换为小写
        
        if ext == ".json":
            # 加载 JSON 文件
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 确保返回的是列表
                dataset = data if isinstance(data, list) else [data]
        
        elif ext == ".jsonl":
            # 加载 JSONL 文件
            dataset = []
            with jsonlines.open(dataset_path) as reader:
                for line in reader:
                    dataset.append(line)
        
        elif ext == ".parquet":
            # 加载 Parquet 文件并转换为列表
            df = pd.read_parquet(dataset_path)
            dataset = df.to_dict(orient='records')  # 转换为字典列表
            
            # 处理 parquet 加载后的数据类型问题
            for item in dataset:
                # 确保 messages 和 prompt 字段是 Python 列表而不是 numpy 数组
                if 'messages' in item and hasattr(item['messages'], 'tolist'):
                    item['messages'] = item['messages'].tolist()
                elif 'messages' in item and not isinstance(item['messages'], list):
                    item['messages'] = list(item['messages'])
                
                if 'prompt' in item and hasattr(item['prompt'], 'tolist'):
                    item['prompt'] = item['prompt'].tolist()
                elif 'prompt' in item and not isinstance(item['prompt'], list):
                    item['prompt'] = list(item['prompt'])
        
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    return dataset

class BaseEvaluator:
    def __init__(
        self,
        api_key: str,
        reward_calculator: BaseRewardCalculator,
        api_url: str = None,
        api_model: str = None,
        api_extra_headers: dict = None,
        api_extra_params: dict = None,
        verify_correction_kwargs: dict = None,
        max_assistant_turns: int = None,
        max_user_turns: int = None,
        tokenizer_path = None,
        **kwargs,
        ):
        self.api_model = api_model
        self.api_extra_headers = api_extra_headers or {}
        self.api_extra_params = api_extra_params or {}
        self.verify_correction_kwargs = verify_correction_kwargs or {}
        self.max_assistant_turns = max_assistant_turns
        self.max_user_turns = max_user_turns
        self.client = openai.AsyncOpenAI(base_url=api_url, api_key=api_key, default_headers=api_extra_headers, http_client=httpx.AsyncClient(verify=False))
        self.bootcamp_registry: Dict[str, dict] = {}
        self.reward_calculator = reward_calculator
        self.tokenizer_path = tokenizer_path
        self.tokenizer = self._get_tokenizer()
        
    def _get_tokenizer(self):
        if not self.tokenizer_path:
            return None
        try:
            default_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, 
                trust_remote_code=True
            )
            return default_tokenizer
        except Exception as e:
            print(f"[WARNING] 无法加载 tokenizer: {e}")
            return None

    def _build_payload(self, input_data: dict) -> dict:
        messages = input_data["messages"]
        # 使用 input_data 中的 tools
        tools = input_data.get("tools")
        tool_choice = input_data.get("tool_choice", "auto")

        payload = {
            "model": self.api_model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        # 添加额外的模型参数
        if self.api_extra_params:
            payload.update(self.api_extra_params)

        return payload
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
        before_sleep=lambda retry_state: print(f"重试中... 第{retry_state.attempt_number}次尝试失败: \n{retry_state.outcome.exception()}")
        )
    async def _call_api(self, payload: dict) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(**payload)
        except Exception as e:
            # print("Error happened when processing playload:")
            # print(payload)
            raise e
        # print("DEBUG response", response)
        response_dict = response.model_dump()
        # 提取 token usage 信息
        usage = response_dict.get("usage", {})
        return response_dict, usage
    
    def _load_tools_from_yaml(self, yaml_path: str) -> Tuple[List[Dict], Dict[str, Dict[str, Any]]]:
        """
        从 YAML 文件加载工具配置，构建 tools 和 tool_registry
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        tools_config = config.get("tools", [])
        tool_schemas = []
        tool_instances = {}
        for tool_cfg in tools_config:
            try:
                func_name, schema, tool_instance = load_tool_from_config(tool_cfg)
                tool_instances[func_name] = {
                    "instance": tool_instance
                }
                tool_schemas.append(schema)
                # print(f"✅ 已加载工具: {func_name}", end=';')
            except Exception as e:
                print(f"❌ 加载工具失败: {e}")
                import traceback
                traceback.print_exc()
        return tool_schemas, tool_instances

    def _load_interaction_from_yaml(self, yaml_path: str) -> BaseInteraction:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        interaction_config = config.get("interaction", [])
        if len(interaction_config) > 1:
            raise ValueError("Interaction config should only contain one interaction")
        interaction_instance = load_interaction_from_config(interaction_config[0])
        return interaction_instance
    
    def _messages_to_context(self, messages: List[Dict[str, Any]], tools: List[Dict] = []) -> str:
        """
        将messages列表转换为完整的上下文字符串
        
        优先使用 tokenizer.apply_chat_template，如果没有 tokenizer 则使用自定义格式化
        
        Args:
            messages: 消息列表，每个消息包含role和content字段
            
        Returns:
            str: 完整的对话上下文字符串
        """
        # 优先使用 transformers 的 apply_chat_template
        tokenizer_to_use = self.tokenizer
        if tokenizer_to_use is not None:
            try:
                # 处理 reasoning_content，将其合并到 content 中
                processed_messages = []
                for message in messages:
                    processed_message = message.copy()
                    # if "reasoning_content" in message:
                    #     reasoning_content = message["reasoning_content"]
                    #     content = message.get("content", "")
                    #     processed_message["content"] = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
                    #     # 移除 reasoning_content 字段，因为 apply_chat_template 不识别它
                    #     processed_message.pop("reasoning_content", None)
                    processed_messages.append(processed_message)
                
                # 使用 apply_chat_template 转换，不添加生成提示，不进行分词
                return tokenizer_to_use.apply_chat_template(
                    processed_messages,
                    tools=tools, 
                    add_generation_prompt=False, 
                    tokenize=False
                )
            except Exception as e:
                print(f"[WARNING] apply_chat_template 失败，使用回退方案: {e}")
                # 如果失败，继续使用自定义格式化
        
        # 回退方案：自定义格式化
        context_parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # 处理reasoning_content（如果存在）
            if "reasoning_content" in message:
                reasoning_content = message["reasoning_content"]
                if reasoning_content is not None and reasoning_content.strip() != "":
                    content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            
            # 处理工具调用（tool_calls）
            if role == "assistant" and "tool_calls" in message and message["tool_calls"]:
                tool_calls_content = ""
                for tool_call in message["tool_calls"]:
                    function_name = tool_call.get("function", {}).get("name", "")
                    arguments = tool_call.get("function", {}).get("arguments", "")
                    tool_calls_content += f"Function: {function_name}\nArguments: {arguments}\n\n"
                
                # 将工具调用内容用<tool_call>包裹
                if tool_calls_content:
                    content = f"{content}\n<tool_call>\n{tool_calls_content.strip()}</tool_call>"
            
            # 根据角色格式化消息
            if role == "user":
                context_parts.append(f"User:\n{content}\n")
            elif role == "assistant":
                context_parts.append(f"Assistant:\n{content}\n")
            elif role == "system":
                context_parts.append(f"System:\n{content}\n")
            elif role == "tool":
                # 工具响应已经用<tool_response>包裹了
                context_parts.append(f"<tool_response>\n{content}\n</tool_response>\n")
            else:
                # 其他角色
                context_parts.append(f"{role.capitalize()}:\n{content}\n")
        
        return "".join(context_parts)

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict],
        context_instance_id_dict: Optional[Dict[str, str]] = None,
        sample_extra_info: Dict[str, Any] = None,
        tool_instances: Dict[str, Dict[str, Any]] = None,
        ) -> Tuple[str,List[Dict[str, Any]],float,dict,float]:
        """
        执行工具调用，并支持动态传入额外参数。
        Args:
            tool_calls (List[Dict]): 工具调用列表。
            sample_extra_info (Dict[str, Any]): 样本中的额外信息（如 create_kwargs 等）。
        Returns:
            List[Dict[str, Any]]: 工具调用结果。
        """
        tool_messages = []
        sample_extra_info = sample_extra_info or {}
        tool_instances = tool_instances or getattr(self, 'tool_instances', {})
        tool_reward = 0.0
        tool_metrics = {}
        tool_cumulative_reward = 0.0
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            if tool_name not in tool_instances:
                content = f"Error: 工具 '{tool_name}' 未注册。"
            else:
                try:
                    args = json.loads(arguments)
                    # 获取工具实例及其额外参数
                    tool_instance = tool_instances[tool_name]["instance"]

                    # 动态更新 create_kwargs
                    if "tools_kwargs" in sample_extra_info and tool_name in sample_extra_info["tools_kwargs"]:
                        create_kwargs = sample_extra_info["tools_kwargs"][tool_name].get("create_kwargs", {})
                    else:
                        create_kwargs = {}

                    # 调用工具的 create 和 execute 方法
                    create_result = await tool_instance.create(context_instance_id_dict[tool_name], **create_kwargs)
                    # 兼容返回一个值或两个值的情况
                    if isinstance(create_result, tuple):
                        current_instance_id, current_tool_create_response = create_result
                    else:
                        current_instance_id = create_result
                        current_tool_create_response = None
                    context_instance_id_dict[tool_name] = current_instance_id
                    tool_result = await tool_instance.execute(current_instance_id, args)
                    # 计算工具累计奖励
                    tool_cumulative_reward = await tool_instance.calc_reward(current_instance_id)
                    tool_response, tool_reward, tool_metrics = tool_result
                    content = str(tool_response)
                except Exception as e:
                    # import traceback
                    # traceback.print_exc()
                    content = f"Error calling {tool_name}: {str(e)}"
            tool_messages.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call["id"]
            })
        return context_instance_id_dict,tool_messages,tool_reward,tool_metrics,tool_cumulative_reward

    async def _evaluate_one(
        self,
        input_data: dict,
        ) -> dict:
        
        data_source = input_data.get("data_source", None)
        
        
        if self.bootcamp_registry:
            # 根据 data_source 获取对应的组件配置
            if data_source and data_source in self.bootcamp_registry:
                config = self.bootcamp_registry[data_source]
                tool_schemas = config["tool_schemas"]
                tool_instances = config["tool_instances"]
                interaction_instance = config["interaction_instance"]
                reward_calculator = config["reward_calculator_class"]
            else:
                print(f"❌ 未找到数据源: {data_source}")
                return None
        else:
            if not self.reward_calculator:
                raise ValueError("必须提供 bootcamp_registry 或 reward_calculator")
            tool_schemas = self.tool_schemas
            tool_instances = self.tool_instances
            interaction_instance = self.interaction
            reward_calculator = self.reward_calculator
        
        # 兼容 prompt/messages 字段
        if "messages" in input_data:
            messages = input_data["messages"].copy()
        elif "prompt" in input_data:
            messages = input_data["prompt"].copy()
        else:
            raise ValueError("输入数据必须包含 'messages' 或 'prompt' 字段")

        needed_tools = []
        extra_info = input_data.get("extra_info", {})
        if extra_info.get("need_tools_kwargs") and "tools_kwargs" in extra_info and tool_schemas:
            # 只选择需要的工具
            needed_tool_names = set(extra_info["tools_kwargs"].keys())
            needed_tools = [tool for tool in tool_schemas if tool["function"]["name"] in needed_tool_names]
        if 'image' in input_data and input_data['image']:
            prompt = messages[0]["content"]
            image_path_list = input_data["image"]
            
            content_list = [{"type": "text", "text": prompt}]
            
            for image_item in image_path_list:
                image_base64 = encode_image_file_to_base64(image_item)
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                )

            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]
        payload = self._build_payload({
            "messages": messages,
            "tools": needed_tools,
            "tool_choice": "auto"
        })
        # print("DEBUG payload", payload)
        all_payloads = [payload]
        try:
            turn_record = {}
            context_instance_id_dict = {}
            # 初始化 token 统计
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            if tool_instances:
                for tool_name in tool_instances:
                    context_instance_id_dict[tool_name] = None
            if interaction_instance:
                if "interaction_kwargs" in input_data["extra_info"]:
                    interaction_instance_id = await interaction_instance.start_interaction(identity=input_data["extra_info"]["interaction_kwargs"]["identity"])
                else:
                     interaction_instance_id = await interaction_instance.start_interaction()
            else:
                interaction_instance_id = None
            # 循环控制逻辑：基于assistant和user轮次
            assistant_turn_count = 0
            user_turn_count = 0
            interaction_turn = 0
            prompt_tokens = None
            global_seq_tokens = 0
            while self.max_user_turns is None or user_turn_count < self.max_user_turns:
                current_assistant_turn_count = 0
                current_tool_calls_executed = 0
                while self.max_user_turns is None or user_turn_count < self.max_user_turns:
                    # print("DEBUG payload", payload)
                    raw_response, usage = await self._call_api(payload)
                    if prompt_tokens == None:
                        prompt_tokens = usage.get("prompt_tokens", 0)

                    # 累计 token 消耗
                    if not prompt_tokens:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)
                    global_seq_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                    message = raw_response["choices"][0]["message"]
                    
                    # def message_transform(message):
                    #     message_allowed_keys_set = set(['role', 'content', 'tool_calls', 'function_call'])
                    #     think_format = "<think>\n{}\n</think>\n\n"
                    #     if message.get('reasoning_content'):
                    #         if not message['content']:
                    #             message['content'] = ''
                    #         message['content'] = think_format.format(message['reasoning_content']) + message['content'] 
                    #     message_keys = set(message.keys())
                    #     message_keys_to_remove = message_keys - message_allowed_keys_set
                    #     for key in message_keys_to_remove:
                    #         message.pop(key)
                    #     return message
                    
                    messages.append(message)  
                    tool_calls = message.get("tool_calls", [])
                    
                    # 记录assistant轮次（每次调用API都算作一次assistant轮次）
                    
                    assistant_turn_count += 1
                    current_assistant_turn_count += 1
                    turn_record[f"interaction_turn_{interaction_turn}"] = {
                        "assistant_turns": current_assistant_turn_count,
                        "tool_calls_executed": current_tool_calls_executed
                    }
                    if self.max_assistant_turns is not None and assistant_turn_count >= self.max_assistant_turns:
                        break
                    
                    if not tool_calls:
                        # 如果assistant没有工具调用，当轮interaction结束
                        break
                    else:
                        # 记录当前轮次的工具调用次数
                        current_tool_calls_executed += len(tool_calls)
                    # 提取样本中的额外信息
                    sample_extra_info = input_data.get("extra_info", {})
                    context_instance_id_dict,tool_messages,tool_reward,tool_metrics,tool_cumulative_reward = await self._execute_tool_calls(tool_calls, context_instance_id_dict, sample_extra_info, tool_instances)
                    
                    messages.extend(tool_messages)
                    user_turn_count += len(tool_messages)
                    payload = self._build_payload({
                        "messages": messages,
                        "tools": needed_tools,
                        "tool_choice": "auto",
                    })
                    all_payloads.append(payload)
                
                # 记录当前轮次的统计信息
                turn_record[f"interaction_turn_{interaction_turn}"]["tool_calls_executed"] = current_tool_calls_executed
                if self.max_assistant_turns is not None and assistant_turn_count >= self.max_assistant_turns:
                    break

                # User响应轮次（通过interaction_instance）
                if interaction_instance:
                    should_terminate_sequence, response_content, current_turn_score, additional_data = await interaction_instance.generate_response(interaction_instance_id, messages)
                    if should_terminate_sequence:
                        break
                    else:
                        user_turn_count += 1
                        interaction_turn += 1
                        messages.append({
                            "role": "user",
                            "content": response_content
                        })
                else:
                    # 如果没有interaction_instance，直接结束
                    break

            # 释放工具
            # print("DEBUG context_instance_id_dict", context_instance_id_dict)
            for tool_name,instance_id in context_instance_id_dict.items():
                if instance_id:
                    tool_instance = tool_instances[tool_name]["instance"]
                    await tool_instance.release(instance_id)

            
            # 将整个消息上下文转换为字符串用于extract_output
            full_context = self._messages_to_context(messages,tools=needed_tools)
            if "prompt" in input_data:
                response_context = self._messages_to_context(messages[len(input_data["prompt"]):])
            elif "messages" in input_data:
                response_context = self._messages_to_context(messages[len(input_data["messages"]):])
            # print("DEBUG full_context", full_context)
            score = reward_calculator.verify_score(model_output=response_context, identity=input_data["reward_model"]["ground_truth"], **self.verify_correction_kwargs) if reward_calculator else None
            extracted_output = reward_calculator.extract_output(response_context)
            # has reached_max_turns?
            reached_max_turns = (
                (self.max_assistant_turns is not None and assistant_turn_count >= self.max_assistant_turns) or
                (self.max_user_turns is not None and user_turn_count >= self.max_user_turns)
            )
            
            return {
                "input": input_data,
                "tools": needed_tools,
                "messages": messages,
                "extracted_output": extracted_output,
                "ground_truth": input_data["reward_model"]["ground_truth"],
                "score": score,
                "reached_max_turns": reached_max_turns,
                "turn_record": turn_record,
                "success": True,
                "full_context": full_context,
                "response_context": response_context,
                "prompt_tokens": prompt_tokens,
                "global_seq_tokens": global_seq_tokens,
                "token_usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
                "evaluation_config": {
                    "model": self.api_model,
                    "api_extra_params": self.api_extra_params,
                    "api_extra_headers": self.api_extra_headers,
                    "max_assistant_turns": self.max_assistant_turns,
                    "max_user_turns": self.max_user_turns,
                }
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "input": input_data,
                "tools": needed_tools if 'tools' in locals() else [],
                "messages": messages,
                "output": None,
                "score": 0,
                "error": str(e),
                "reached_max_turns": False,
                "turn_record": turn_record,
                "success": False,
                "prompt_tokens": prompt_tokens,
                "global_seq_tokens": global_seq_tokens,
                "token_usage": {
                    "prompt_tokens": total_prompt_tokens if 'total_prompt_tokens' in locals() else 0,
                    "completion_tokens": total_completion_tokens if 'total_completion_tokens' in locals() else 0,
                    "total_tokens": total_tokens if 'total_tokens' in locals() else 0
                },
                "evaluation_config": {
                    "model": self.api_model,
                    "api_extra_params": self.api_extra_params,
                    "api_extra_headers": self.api_extra_headers,
                    "max_assistant_turns": self.max_assistant_turns,
                    "max_user_turns": self.max_user_turns,
                }
            }

    async def _evaluate_batch(
        self,
        input_list: List[dict],
        max_concurrent: int = 1,
        output_path: Optional[str] = None  # 新增参数
        ) -> List[dict]:
        results = []
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(input_list)


        # 创建进度条和锁
        progress_bar = tqdm(
            total=len(input_list), 
            desc="Evaling...",
            colour="cyan",
            dynamic_ncols=True,  # 允许动态调整宽度
            unit_scale=False
        )
        progress_lock = asyncio.Lock()
        file_write_lock = asyncio.Lock()

        async def worker(idx, input_data):
            async with semaphore:
                result = await self._evaluate_one(input_data)
                results[idx] = result
                if output_path:
                    async with file_write_lock:
                        with open(output_path, "a", encoding="utf-8") as f:
                            try:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            except Exception as e:
                                print(f"❌ 写入结果失败: {e}")
                                print(f"❌ 写入结果: {result}")
                
                # 任务完成时立即更新进度条
                async with progress_lock:
                    progress_bar.update(1)

        # 创建所有任务
        tasks = [worker(idx, input_data) for idx, input_data in enumerate(input_list)]
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        # 关闭进度条
        progress_bar.close()
        
        return results

    def _load_bootcamp_registry(self, bootcamp_registry: str):
        with jsonlines.open(bootcamp_registry) as reader:
            for line in reader:
                data_source = line.get("data_source")
                yaml_tool_path = line.get("yaml_tool_path")
                yaml_interaction_path = line.get("yaml_interaction_path")
                reward_calculator_class_path = line.get("reward_calculator_class")
                try:
                    tool_schemas, tool_instances = self._load_tools_from_yaml(yaml_tool_path)
                    interaction_instance = self._load_interaction_from_yaml(yaml_interaction_path)
                    reward_calculator_class = load_class_from_string(reward_calculator_class_path)
                    self.bootcamp_registry[data_source] = {
                        "tool_schemas": tool_schemas,
                        "tool_instances": tool_instances,
                        "interaction_instance": interaction_instance,
                        "reward_calculator_class": reward_calculator_class
                    }
                except Exception as e:
                    import pprint
                    pprint.pprint(line)
                    print(f"❌ 加载 {data_source} 配置失败: {e}")
                    raise e
            
    async def run_evaluation(
        self,
        dataset: Optional[List[dict]] = None,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        yaml_tool_path: Optional[str] = None,
        yaml_interaction_path: Optional[str] = None,
        max_concurrent: int = 1,
        bootcamp_registry: Optional[str] = None,
        resume_from_result_path: Optional[str] = None
        ) -> List[dict]:
        """
        启动完整评测流程

        参数:
        - dataset: 数据列表
        - dataset_path: JSON 文件路径（包含测试用例列表）
        - tool_registry: 自定义工具注册表（可选）
        - output_dir: 结果保存路径（JSONL）
        - yaml_tool_path: 工具 YAML 配置路径（如果传入，会覆盖当前 tools）
        """
        # 加载工具配置（可选）
        if yaml_tool_path:
            self.tool_schemas, self.tool_instances = self._load_tools_from_yaml(yaml_tool_path)
        else:
            self.tool_schemas, self.tool_instances = None, None
        if yaml_interaction_path:
            self.interaction = self._load_interaction_from_yaml(yaml_interaction_path)
        else:
            self.interaction = None
        if bootcamp_registry:
            self._load_bootcamp_registry(bootcamp_registry)
        # 加载数据集
        if dataset_path and not dataset:
            dataset = load_dataset(dataset_path)

        if not dataset:
            raise ValueError("必须提供 dataset 或 dataset_path")

        # 断点重试逻辑
        completed_inputs = set()
        original_dataset_size = len(dataset)
        
        if resume_from_result_path and os.path.exists(resume_from_result_path):
            print(f"🔄 检测到断点重试模式，正在从 {resume_from_result_path} 加载已完成的结果...")
            try:
                with open(resume_from_result_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line.strip())
                            if result.get("input"):
                                # 将input转换为字符串作为唯一标识
                                input_key = json.dumps(result["input"], sort_keys=True, ensure_ascii=False)
                                completed_inputs.add(input_key)
                print(f"📊 已完成 {len(completed_inputs)} 个样本，剩余 {original_dataset_size - len(completed_inputs)} 个样本需要评测")
                # 过滤已完成的样本
                filtered_dataset = []
                for item in dataset:
                    item_key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if item_key not in completed_inputs:
                        filtered_dataset.append(item)
                dataset = filtered_dataset
                # 使用现有文件路径作为输出路径
                output_path = resume_from_result_path
            except Exception as e:
                print(f"⚠️ 读取已完成结果时发生错误: {e}，将重新开始评测")
                completed_inputs = set()
                output_path = os.path.join(output_dir, f"{self.api_model.replace('/', '-').strip('-')}/eval_results_{format_time_now()}.jsonl")
        else:
            # 正常模式，生成新的输出文件
            output_path = os.path.join(output_dir, f"{self.api_model.replace('/', '-').strip('-')}/eval_results_{format_time_now()}.jsonl")
        
        print(f"🚀 Starting evaluation with {len(dataset)} samples...")
        
        # 清空或创建输出文件
        if not resume_from_result_path or not os.path.exists(output_path):
            if output_path and os.path.exists(output_path):
                open(output_path, "w", encoding="utf-8").close()
        
        # Create result file
        if output_path and not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"💾 Evaluation results will be saved to: {output_path}")
        
        if len(dataset) == 0:
            print("✅ 所有样本已完成评测!")
            # 加载完整结果用于报告生成
            results = []
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line.strip()))
        else:
            results = await self._evaluate_batch(dataset, max_concurrent=max_concurrent, output_path=output_path)
        summary_path = output_path.replace(".jsonl", ".csv")
        
        # 如果是断点重试模式，确保加载所有结果用于统计
        if resume_from_result_path and len(completed_inputs) > 0:
            # 重新加载完整结果
            all_results = []
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_results.append(json.loads(line.strip()))
            results = all_results
        
        # Save evaluation report, record accuracy, evaluation set, evaluation parameters, etc.
        # Calculate accuracy
        total = len(results)
        score_sum = sum(
            r.get("score", 0)
            for r in results
            if r.get("success") and isinstance(r.get("score"), (int, float))
        )
        avg_score = score_sum / total if total > 0 else 0

        # Generate detailed evaluation report
        report_data = self._generate_evaluation_report(
            results, dataset_path, yaml_tool_path, output_path, avg_score, total
        )
        
        # Save CSV report
        self._save_csv_report(summary_path, report_data)
        
        # Print console report
        self._print_console_report(report_data)

        return results

    def _generate_evaluation_report(
        self, 
        results: List[dict], 
        dataset_path: Optional[str], 
        yaml_tool_path: Optional[str], 
        output_path: str, 
        avg_score: float, 
        total: int
    ) -> dict:
        """
        Generate detailed evaluation report data
        
        Returns:
            dict: Dictionary containing all report data
        """
        # Basic statistics
        success_count = sum(1 for r in results if r.get("success"))
        error_count = total - success_count
        
        # Group statistics by data_source (one-to-many relationship: one data_source corresponds to multiple generators)
        data_source_stats = {}
        
        # Detailed failure analysis
        error_analysis = {"errors": [], "error_types": {}}
        
        for r in results:
            # Get data_source and generator_name
            data_source = r.get("input", {}).get("data_source", "Unknown")
            generator_name = r.get("input", {}).get("extra_info", {}).get("generator_name", "")
            
            # Initialize data_source statistics
            if data_source not in data_source_stats:
                data_source_stats[data_source] = {
                    "total_count": 0,
                    "success_count": 0, 
                    "error_count": 0,
                    "total_score": 0,
                    "avg_score": 0,
                    "max_score": float('-inf'),  # 最高分数
                    "min_score": float('inf'),   # 最低分数
                    "total_assistant_turns": 0,  # 总assistant轮数
                    "avg_assistant_turns": 0,    # 平均assistant轮数
                    "total_tool_calls": 0,       # 总工具调用次数
                    "avg_tool_calls": 0,         # 平均工具调用次数
                    "total_interaction_turns": 0,  # 总interaction轮数
                    "avg_interaction_turns": 0,    # 平均interaction轮数
                    "total_initial_prompt_tokens": 0,           # 初始 prompt tokens
                    "avg_initial_prompt_tokens": 0,      # 平均初始 prompt tokens
                    "total_global_seq_tokens": 0,  # 总 global sequence tokens
                    "avg_global_seq_tokens": 0,    # 平均 global sequence tokens
                    "total_cumulative_prompt_tokens": 0,    # 总 combined prompt tokens
                    "total_completion_tokens": 0,  # 总 completion tokens
                    "total_tokens": 0,           # 总 tokens
                    "avg_cumulative_prompt_tokens": 0,      # 平均 prompt tokens
                    "avg_completion_tokens": 0,  # 平均 completion tokens
                    "avg_tokens": 0,             # 平均 tokens
                    "generators": {}  # All generators under this data source
                }
            
            # Initialize generator statistics
            if generator_name and generator_name not in data_source_stats[data_source]["generators"]:
                data_source_stats[data_source]["generators"][generator_name] = {
                    "total_count": 0,
                    "success_count": 0, 
                    "error_count": 0,
                    "total_score": 0,
                    "avg_score": 0,
                    "max_score": float('-inf'),  # 最高分数
                    "min_score": float('inf'),   # 最低分数
                    "total_assistant_turns": 0,  # 总assistant轮数
                    "avg_assistant_turns": 0,    # 平均assistant轮数
                    "total_tool_calls": 0,       # 总工具调用次数
                    "avg_tool_calls": 0,         # 平均工具调用次数
                    "total_interaction_turns": 0,  # 总interaction轮数
                    "avg_interaction_turns": 0,    # 平均interaction轮数
                    "total_initial_prompt_tokens": 0,           # 初始 prompt tokens
                    "avg_initial_prompt_tokens": 0,      # 平均初始 prompt tokens
                    "total_global_seq_tokens": 0,  # 总 global sequence tokens
                    "avg_global_seq_tokens": 0,    # 平均 global sequence tokens
                    "total_cumulative_prompt_tokens": 0,    # 总 combined prompt tokens
                    "total_completion_tokens": 0,  # 总 completion tokens
                    "total_tokens": 0,           # 总 tokens
                    "avg_cumulative_prompt_tokens": 0,      # 平均 prompt tokens
                    "avg_completion_tokens": 0,  # 平均 completion tokens
                    "avg_tokens": 0,             # 平均 tokens
                }
            
            # Update data_source level statistics
            data_source_stats[data_source]["total_count"] += 1
            
            if r.get("success"):
                data_source_stats[data_source]["success_count"] += 1
                current_score = r.get("score", 0)
                if isinstance(current_score, (int, float)):
                    data_source_stats[data_source]["total_score"] += current_score
                    # 更新最高分和最低分
                    data_source_stats[data_source]["max_score"] = max(
                        data_source_stats[data_source]["max_score"], 
                        current_score
                    )
                    data_source_stats[data_source]["min_score"] = min(
                        data_source_stats[data_source]["min_score"], 
                        current_score
                    )
                
                # 计算工具调用统计数据
                assistant_turns, tool_calls, interaction_turns = self._calculate_tool_statistics(r.get("turn_record", {}))
                data_source_stats[data_source]["total_assistant_turns"] += assistant_turns
                data_source_stats[data_source]["total_tool_calls"] += tool_calls
                data_source_stats[data_source]["total_interaction_turns"] += interaction_turns
                
                # sample token usage statistics
                data_source_stats[data_source]["total_initial_prompt_tokens"] += r.get("prompt_tokens", 0)
                data_source_stats[data_source]["total_global_seq_tokens"] += r.get("global_seq_tokens", 0)

                # 累计 token usage使用统计
                token_usage = r.get("token_usage", {})
                data_source_stats[data_source]["total_global_seq_tokens"] += token_usage.get("global_seq_tokens", 0)
                data_source_stats[data_source]["total_cumulative_prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                data_source_stats[data_source]["total_completion_tokens"] += token_usage.get("completion_tokens", 0)
                data_source_stats[data_source]["total_tokens"] += token_usage.get("total_tokens", 0)
                
                # Update generator level statistics
                if generator_name:
                    data_source_stats[data_source]["generators"][generator_name]["total_count"] += 1
                    data_source_stats[data_source]["generators"][generator_name]["success_count"] += 1
                    if isinstance(current_score, (int, float)):
                        data_source_stats[data_source]["generators"][generator_name]["total_score"] += current_score
                        # 更新生成器的最高分和最低分
                        data_source_stats[data_source]["generators"][generator_name]["max_score"] = max(
                            data_source_stats[data_source]["generators"][generator_name]["max_score"], 
                            current_score
                        )
                        data_source_stats[data_source]["generators"][generator_name]["min_score"] = min(
                            data_source_stats[data_source]["generators"][generator_name]["min_score"], 
                            current_score
                        )
                    
                    # 更新生成器的工具调用统计数据
                    data_source_stats[data_source]["generators"][generator_name]["total_assistant_turns"] += assistant_turns
                    data_source_stats[data_source]["generators"][generator_name]["total_tool_calls"] += tool_calls
                    data_source_stats[data_source]["generators"][generator_name]["total_interaction_turns"] += interaction_turns
                    # 累计生成器的 token 使用统计
                    data_source_stats[data_source]["generators"][generator_name]["total_initial_prompt_tokens"] += r.get("prompt_tokens", 0)
                    data_source_stats[data_source]["generators"][generator_name]["total_global_seq_tokens"] += r.get("global_seq_tokens", 0)
                    data_source_stats[data_source]["generators"][generator_name]["total_cumulative_prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                    data_source_stats[data_source]["generators"][generator_name]["total_completion_tokens"] += token_usage.get("completion_tokens", 0)
                    data_source_stats[data_source]["generators"][generator_name]["total_tokens"] += token_usage.get("total_tokens", 0)
            else:
                data_source_stats[data_source]["error_count"] += 1
                
                # Update generator level statistics
                if generator_name:
                    data_source_stats[data_source]["generators"][generator_name]["total_count"] += 1
                    data_source_stats[data_source]["generators"][generator_name]["error_count"] += 1
                
                # Record error information
                error_info = {
                    "data_source": data_source,
                    "generator_name": generator_name,
                    "error": r.get("error", "Unknown error"),
                    "input_id": r.get("input", {}).get("id", "Unknown")
                }
                error_analysis["errors"].append(error_info)
                
                # 统计错误类型
                error_type = str(r.get("error", "Unknown error"))[:50]
                error_analysis["error_types"][error_type] = error_analysis["error_types"].get(error_type, 0) + 1
        
        # 计算平均分和平均工具调用统计
        for data_source, stats in data_source_stats.items():
            # 计算 data_source 层级平均分
            if stats["success_count"] > 0:
                stats["avg_score"] = stats["total_score"] / stats["success_count"]
                stats["avg_assistant_turns"] = stats["total_assistant_turns"] / stats["success_count"]
                stats["avg_tool_calls"] = stats["total_tool_calls"] / stats["success_count"]
                stats["avg_cumulative_prompt_tokens"] = stats["total_cumulative_prompt_tokens"] / stats["success_count"]
                stats["avg_completion_tokens"] = stats["total_completion_tokens"] / stats["success_count"]
                stats["avg_tokens"] = stats["total_tokens"] / stats["success_count"]
                stats["avg_interaction_turns"] = stats["total_interaction_turns"] / stats["success_count"]
                stats["avg_initial_prompt_tokens"] = stats["total_initial_prompt_tokens"] / stats["success_count"]
                stats["avg_global_seq_tokens"] = stats["total_global_seq_tokens"] / stats["success_count"]
            else:
                # 如果没有成功样本，重置最大最小分数
                stats["max_score"] = 0
                stats["min_score"] = 0
                stats["avg_assistant_turns"] = 0
                stats["avg_tool_calls"] = 0
                stats["avg_interaction_turns"] = 0
                stats["avg_cumulative_prompt_tokens"] = 0
                stats["avg_completion_tokens"] = 0
                stats["avg_tokens"] = 0
                stats["avg_initial_prompt_tokens"] = 0
                stats["avg_global_seq_tokens"] = 0
            
            # 计算各 generator 平均分和平均工具调用统计
            for generator_name, gen_stats in stats["generators"].items():
                if gen_stats["success_count"] > 0:
                    gen_stats["avg_score"] = gen_stats["total_score"] / gen_stats["success_count"]
                    gen_stats["avg_assistant_turns"] = gen_stats["total_assistant_turns"] / gen_stats["success_count"]
                    gen_stats["avg_tool_calls"] = gen_stats["total_tool_calls"] / gen_stats["success_count"]
                    gen_stats["avg_cumulative_prompt_tokens"] = gen_stats["total_cumulative_prompt_tokens"] / gen_stats["success_count"]
                    gen_stats["avg_completion_tokens"] = gen_stats["total_completion_tokens"] / gen_stats["success_count"]
                    gen_stats["avg_tokens"] = gen_stats["total_tokens"] / gen_stats["success_count"]
                    gen_stats["avg_interaction_turns"] = gen_stats["total_interaction_turns"] / gen_stats["success_count"]
                    gen_stats["avg_initial_prompt_tokens"] = gen_stats["total_initial_prompt_tokens"] / gen_stats["success_count"]
                    gen_stats["avg_global_seq_tokens"] = gen_stats["total_global_seq_tokens"] / gen_stats["success_count"]
                else:
                    gen_stats["max_score"] = 0
                    gen_stats["min_score"] = 0
                    gen_stats["avg_assistant_turns"] = 0
                    gen_stats["avg_tool_calls"] = 0
                    gen_stats["avg_interaction_turns"] = 0
                    gen_stats["avg_cumulative_prompt_tokens"] = 0
                    gen_stats["avg_completion_tokens"] = 0
                    gen_stats["avg_tokens"] = 0
                    gen_stats["avg_initial_prompt_tokens"] = 0
                    gen_stats["avg_global_seq_tokens"] = 0
        # 汇总报告数据
        report_data = {
            "basic_info": {
                "model": getattr(self, "api_model", "Unknown"),
                "dataset_path": dataset_path if dataset_path else "Passed-in dataset",
                "tool_config": yaml_tool_path if yaml_tool_path else "Default",
                "output_path": output_path,
                "max_assistant_turns": self.max_assistant_turns,
                "max_user_turns": self.max_user_turns,
                "api_extra_params": getattr(self, "api_extra_params", {}),
                "api_extra_headers": getattr(self, "api_extra_headers", {}),
            },
            "overall_stats": {
                "total_samples": total,
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": success_count / total if total > 0 else 0,
                "overall_avg_score": avg_score,
            },
            "data_source_stats": data_source_stats,
            "error_analysis": error_analysis
        }
        
        return report_data
    
    def _calculate_tool_statistics(self, turn_record: dict) -> Tuple[int, int]:
        """
        计算工具相关统计数据
        
        Args:
            turn_record: 包含每轮交互记录的字典
            
        Returns:
            Tuple[int, int]: (总assistant轮数, 总工具调用次数)
        """
        total_assistant_turns = 0
        total_tool_calls = 0
        total_interaction_turns = 0
        for turn_key, turn_data in turn_record.items():
            if turn_key.startswith("interaction_turn_"):
                total_assistant_turns += turn_data.get("assistant_turns", 0)
                total_tool_calls += turn_data.get("tool_calls_executed", 0)
                total_interaction_turns += 1
        return total_assistant_turns, total_tool_calls, total_interaction_turns

    def _save_csv_report(self, summary_path: str, report_data: dict) -> None:
        """
        保存结构化的 CSV 评测报告
        """
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            
            # 1. Basic Evaluation Information
            writer.writerow(["Basic Evaluation Information"])
            writer.writerow(["Item", "Value"])
            for key, value in report_data["basic_info"].items():
                key_en = {
                    "model": "Evaluation Model",
                    "dataset_path": "Dataset Path", 
                    "tool_config": "Tool Configuration",
                    "output_path": "Output Path",
                    "max_tool_turns": "Max Tool Turns",
                    "max_assistant_turns": "Max Assistant Turns",
                    "max_user_turns": "Max User Turns"
                }.get(key, key)
                writer.writerow([key_en, value])
            
            writer.writerow([])  # 空行分隔
            
            # 2. Overall Statistics
            writer.writerow(["Overall Statistics"])
            writer.writerow(["Metric", "Value"])
            overall = report_data["overall_stats"]
            writer.writerow(["Total Samples", overall["total_samples"]])
            writer.writerow(["Successful Samples", overall["success_count"]])
            writer.writerow(["Failed Samples", overall["error_count"]])
            writer.writerow(["Success Rate", f"{overall['success_rate']:.2%}"])
            writer.writerow(["Overall Average Score", f"{overall['overall_avg_score']:.4f}"])
            
            writer.writerow([])  # 空行分隔
            
            # 3. Data Source Summary Statistics
            if report_data["data_source_stats"]:
                writer.writerow(["Data Source Summary Statistics"])
                writer.writerow(["Data Source", "Total Samples", '',"Success Count", "Failed Count", "Success Rate", 
                            "Average Score", "Max Score", "Min Score", "Avg Assistant Turns", "Avg Tool Calls", "Avg Interaction Turns", "Avg Initial Prompt Tokens", "Avg Completion Tokens", "Avg Global Sequence Tokens",
                            "Avg Cumulative Prompt Tokens", "Avg Total Tokens"])
                
                for data_source, stats in report_data["data_source_stats"].items():
                    success_rate = stats["success_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
                    writer.writerow([
                        data_source,
                        stats["total_count"],
                        '',
                        stats["success_count"], 
                        stats["error_count"],
                        f"{success_rate:.2%}",
                        f"{stats['avg_score']:.4f}",
                        f"{stats['max_score']:.4f}",
                        f"{stats['min_score']:.4f}",
                        f"{stats['avg_assistant_turns']:.2f}",
                        f"{stats['avg_tool_calls']:.2f}",
                        f"{stats['avg_interaction_turns']:.2f}",
                        f"{stats['avg_initial_prompt_tokens']:.2f}",
                        f"{stats['avg_completion_tokens']:.2f}",
                        f"{stats['avg_global_seq_tokens']:.2f}",
                        f"{stats['avg_cumulative_prompt_tokens']:.2f}",
                        f"{stats['avg_tokens']:.2f}",
                    ])
                
                writer.writerow([])  # 空行分隔
                
                # 4. Generator Detailed Statistics (Flattened Table)
                writer.writerow(["Generator Detailed Statistics"])
                writer.writerow(["Data Source", "Generator Name", "Sample Count", "Success Count", "Failed Count", 
                            "Success Rate", "Average Score", "Max Score", "Min Score", "Avg Assistant Turns", "Avg Tool Calls", "Avg Interaction Turns",
                            "Avg Initial Prompt Tokens", "Avg Completion Tokens", "Avg Cumulative Prompt Tokens","Avg Global Sequence Tokens",
                            "Avg Total Tokens"])
                
                for data_source, stats in report_data["data_source_stats"].items():
                    if stats["generators"]:
                        # 按generator_name字典序排序Generator
                        sorted_generators = sorted(stats["generators"].items(), key=lambda x: x[0])
                        for generator_name, gen_stats in sorted_generators:
                            gen_success_rate = gen_stats["success_count"] / gen_stats["total_count"] if gen_stats["total_count"] > 0 else 0
                            writer.writerow([
                                data_source,
                                generator_name,
                                gen_stats["total_count"],
                                gen_stats["success_count"],
                                gen_stats["error_count"],
                                f"{gen_success_rate:.2%}",
                                f"{gen_stats['avg_score']:.4f}",
                                f"{gen_stats['max_score']:.4f}",
                                f"{gen_stats['min_score']:.4f}",
                                f"{gen_stats['avg_assistant_turns']:.2f}",
                                f"{gen_stats['avg_tool_calls']:.2f}",
                                f"{gen_stats['avg_interaction_turns']:.2f}",
                                f"{gen_stats['avg_initial_prompt_tokens']:.2f}",
                                f"{gen_stats['avg_completion_tokens']:.2f}",
                                f"{gen_stats['avg_global_seq_tokens']:.2f}",
                                f"{gen_stats['avg_cumulative_prompt_tokens']:.2f}",
                                f"{gen_stats['avg_tokens']:.2f}",
                            ])
                    else:
                        # 如果没有生成器信息，显示数据源本身
                        ds_success_rate = stats["success_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
                        writer.writerow([
                            data_source,
                            "N/A",
                            stats["total_count"],
                            stats["success_count"],
                            stats["error_count"],
                            f"{ds_success_rate:.2%}",
                            f"{stats['avg_score']:.4f}",
                            f"{stats['max_score']:.4f}",
                            f"{stats['min_score']:.4f}",
                            f"{stats['avg_assistant_turns']:.2f}",
                            f"{stats['avg_tool_calls']:.2f}",
                            f"{stats['avg_interaction_turns']:.2f}",
                            f"{stats['avg_initial_prompt_tokens']:.2f}",
                            f"{stats['avg_completion_tokens']:.2f}",
                            f"{stats['avg_global_seq_tokens']:.2f}",
                            f"{stats['avg_cumulative_prompt_tokens']:.2f}",
                            f"{stats['avg_completion_tokens']:.2f}",
                            f"{stats['avg_tokens']:.2f}",
                            f"{stats['avg_interaction_turns']:.2f}"
                        ])
                
                writer.writerow([])  # 空行分隔
            
            # 5. Error Analysis
            if report_data["error_analysis"]["errors"]:
                writer.writerow(["Error Type Statistics"])
                writer.writerow(["Error Type", "Occurrence Count"])
                for error_type, count in report_data["error_analysis"]["error_types"].items():
                    writer.writerow([error_type, count])
                
                writer.writerow([])  # 空行分隔
                
                writer.writerow(["Detailed Error Information"])
                writer.writerow(["Generator Name", "Data Source", "Sample ID", "Error Message"])
                for error in report_data["error_analysis"]["errors"]:
                    writer.writerow([
                        error["generator_name"] if error["generator_name"] else "N/A", 
                        error["data_source"], 
                        error["input_id"], 
                        error["error"]
                    ])

    def _print_console_report(self, report_data: dict) -> None:
        """
        Print formatted console report
        """
        # Output file path
        print(f"\n💾 Evaluation report saved to: {report_data['basic_info']['output_path'].replace('.jsonl', '.csv')}")
        
        # Overall Summary Section
        overall = report_data["overall_stats"]
        print(f"\n{'='*100}")
        print(f"{'📊 EVALUATION SUMMARY':^100}")
        print(f"{'='*100}")
        print(f"  ✅ Overall Status     : {overall['success_count']}/{overall['total_samples']} successful (Success Rate: {overall['success_rate']:.1%})")
        print(f"  📈 Average Score      : {overall['overall_avg_score']:.4f}")
        print(f"{'='*100}")
        
        # Statistics grouped by data source (hierarchical structure)
        if report_data["data_source_stats"]:
            print(f"\n{'='*159}")
            print(f"{'📋 STATISTICS BY DATA SOURCE':^159}")
            print(f"{'='*159}")
            
            # Define column widths for consistency
            col_widths = {
                'source': 20,
                'samples': 10,
                'success': 10,
                'avg_score': 12,
                'max_score': 12,
                'min_score': 12,
                'avg_assistant_turns': 20,
                'avg_tool_calls': 15,
                'avg_completion_tokens': 20,
                'avg_interaction_turns': 20,
            }
            
            # Table header
            header = (f"{'Data Source':<{col_widths['source']}} "
                     f"{'Samples':>{col_widths['samples']}} "
                     f"{'Success':>{col_widths['success']}} "
                     f"{'Avg-Score':>{col_widths['avg_score']}} "
                     f"{'Max-Score':>{col_widths['max_score']}} "
                     f"{'Min-Score':>{col_widths['min_score']}} "
                     f"{'Avg-Assistant-Turns':>{col_widths['avg_assistant_turns']}} "
                     f"{'Avg-Tool-Calls':>{col_widths['avg_tool_calls']}} "
                     f"{'Avg-Interaction-Turns':>{col_widths['avg_interaction_turns']}} "
                     f"{'Avg-Completion-Tokens':>{col_widths['avg_completion_tokens']}}")
            print(header)
            print(f"{'-'*159}")
            
            # 为数字列居中对齐，通过在列宽基础上向右偏移列宽/2
            def center_value(val_str, width):
                # 居中对齐字符串
                return f"{val_str:^{width}}"
            
            for data_source, stats in report_data["data_source_stats"].items():
                success_rate = stats["success_count"] / stats["total_count"] if stats["total_count"] > 0 else 0

                row = (
                    f"{data_source:<{col_widths['source']}} "
                    f"{center_value(str(stats['total_count']), col_widths['samples'])} "
                    f"{center_value(f'{success_rate:.1%}', col_widths['success'])} "
                    f"{center_value(f'{stats['avg_score']:.4f}', col_widths['avg_score'])} "
                    f"{center_value(f'{stats['max_score']:.4f}', col_widths['max_score'])} "
                    f"{center_value(f'{stats['min_score']:.4f}', col_widths['min_score'])} "
                    f"{center_value(f'{stats['avg_assistant_turns']:.2f}', col_widths['avg_assistant_turns'])} "
                    f"{center_value(f'{stats['avg_tool_calls']:.2f}', col_widths['avg_tool_calls'])} "
                    f"{center_value(f'{stats['avg_interaction_turns']:.2f}', col_widths['avg_interaction_turns'])} "
                    f"{center_value(f'{stats['avg_completion_tokens']:.2f}', col_widths['avg_completion_tokens'])}"
                )
                print(row)
                
                # If there are generator subdivisions, show generator statistics
                if stats["generators"]:
                    # Generator breakdown separator
                    separator_line = f"  {'└─ Generators:':<{col_widths['source']-2}}"
                    print(separator_line)
                    
                    # 按generator_name字典序排序Generator
                    sorted_generators = sorted(stats["generators"].items(), key=lambda x: x[0])
                    for idx, (generator_name, gen_stats) in enumerate(sorted_generators):
                        gen_success_rate = gen_stats["success_count"] / gen_stats["total_count"] if gen_stats["total_count"] > 0 else 0
                        is_last = (idx == len(stats["generators"]) - 1)
                        prefix = "  ● "
                        
                        # Adjust generator name width to account for prefix
                        gen_name_width = col_widths['source'] - len(prefix)

                        gen_row = (
                            f"{prefix}{generator_name:<{gen_name_width}} "
                            f"{center_value(str(gen_stats['total_count']), col_widths['samples'])} "
                            f"{center_value(f'{gen_success_rate:.1%}', col_widths['success'])} "
                            f"{center_value(f'{gen_stats['avg_score']:.4f}', col_widths['avg_score'])} "
                            f"{center_value(f'{gen_stats['max_score']:.4f}', col_widths['max_score'])} "
                            f"{center_value(f'{gen_stats['min_score']:.4f}', col_widths['min_score'])} "
                            f"{center_value(f'{gen_stats['avg_assistant_turns']:.2f}', col_widths['avg_assistant_turns'])} "
                            f"{center_value(f'{gen_stats['avg_tool_calls']:.2f}', col_widths['avg_tool_calls'])} "
                            f"{center_value(f'{gen_stats['avg_interaction_turns']:.2f}', col_widths['avg_interaction_turns'])} "
                            f"{center_value(f'{gen_stats['avg_completion_tokens']:.2f}', col_widths['avg_completion_tokens'])}"
                        )
                        print(gen_row)
                    print(f"{'-'*159}")
            
            print(f"{'='*159}")
        
        # Error Summary Section
        if report_data["error_analysis"]["errors"]:
            print(f"\n{'='*100}")
            print(f"{'⚠️  ERROR SUMMARY':^100}")
            print(f"{'='*100}")
            print(f"  Total Errors: {len(report_data['error_analysis']['errors'])}")
            print(f"\n  Main Error Types:")
            for error_type, count in list(report_data["error_analysis"]["error_types"].items())[:3]:
                print(f"    • {error_type:<80} ({count} occurrence{'s' if count > 1 else ''})")
            print(f"{'='*100}")
