#!/usr/bin/env python3
"""
通用命令行评测脚本

修改说明：
已增强以支持自定义 OpenAI 格式的 API 调用（如 Qwen/InternLM 等）。
支持自动读取 INF_API_KEY 环境变量。
"""

import argparse
import asyncio
import importlib
import os
import ast
import sys
import json
import traceback
from typing import Dict, Any, Optional, List

# 保持原有引用路径不变
from internbootcamp.src.base_evaluator import BaseEvaluator
from internbootcamp.utils.load_class_from_str import load_class_from_string

def create_evaluator(
    evaluator_class: str = None,
    api_url: str = None,
    api_key: str = None,
    api_model: str = "gpt-3.5-turbo",
    reward_calculator = None,
    max_assistant_turns: int = 10,
    max_user_turns: int = 5,
    api_extra_headers: Optional[Dict] = None,
    api_extra_params: Optional[Dict] = None,
    verify_correction_kwargs: Optional[Dict] = None,
    **kwargs
):
    """
    创建评测器实例
    """
    if evaluator_class:
        evaluator_cls = load_class_from_string(evaluator_class)
    else:
        evaluator_cls = BaseEvaluator
    
    return evaluator_cls(
        api_url=api_url,
        api_key=api_key,
        api_model=api_model,
        reward_calculator=reward_calculator,
        max_assistant_turns=max_assistant_turns,
        max_user_turns=max_user_turns,
        api_extra_headers=api_extra_headers,
        api_extra_params=api_extra_params,
        verify_correction_kwargs=verify_correction_kwargs,
        **kwargs
    )


def parse_extra_headers(headers_str: str) -> Dict[str, str]:
    """
    解析额外的HTTP头部参数
    """
    headers = {}
    if headers_str:
        for header in headers_str.split(','):
            if ':' in header:
                key, value = header.split(':', 1)
                headers[key.strip()] = value.strip()
    return headers


def parse_extra_params(params_str: str) -> Dict[str, any]:
    """
    解析额外的模型参数
    支持 @filename, JSON 字符串, 或 k:v 格式
    """
    if not params_str:
        return {}

    text = params_str.strip()

    # 情况 1：@文件 路径
    if text.startswith('@'):
        file_path = text[1:]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"参数文件不存在: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            loaded = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"参数文件 JSON 解析失败: {e}")
        if not isinstance(loaded, dict):
            raise ValueError("参数文件的 JSON 根类型必须为对象(dict)")
        return loaded

    # 情况 2：直接作为 JSON 字符串
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
        else:
            raise ValueError("JSON 根类型必须为对象(dict)")
    except json.JSONDecodeError:
        # 继续尝试回退到旧格式解析
        try:
            loaded = ast.literal_eval(text)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            print(f"Depreciated Warning: {text} 额外参数未能解析为 JSON 格式，将使用旧格式解析。")

    # 回退：旧格式 "k:v,k2:v2"
    params: Dict[str, Any] = {}
    for param in text.split(','):
        if ':' not in param:
            continue
        key, value = param.split(':', 1)
        key = key.strip()
        value = value.strip()

        # 自动转换类型
        try:
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                params[key] = int(value)
            elif '.' in value:
                params[key] = float(value)
            elif value.lower() in ('true', 'false'):
                params[key] = value.lower() == 'true'
            else:
                params[key] = value
        except ValueError:
            params[key] = value

    return params


def main():
    parser = argparse.ArgumentParser(
        description="通用命令行评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例: 
1. OpenAI:
   export OPENAI_API_KEY=sk-xxx
   python run_evaluation.py ...

2. 自定义 API (如 Qwen):
   export INF_API_KEY=xxx
   python run_evaluation.py --api-url https://.../v1 --api-model qwen3-next ...
"""
    )
    
    parser.add_argument('--dataset-path', type=str, required=True, help='数据集文件路径 (.json, .jsonl, .parquet)')
    parser.add_argument('--output-dir', type=str, required=True, help='评测结果输出目录')
    
    # API 相关配置
    parser.add_argument('--api-key', type=str, default=None, help='API 密钥 (如果不指定，将按顺序查找环境变量 INF_API_KEY, OPENAI_API_KEY)')
    parser.add_argument('--api-url', type=str, default=None, help='API Base URL (例如: https://.../v1)')
    parser.add_argument('--api-model', type=str, default='gpt-3.5-turbo', help='模型名称')
    
    # 额外参数
    parser.add_argument('--api-extra-headers', type=str, default=None, help='额外的API头部')
    parser.add_argument('--api-extra-params', type=str, default=None, help='额外的模型参数 (JSON或文件)')
    parser.add_argument('--verify-correction-kwargs', type=str, default=None, help='传递给奖励计算器的额外参数')
    
    # 类配置
    parser.add_argument('--evaluator-class', type=str, default=None, help='评测器类路径')
    parser.add_argument('--reward-calculator-class', type=str, default=None, help='奖励计算器类路径')
    
    # 工具与交互配置
    parser.add_argument('--tool-config', type=str, default=None, help='工具配置YAML文件路径')
    parser.add_argument('--interaction-config', type=str, default=None, help='交互配置YAML文件路径')
    
    # 轮次控制
    parser.add_argument('--max-assistant-turns', type=int, default=None, help='assistant响应的最大轮次')
    parser.add_argument('--max-user-turns', type=int, default=None, help='user输入的最大轮次')
    
    # 旧参数兼容
    parser.add_argument('--max-tool-turns-per-interaction', type=int, default=None, help='(已弃用) 请使用 max-assistant-turns')
    parser.add_argument('--max-interaction-turns', type=int, default=None, help='(已弃用) 请使用 max-assistant-turns')
    
    # 运行控制
    parser.add_argument('--max-concurrent', type=int, default=1, help='最大并发数')
    parser.add_argument('--verbose', action='store_true', help='输出详细信息')
    parser.add_argument('--dry-run', action='store_true', help='只验证配置')
    parser.add_argument('--tokenizer-path', type=str, default=None, help='tokenizer路径')
    parser.add_argument('--bootcamp-registry', type=str, default=None, help='bootcamp注册表路径')
    parser.add_argument('--resume-from-result-path', type=str, default=None, help='断点重试模式：指定结果文件路径')
    parser.add_argument('--max-iterations', type=int, default=None, help='单轮数据最大迭代次数')
    
    args = parser.parse_args()
    
    # -----------------------------------------------------------
    # 核心修改：API Key 智能获取逻辑
    # 优先级：命令行参数 > INF_API_KEY (你的自定义环境) > OPENAI_API_KEY
    # -----------------------------------------------------------
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("INF_API_KEY")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ 错误: 未找到 API Key。请通过 --api-key 参数，或设置环境变量 INF_API_KEY / OPENAI_API_KEY。")
        sys.exit(1)

    # 验证输入文件
    if not os.path.exists(args.dataset_path):
        print(f"❌ 错误: 数据集文件不存在: {args.dataset_path}")
        sys.exit(1)
    
    # 验证断点重试文件
    if args.resume_from_result_path and not os.path.exists(args.resume_from_result_path):
        print(f"❌ 错误: 断点重试文件不存在: {args.resume_from_result_path}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理参数兼容性
    max_assistant_turns = args.max_assistant_turns
    if args.max_tool_turns_per_interaction:
        print("⚠️  警告: --max-tool-turns-per-interaction已弃用，将覆盖max-assistant-turns")
        max_assistant_turns = args.max_tool_turns_per_interaction
    if args.max_interaction_turns and not max_assistant_turns:
        print("⚠️  警告: --max-interaction-turns已弃用，将作为fallback")
        max_assistant_turns = args.max_interaction_turns
        
    # 如果都没设置，给个默认值防止报错
    if max_assistant_turns is None:
        max_assistant_turns = 10

    if args.verbose:
        print("🔧 配置信息:")
        print(f"  数据集路径: {args.dataset_path}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  API URL: {args.api_url if args.api_url else '默认 (OpenAI Official)'}")
        print(f"  API 模型: {args.api_model}")
        print(f"  API Key: {api_key[:6]}...{api_key[-4:]} (已加载)")
        print(f"  工具配置: {args.tool_config}")
        print(f"  最大Assistant轮次: {max_assistant_turns}")
        print(f"  最大并发: {args.max_concurrent}")

    try:
        # 解析额外头部和参数
        extra_headers = parse_extra_headers(args.api_extra_headers) if args.api_extra_headers else None
        extra_params = parse_extra_params(args.api_extra_params) if args.api_extra_params else None
        verify_correction_kwargs = parse_extra_params(args.verify_correction_kwargs) if args.verify_correction_kwargs else None
        
        # 创建奖励管理器
        if args.verbose:
            print("📊 正在创建奖励计算器...")
        if args.reward_calculator_class:
            reward_calculator = load_class_from_string(args.reward_calculator_class)
        else:
            reward_calculator = None
        
        # 创建评测器
        if args.verbose:
            print("🤖 正在创建评测器...")
            
        evaluator = create_evaluator(
            evaluator_class=args.evaluator_class,
            api_url=args.api_url,     # 传入自定义 URL
            api_key=api_key,          # 传入解析后的 Key
            api_model=args.api_model,
            reward_calculator=reward_calculator,
            max_assistant_turns=max_assistant_turns,
            max_user_turns=args.max_user_turns,
            api_extra_headers=extra_headers,
            api_extra_params=extra_params,
            verify_correction_kwargs=verify_correction_kwargs,
            tokenizer_path=args.tokenizer_path,
            max_iterations=args.max_iterations
        )
        
        if args.dry_run:
            print("✅ 配置验证通过！(干运行模式)")
            return
        
        # 运行评测
        asyncio.run(evaluator.run_evaluation(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            yaml_tool_path=args.tool_config,
            yaml_interaction_path=args.interaction_config,
            max_concurrent=args.max_concurrent,
            bootcamp_registry=args.bootcamp_registry,
            resume_from_result_path=args.resume_from_result_path
        ))
        
    except Exception as e:
        print(f"❌ 评测过程中发生错误: {str(e)}")
        if args.verbose:
            print("详细错误信息:")
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()