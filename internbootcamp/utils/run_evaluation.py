#!/usr/bin/env python3
"""
通用命令行评测脚本 (已修改以兼容自定义 OpenAI 格式 API)

修改说明:
1. 增加了对环境变量 INF_API_KEY 的自动读取支持。
2. 增加了 API Key 的校验逻辑。
3. 优化了 api_url 的传入逻辑。
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

from internbootcamp.src.base_evaluator import BaseEvaluator
from internbootcamp.utils.load_class_from_str import load_class_from_string

# ... [create_evaluator, parse_extra_headers, parse_extra_params 函数保持不变] ...
# 为了节省篇幅，这里省略这三个辅助函数的代码，它们不需要修改
# 请在实际文件中保留原有的 create_evaluator, parse_extra_headers, parse_extra_params

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
    """(保持原有代码不变)"""
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
    """(保持原有代码不变)"""
    headers = {}
    if headers_str:
        for header in headers_str.split(','):
            if ':' in header:
                key, value = header.split(':', 1)
                headers[key.strip()] = value.strip()
    return headers


def parse_extra_params(params_str: str) -> Dict[str, any]:
    """(保持原有代码不变)"""
    if not params_str:
        return {}
    text = params_str.strip()
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
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
        else:
            raise ValueError("JSON 根类型必须为对象(dict)")
    except json.JSONDecodeError:
        try:
            loaded = ast.literal_eval(text)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            print(f"Depreciated Warning: {text} 额外参数未能解析为 JSON 格式，将使用旧格式解析。")

    params: Dict[str, Any] = {}
    for param in text.split(','):
        if ':' not in param:
            continue
        key, value = param.split(':', 1)
        key = key.strip()
        value = value.strip()
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
1. 基础用法:
   python eval.py --dataset-path test.json --api-key sk-xxxx

2. 使用自定义API (如您的示例):
   export INF_API_KEY="stpmj/..."
   python eval.py --dataset-path test.json \
                  --api-url "https://heke889hhd88c5bcm8p5bo8d5g8k98kb.openapi-sj.sii.edu.cn/v1" \
                  --api-model "qwen3-next"
"""
    )
    
    parser.add_argument('--dataset-path', type=str, required=True, help='数据集文件路径 (.json, .jsonl, .parquet)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='评测结果输出目录')
    
    # 修改：help信息中说明支持环境变量
    parser.add_argument('--api-key', type=str, default=None, help='API 密钥 (优先使用命令行参数，缺省时读取环境变量 INF_API_KEY 或 OPENAI_API_KEY)')
    parser.add_argument('--api-url', type=str, default=None, help='API Base URL (例如: https://.../v1)')
    
    parser.add_argument('--api-model', type=str, default='gpt-3.5-turbo', help='模型名称 (默认: gpt-3.5-turbo)')
    parser.add_argument('--api-extra-headers', type=str, default=None, help='额外的API头部，格式: "key1:value1"')
    parser.add_argument('--api-extra-params', type=str, default=None, help='额外的模型参数')
    parser.add_argument('--verify-correction-kwargs', type=str, default=None, help='额外的奖励计算函数参数')
    parser.add_argument('--evaluator-class', type=str, default=None, help='评测器类路径')
    parser.add_argument('--reward-calculator-class', type=str, default=None, help='奖励计算器类路径')
    parser.add_argument('--tool-config', type=str, default=None, help='工具配置YAML文件路径')
    parser.add_argument('--interaction-config', type=str, default=None, help='交互配置YAML文件路径')
    parser.add_argument('--max-assistant-turns', type=int, default=10, help='assistant响应的最大轮次')
    parser.add_argument('--max-user-turns', type=int, default=5, help='user输入的最大轮次')
    parser.add_argument('--max-concurrent', type=int, default=1, help='最大并发数')
    parser.add_argument('--verbose', action='store_true', help='输出详细信息')
    parser.add_argument('--dry-run', action='store_true', help='只验证配置')
    parser.add_argument('--tokenizer-path', type=str, default=None, help='tokenizer路径')
    parser.add_argument('--bootcamp-registry', type=str, default=None, help='bootcamp注册表路径')
    parser.add_argument('--resume-from-result-path', type=str, default=None, help='断点重试模式文件路径')
    parser.add_argument('--max-iterations', type=int, default=None, help='单轮数据最大迭代次数')
    
    # 兼容旧参数（不显示在help中或作为可选）
    parser.add_argument('--max-tool-turns-per-interaction', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--max-interaction-turns', type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    
    # --------------------------------------------------------------------------
    # [修改点 1]: 智能获取 API Key
    # 逻辑：命令行参数 > 环境变量 INF_API_KEY (你的示例) > 环境变量 OPENAI_API_KEY
    # --------------------------------------------------------------------------
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("INF_API_KEY")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ 错误: 未找到 API Key。请通过 --api-key 参数或环境变量 INF_API_KEY / OPENAI_API_KEY 提供。")
        sys.exit(1)

    # 验证输入文件
    if not os.path.exists(args.dataset_path):
        print(f"❌ 错误: 数据集文件不存在: {args.dataset_path}")
        sys.exit(1)
    
    if args.resume_from_result_path and not os.path.exists(args.resume_from_result_path):
        print(f"❌ 错误: 断点重试文件不存在: {args.resume_from_result_path}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理参数兼容性
    max_assistant_turns = args.max_assistant_turns
    if args.max_tool_turns_per_interaction:
        print("⚠️  警告: --max-tool-turns-per-interaction已弃用，将覆盖max-assistant-turns")
        max_assistant_turns = args.max_tool_turns_per_interaction

    if args.verbose:
        print("🔧 配置信息:")
        print(f"  数据集路径: {args.dataset_path}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  API URL: {args.api_url if args.api_url else 'Default (OpenAI)'}")
        print(f"  API 模型: {args.api_model}")
        print(f"  API Key: {api_key[:6]}******{api_key[-4:]} (已掩码)")
        print(f"  最大assistant轮次: {max_assistant_turns}")
    
    try:
        extra_headers = parse_extra_headers(args.api_extra_headers) if args.api_extra_headers else None
        extra_params = parse_extra_params(args.api_extra_params) if args.api_extra_params else None
        verify_correction_kwargs = parse_extra_params(args.verify_correction_kwargs) if args.verify_correction_kwargs else None
        
        # 创建奖励管理器
        if args.reward_calculator_class:
            reward_calculator = load_class_from_string(args.reward_calculator_class)
        else:
            reward_calculator = None
        
        # 创建评测器
        # --------------------------------------------------------------------------
        # [修改点 2]: 使用处理后的 api_key 变量，而不是 args.api_key
        # --------------------------------------------------------------------------
        evaluator = create_evaluator(
            evaluator_class=args.evaluator_class,
            api_url=args.api_url,
            api_key=api_key,  # 使用自动获取的 Key
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
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
