# InternBootcampv2 指南

## 目录

- [1. 架构概述](#1-架构概述)
  - [1.1 InternBootcampv2核心改进](#11-internbootcampv2核心改进)
  - [1.2 Multi-round Toolcall的实现原理](#12-multi-round-toolcall的实现原理)
    - [1.2.1 与Bootcampv1的差异](#121-与bootcampv1的差异)
    - [1.2.2 复杂Bootcamp的代码逻辑](#122-复杂bootcamp的代码逻辑)
- [2. 环境准备](#2-环境准备)
  - [2.1 安装依赖](#21-安装依赖)
  - [2.2 系统架构设计](#22-系统架构设计)
    - [2.2.1 核心组件](#221-核心组件)
    - [2.2.2 设计原则](#222-设计原则)
- [3. 核心组件开发](#3-核心组件开发)
  - [3.1 指令生成器开发](#31-指令生成器开发)
    - [3.1.1 功能职责](#311-功能职责)
    - [3.1.2 开发指南](#312-开发指南)
    - [3.1.3 配置文件管理](#313-配置文件管理)
    - [3.1.4 数据生成操作](#314-数据生成操作)
      - [3.1.4.1 单个配置生成](#3141-单个配置生成)
      - [3.1.4.2 生成数据格式](#3142-生成数据格式)
      - [3.1.4.3 批量数据生成](#3144-批量数据生成)
  - [3.2 工具类开发](#32-工具类开发)
    - [3.2.1 功能职责](#321-功能职责)
    - [3.2.2 开发指南](#322-开发指南)
    - [3.2.3 配置文件定义](#323-配置文件定义)
  - [3.3 交互类开发](#33-交互类开发)
    - [3.3.1 功能职责](#331-功能职责)
    - [3.3.2 开发指南](#332-开发指南)
    - [3.3.3 接口规范](#333-接口规范)
    - [3.3.4 配置文件定义](#334-配置文件定义)
    - [3.3.5 数据格式约定](#335-数据格式约定)
    - [3.3.6 运行时集成](#336-运行时集成)
    - [3.3.7 示例实现](#337-示例实现)
  - [3.4 分布式工具服务器](#34-分布式工具服务器)
    - [3.4.1 架构概述](#341-架构概述)
    - [3.4.2 运行模式](#342-运行模式)
      - [3.4.2.1 Master模式（生产推荐）](#3421-master模式生产推荐)
      - [3.4.2.2 Worker模式（生产推荐）](#3422-worker模式生产推荐)
      - [3.4.2.3 Unified模式（调试推荐）](#3423-unified模式调试推荐)
    - [3.4.3 核心特性](#343-核心特性)
    - [3.4.4 命令行参数](#344-命令行参数)
    - [3.4.5 配置文件转换](#345-配置文件转换)
    - [3.4.6 监控与管理](#346-监控与管理)
  - [3.5 奖励计算器开发](#35-奖励计算器开发)
    - [3.5.1 功能职责](#351-功能职责)
    - [3.5.2 开发指南](#352-开发指南)
- [4. 模型评估](#4-模型评估)
  - [4.1 评估配置](#41-评估配置)
    - [4.1.1 基础评估配置](#411-基础评估配置)
    - [4.1.2 新增核心功能](#412-新增核心功能)
    - [4.1.3 参数列表](#413-参数列表)
  - [4.2 评估输出](#42-评估输出)
  - [4.3 评估数据后处理](#43-评估数据后处理)
    - [4.3.1 功能概述](#431-功能概述)
    - [4.3.2 使用方式](#432-使用方式)
    - [4.3.3 命令行参数](#433-命令行参数)
    - [4.3.4 编程接口](#434-编程接口)
- [5. 总结](#5-总结)


这里是基于InternBootcampv1开发的InternBootcampv2，主要包含了针对具体专业性Bootcamp场景的multi-round toolcall的agentic RL 流程。

## 1. 架构概述

### 1.1 InternBootcampv2核心改进

InternBootcampv2 深入绑定了verl的multi-round toolcall的流程，通过SGLANG-rollout的state control，实现对专业场景的toolcall调用、reward计算、模型训练及推理

### 1.2 Multi-round Toolcall的实现原理

#### 1.2.1 与Bootcampv1的差异

为了支持更复杂的Bootcamp任务，我们需要多轮工具调用来处理。

1. 基于多轮工具调用的复杂推理任务，通常有两种形式：

   （1）LLM-based workflow

   prompt --> LLM rollout(decision) --> tooluse response --> LLM rollout(decision) --> tooluse response ...

   （2）environment-based workflow

   prompt --> LLM rollout --> tooluse --> env decision --> LLM rollout --> tooluse --> env decision ...

   即，一个Bootcamp任务存在两种LLM与环境的交互形式

2. 为什么要multi-round toolcall & multi-round toolcall很复杂：

   （1）single-round toolcall & 多轮对话，本质上都可以refine成single-round的交互

   （2）multi-round toolcall 的调用在RL framework中，需要async的调用，导致rollout逻辑会更加复杂

#### 1.2.2 复杂Bootcamp的代码逻辑

1. 一个需要多轮调用工具的Bootcamp流程：两种情况 toolcall or interaction-call ，以下统称toolcall

   流程包含了：
   
   （1）模型的prompt Design，workflow的整个流程；
   
   prompt Design，在verl里体现为数据构造，对应包含data-source来明确task的类型，
   
   生成这种数据的流程，在Bootcamp中，参考InternBootcampv1的对应思路；

   （2）对应需要调用的tool的toollist tool的描述；
   
   verl中， toollist，在toolconfig里填写，并且会在chat-template中基于对应mcp协议，拼接在chat-template里，
   
   tool的config中，直接绑定了对应的tool的实现；

   （3）调用的tool的实现逻辑，对应的io；
   
   verl中，tool的调用，是通过执行execute，在SGLANG的rollout中对应调用的，
   
   其中，SGLANG的rollout，会根据对应的state，对rollout过程进行控制，这些state包含了toolcall 和terminate等信号，
   
   同时，在这里会计算每一轮的toolcall返回的中间过程reward；
   
   SGLANG的rollout中，需要传入的参数包含了
   
   - 完整的workflow，即prompt、toollist等（这里很重要，因为previous sglang&verl版本中对rollout的传参控制中不包含toollist等内容）；

   （4）基于tool的io的返回，模型的multi-round rollout；
   
   整体rollout结束后，会通过reward score 处计算最终的reward 

   （5）基于toolcall返回 or 基于模型判断 or 基于max-step 等条件，任务workflow的最终终止；
   
   verl中，state中控制了流程的终止

## 2. 环境准备

### 2.1 安装依赖

#### 2.1.1 安装InternBootcamp

```bash
git clone ssh://git@gitlab.pjlab.org.cn:1122/chenyongkang/internbootcamp_v2.git --recurse-submodules
cd internbootcamp_v2
pip install -e ./
```

#### 2.1.2 安装verl

TODO

### 2.2 系统架构设计

#### 2.2.1 核心组件

1. **InstructionGenerator**: 指令生成，负责prompt设计和数据流控制
2. **ToolManager**: 工具管理，处理工具调用和执行
3. **RewardCalculator**: 奖励计算
4. **Evaluator**: 评估器,可以用于评测、rollout、bootcamp开发调试
5. **InteractionManager**: 交互管理，处理用户输入输出

#### 2.2.2 设计原则

- **模块化**: 各组件独立，便于维护和扩展
- **可配置**: 通过配置文件控制行为
- **可扩展**: 支持新工具和场景的快速接入

## 3. 核心组件开发与使用

### 3.1 指令生成器开发

#### 3.1.1 功能职责
- 完成任务生成和数据生成。

#### 3.1.2 开发指南

**继承基类BaseInstructionGenerator**

**示例**: [example_instruction_generator.py](/internbootcamp/bootcamps/example_bootcamp/example_instruction_generator.py)

若需生成数据，须继承`BaseInstructionGenerator`基类，并实现以下两个抽象方法：（若无需生成数据则将数据规范为以下对应格式即可，无需实现InstructionGenerator）

```python
from internbootcamp.src.base_instruction_generator import BaseInstructionGenerator
from typing import Dict, Any, Optional
import random

class CustomInstructionGenerator(BaseInstructionGenerator):
    """自定义指令生成器"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # 配置data_source,基类已如下实现，计算reward时会根据此信息匹配对应的RewardCalculator
        self.data_source = f"bootcamp/{self.__class__.__name__.replace('InstructionGenerator', '').lower()}"
        # 初始化自定义参数
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def case_generator(self) -> Dict[str, Any]:
        """
        生成任务案例，返回包含任务信息的字典
        
        Returns:
            Dict[str, Any]: 任务信息字典，包含ground_truth等关键信息；
            这个任务信息字典将作为对应RewardCalculator的_verify_correction方法和对应Tool的create方法的identity参数传入
        """
        # 实现任务生成逻辑
        # 例如：生成数学题目、电路参数等
        pass
    
    def prompt_func(self, identity: Dict[str, Any]) -> str:
        """
        根据任务信息生成提示语
        
        Args:
            identity (Dict[str, Any]): 任务信息字典
            
        Returns:
            str: 生成的提示词
        """
        # 实现提示语生成逻辑
        # 使用identity中的信息构建输入给语言模型的prompt
        pass
```

**关键要点：**
- `data_source`属性用于标识数据来源，格式为"bootcamp/your_bootcamp_name"
- `case_generator()`方法生成能用来唯一确定任务且验证候选答案的参数集，以字典格式返回
- `prompt_func()`方法根据任务参数生成提示词

**实践建议：**
- Prompt模板：使用多个prompt模板，生成时随机选择，实现多样prompt
- 题目变种：使用参数`data_type`参数控制题目变种类型，实现多样的题型与prompt

#### 3.1.3 配置文件管理

**示例**: [example_instruction_config.yaml](/internbootcamp/bootcamps/example_bootcamp/configs/example_instruction_config.yaml)

使用YAML配置文件来管理数据生成的参数和行为：

```yaml
# custom_instruction_config.yaml
instruction_generators:
  # 基础配置组
  basic_config:
    config:
      min_value: 1
      max_value: 100
      operation_type: "simple"
    generation_ratio: 0.1  # 占总生成数据的40%，所有generation_ratio会自动归一化为比例（如0.4和0.6会分别分配40%和60%的样本数）

  # 高级配置组  
  advanced_config:
    config:
      min_value: 50
      max_value: 500
      operation_type: "complex"
    generation_ratio: 0.7  # 占总生成数据的60%，无需手动保证所有ratio之和为1，系统会自动归一化

  # 带默认tool的配置组
  basic_config_w_tool:
    min_value: 1
      max_value: 100
      operation_type: "simple"
    generation_ratio: 0.1
    yaml_tool_path: "path/to/tool.yaml" # 工具配置文件路径

  # 带默认interaction的配置组
  basic_config_w_tool:
    min_value: 1
      max_value: 100
      operation_type: "simple"
    generation_ratio: 0.1
    yaml_interaction_path: "path/to/interaction.yaml" # 交互配置文件路径

# 全局配置
global_config:
  # 指令生成器类的完整路径
  class_name: "internbootcamp.bootcamps.your_bootcamp.your_instruction_generator.CustomInstructionGenerator"
  
  # 随机种子配置
  enable_random_seed: true
  default_seed: 999
  
  # 数据集划分配置
  default_split_samples:
    train: 10000
    test: 1000
  
  # 是否启用数据打乱
  shuffle: true

  # 是否生成parquet文件
  gen_parquet: false
```

**配置说明：**
- `instruction_generators`: 定义多组配置，每组有不同的参数和生成比例
- `global_config`: 全局设置，包括类路径、随机种子、数据集划分、是否生成parquet格式数据等,也可于此配置tool或interaction config(全局配置会与单条generator配置合并)
- `generation_ratio`: 控制每组配置生成数据的比例，自动归一化

#### 3.1.4 数据生成操作

**示例**: [example_multiturn_w_tool_grpo.sh](/internbootcamp/bootcamps/example_bootcamp/examples/example_multiturn_w_tool_grpo.sh)

##### 3.1.4.1 单个配置生成

例如以下命令
```bash
# data_generate.sh
#!/bin/bash
python -m internbootcamp.utils.data_generation \
    --instruction-config configs/your_instruction_config.yaml \
    --output-dir data/your_bootcamp/ \
    --tool-config configs/your_tool_config.yaml \ #全局配置，与单条配置合并
    --interaction-config configs/your_interaction_config.yaml \ #全局配置，与单条配置合并
    --split-samples train:10000,test:1000 \
    --shuffle \
    --global-config-overrides '{"gen_parquet": true}' \ #全局配置覆盖
    --no-tool \ #开启则不使用任何工具配置
    --no-interaction \ #开启则不使用任何交互配置
```

**参数列表**

| 参数名称 | 类型 | 必需 | 默认值 | 说明 |
|---------|------|------|--------|------|
| `--instruction-config` | str | ✓ | - | 指令管理器配置文件路径，定义数据生成的核心逻辑 |
| `--output-dir` | str | ✓ | - | 输出文件目录，生成的数据集将保存在此目录下 |
| `--tool-config` | str | ✗ | None | 工具配置文件路径，相当于全局配置，生成时与单条配置合并（冲突时优先使用单条配置） |
| `--interaction-config` | str | ✗ | None | 交互配置文件路径，相当于全局配置，生成时与单条配置合并（冲突时优先使用单条配置）|
| `--split-samples` | str | ✗ | None | 数据集划分和样本数，格式为 `train:10000,test:1000,val:500` |
| `--shuffle` | flag | ✗ | False | 是否对生成的数据进行随机打乱 |
| `--gen_parquet` | flag | ✗ | True | 是否生成parquet格式文件（除jsonl外） |
| `--global-config-overrides` | str | ✗ | None | 全局配置覆盖参数，JSON字符串格式，如 `'{"enable_random_seed": true}'` |
| `--no-tool` | flag | ✗ | False | 开启则不使用任何工具配置 |
| `--no-interaction` | flag | ✗ | False | 开启则不使用任何交互配置 |

##### 3.1.4.2 生成数据格式

执行脚本后会以split为后缀生成多个jsonl数据文件

```json
{
    "data_source": "bootcamp/Example",
    "prompt": [
        {
            "content": "你是一位数学专家，擅长进行算术运算。\n\n任务：计算表达式 88086 ÷ 38856 - 34189 + 96429 ÷ 65083 × 2882 + 21625 × 99240 + 97985 × 61813 ÷ 6044 ÷ 79107 × 68650 + 89020 的结果，误差范围为 1e-4\n\n最终答案格式：请以``json\n{\n    \"result\": your_result\n}\n``格式返回结果，且在必要时使用科学计数法（如1e-4、2.5E+3）。\n\n计算建议：\n1. 请在合适的时机运用算术工具，如需要计算大数等自信程度不高计算时，以避免计算错误和无意义的工具调用。\n2. 若需要计算的表达式较长，请在实际计算开始前，先进行计算规划，以避免计算错误和无意义的工具调用。\n下面请开始计算。",
            "role": "user"
        }
    ],
    "reward_model": {
        "ground_truth": {
            "expression": "88086 ÷ 38856 - 34189 + 96429 ÷ 65083 × 2882 + 21625 × 99240 + 97985 × 61813 ÷ 6044 ÷ 79107 × 68650 + 89020",
            "expected_result": 2146993745.4955528,
            "tolerance": "1e-4"
        },
        "style": "rule"
    },
    "extra_info": {
        "tools_kwargs": {

        },
        "need_tools_kwargs": false,
        "index": 102,
        "split": "test",
        "generator_name": "medium_arithmetic_w_interaction",
        "interaction_kwargs": {
            "name": "example_interaction",
            "identity": {
                "expression": "88086 ÷ 38856 - 34189 + 96429 ÷ 65083 × 2882 + 21625 × 99240 + 97985 × 61813 ÷ 6044 ÷ 79107 × 68650 + 89020",
                "expected_result": 2146993745.4955528,
                "tolerance": "1e-4"
            }
        }
    }
}
```


##### 3.1.4.3 批量数据生成

**示例**: [batch_data_generation.py](/internbootcamp/utils/batch_data_generation.py)

现支持批量并行数据生成：
```bash
# 批量数据生成
python -m internbootcamp.utils.batch_data_generation \
    --bootcamp-registry configs/bootcamp_registry.jsonl \ # Bootcamp注册表
    --max-workers 8 \ # 最大并行进程数
    --output-dir data/batch_generated/ \
    --split-samples train:1000,test:100 \ #每个Bootcamp配置生成1000个训练样本和100个测试样本
    --concat-files \
    --continue-on-error
```

**bootcamp注册表作为批量生成配置文件**（bootcamp_registry.jsonl）：   
单条注册表数据结构：
```json
{"instruction_config_path": "internbootcamp/bootcamps/your_bootcamp/configs/your_instruction_config.yaml", "data_source": "bootcamp/YourBootcamp", "yaml_tool_path": "internbootcamp/bootcamps/your_bootcamp/configs/your_tool_config.yaml", "yaml_interaction_path": "internbootcamp/bootcamps/your_bootcamp/configs/your_interaction_config.yaml", "reward_calculator_class": "internbootcamp.bootcamps.your_bootcamp.your_reward_calculator.YourRewardCalculator"}

```

**参数列表**：

| 参数名称 | 类型 | 必需 | 默认值 | 说明 |
|---------|------|------|--------|------|
| `--bootcamp-registry` | str | ✓ | - | Bootcamp注册表文件路径，jsonl格式，每条记录包含一个bootcamp的配置信息 |
| `--max-workers` | int | ✗ | min(16, CPU核心数) | 最大并行工作进程数，自动限制在CPU核心数范围内 |
| `--continue-on-error` | flag | ✗ | False | 遇到错误时继续执行其他配置，不中断整个批量生成流程 |
| `--log-level` | str | ✗ | INFO | 日志级别，可选值：DEBUG、INFO、WARNING、ERROR |
| `--output-dir` | str | ✗ | data/generated | 输出目录，所有生成的数据集将保存在此目录下 |
| `--split-samples` | str | ✗ | train:100,test:0 | 数据集划分和样本数，格式为 `train:10000,test:1000,val:500` |
| `--concat-files` | flag | ✗ | False | 是否将所有生成的文件按split分别合并到文件中 |
| `--no-tool` | flag | ✗ | False | 是否不使用工具配置，开启则忽略所有工具相关配置 |
| `--no-interaction` | flag | ✗ | False | 是否不使用交互配置，开启则忽略所有交互相关配置 |

### 3.2 工具类开发

**示例**: [example_tools.py](/internbootcamp/bootcamps/example_bootcamp/example_tools.py)

#### 3.2.1 功能职责
- 定义和注册领域工具
- 处理工具调用请求
- 计算中间过程奖励
- 管理工具执行结果

#### 3.2.2 开发指南

若自定义Bootcamp需支持工具训练，须继承`BaseTool`基类，并实现以下两个核心抽象方法：
```python
# 示例代码结构
class CustomTool(BaseTool):
    def __init__(self, config):
        super().__init__(config)
        
    async def create(self, instance_id: Optional[str] = None, identity: dict = None, **kwargs) -> str:
        """用于创建针对每条数据所需要的额外变量，在数据加载阶段被执行。
        Args:
            instance_id (Optional[str]): 针对每个instance的id，不指定时由类自动生成
            identity (dict): 每条数据所需要的额外变量，data source应该有identity字段
        Returns:
            instance_id: str
        """
        # 创建工具实例
        pass
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        """工具的执行逻辑
        Args:
            instance_id (str): 
            parameters (dict[str, Any]): 执行工具所需的参数，parameters均有LLM的输出提供

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response (str): 工具的输出
            tool_reward_score (float): 工具计算的reward结果，如有，否则返回0
            tool_metrics (dict): 返回{}
        """
        # 执行工具逻辑
        pass
```

**关键要点：**
- `create()`主要用于实现预定义tool环境变量，或初始化tool环境需维护的变量，这些变量不会暴露给模型。

#### 3.2.3 配置文件定义

```yaml
# custom_tool_config.yaml
tools:
  - class_name: "internbootcamp.bootcamps.your_bootcamp.your_tools.CustomTool"
    config:
      type: "native"
    tool_schema: # tool_schema会作为prompt被输入LLM
      type: "function"
      function:
        name: ""
        description: ""
        parameters:
          type: "object"
          properties:
            param1:
              type: ""
              description: ""
            param2:
              type: ""
              description: ""
          required: [] # 哪些输入参数是强制需要的
```

### 3.3 交互类开发

**示例**: [example_interaction.py](/internbootcamp/bootcamps/example_bootcamp/example_interaction.py)

#### 3.3.1 功能职责
- **回合式交互控制**: 在工具多轮调用之后，向模型返回"反馈/追问"，并决定是否终止该样本的交互序列。
- **回合级得分**: 可基于当前模型输出计算本回合分数（如即时奖励/提示性奖励），并在需要时累计。
- **状态管理**: 通过`instance_id`维护每个样本的交互上下文（如标准答案`identity`、中间状态等）。
- **与工具协同**: 交互类通常在一轮或多轮工具调用后介入，形成"模型 → 工具 → 交互 → 模型"的闭环。

#### 3.3.2 开发指南

- 继承`BaseInteraction`并实现核心方法：
  - `start_interaction(instance_id, identity, **kwargs)`：初始化交互会话，保存`identity`等上下文。
  - `generate_response(instance_id, messages, **kwargs)`：产出"反馈/追问"文本，并返回是否应终止、回合得分等。
  - `calculate_score(instance_id, **kwargs)`：如需单独计算回合得分，可实现。
  - `finalize_interaction(instance_id, **kwargs)`：释放会话资源。
- 可参考`internbootcamp/bootcamps/example_bootcamp/example_interaction.py`，其使用`ExampleRewardCalculator`对最近一次assistant输出进行判分，并在满分时终止交互。

#### 3.3.3 接口规范

```python
class BaseInteraction:
    async def start_interaction(self, instance_id: Optional[str] = None, identity: dict[str, Any] = None, **kwargs) -> str: ...

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Returns:
          - should_terminate_sequence: 是否终止该样本后续交互
          - response_content: 返回给模型的反馈/追问内容（将以user角色注入下一轮）
          - current_turn_score: 本回合得分
          - additional_data: 额外元信息
        """

    async def calculate_score(self, instance_id: str, **kwargs) -> float: ...
    async def finalize_interaction(self, instance_id: str, **kwargs) -> bool: ...
```

#### 3.3.4 配置文件定义

```yaml
# example_interaction_config.yaml
interaction:
  - name: "example_interaction"
    class_name: "internbootcamp.bootcamps.example_bootcamp.example_interaction.ExampleInteraction"
    config: {}
```
- 仅支持配置一个交互类（内部会校验长度为1）。
- `name`可选；不填则默认使用类名派生的名称。

#### 3.3.5 数据格式约定

- 训练/评测时，`Evaluator`会从数据样本的`extra_info.interaction_kwargs.identity`中取出`identity`传入`start_interaction`。
- 建议在构造数据时写入如下字段：
```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {"ground_truth": {"identity": {"...": "..."}}, "style": "rule"},
  "extra_info": {
    "interaction_kwargs": {
      "identity": {"...": "标准答案或判分所需信息 ..."}
    }
  }
}
```
- 在`ExampleInteraction`中，会用`self._instance_dict[instance_id]["identity"]`结合最近的assistant输出进行打分并决定是否终止。

#### 3.3.6 运行时集成

命令行评测时通过交互配置启用交互闭环：
```bash
python -m internbootcamp.utils.run_evaluation \
  --dataset-path data/your_bootcamp/test.jsonl \
  --output-dir outputs/ \
  --api-key "$API_KEY" \
  --api-url "$API_URL" \
  --api-model "gpt-3.5-turbo" \
  --reward-calculator-class "internbootcamp.bootcamps.example_bootcamp.example_reward_calculator.ExampleRewardCalculator" \
  --tool-config configs/your_tool_config.yaml \
  --interaction-config internbootcamp/bootcamps/example_bootcamp/configs/example_interaction_config.yaml \
  --max-tool-turns-per-interaction 5 \
  --max-interaction-turns 3
```

#### 3.3.7 示例实现

- `internbootcamp/bootcamps/example_bootcamp/example_interaction.py`：
  - 取最近一次assistant输出作为候选答案。
  - 通过`ExampleRewardCalculator.verify_score(...)`打分。
  - `score == 1.0`时返回`should_terminate_sequence=True`并给出正向反馈；否则给出纠错性反馈并继续下一轮。

### 3.4 分布式工具服务器

**示例**: [分布式工具服务器文档](/internbootcamp/utils/tool_server/README.md) | [DLC启动脚本](/internbootcamp/bootcamps/AnalogCircuitSizing/scripts/dlc_launch_bootcamp_tool_server.sh)

#### 3.4.1 架构概述

基于Master-Worker分布式架构的工具服务器，支持工具在多机器间分布式部署和执行，提供高并发处理能力和负载均衡。

#### 3.4.2 运行模式

##### 3.4.2.1 Master模式（生产推荐）

启动Master服务器，等待Worker注册：
```bash
python -m internbootcamp.utils.tool_server.cli \
    --mode master \
    --tools_yaml_path config.yaml \
    --port 8000 \
    --output_dir outputs/
```

##### 3.4.2.2 Worker模式（生产推荐）

在其他机器上启动Worker并注册到Master：  
指定一个tools_config:
```bash
python -m internbootcamp.utils.tool_server.cli \
    --mode worker \
    --tools_yaml_path config.yaml \
    --master_url http://master_ip:master_port \
    --port 8001 \ # Worker服务器的起始端口号，系统会在端口的高位端口中为多个Worker智能分配可用端口
    --num_workers 3 # Worker服务器数量
```
指定一个bootcamp注册表：
```bash
python -m internbootcamp.utils.tool_server.cli \
    --mode worker \
    --bootcamp_registry configs/bootcamp_registry.jsonl \
    --master_url http://master_ip:master_port \
    --port 8001 \ # Worker服务器的起始端口号，系统会在端口的高位端口中为多个Worker智能分配可用端口
    --num_workers 3 # Worker服务器数量
```

##### 3.4.2.3 Unified模式（调试推荐）

在单机上同时启动Master和多个Worker：
```bash
python -m internbootcamp.utils.tool_server.cli \
    --mode unified \
    --tools_yaml_path config.yaml \
    --port 8000 \
    --num_workers 5 \
    --keep_running \
    --test_servers \
    --log_dir ./logs/
```

#### 3.4.3 核心特性

**1. 动态Worker注册与负载均衡**：
- Worker可动态注册到Master，支持热插拔
- Master自动将请求分配到负载最轻的Worker
- 支持实例ID到Worker的路由映射

**2. 健康监控与容错**：
- Master监控Worker健康状态，自动清理死掉的Worker
- 支持Worker断线重连和故障恢复
- 详细的健康检查和状态监控

**3. 批量工具配置支持**：
```bash
# 使用bootcamp注册表批量加载所有工具
python -m internbootcamp.utils.tool_server.cli \
    --mode unified \
    --bootcamp_registry configs/bootcamp_registry.jsonl \
    --port 8000 \
    --num_workers 8
```

**4. 配置自动生成**：
- 自动生成适配分布式架构的MCP工具配置文件
- 智能端口分配和外网IP获取
- 工具类自动更新为BaseMCPTool

**5. 实时可视化监控仪表板**：
- Master服务器提供Web界面实时监控系统状态
- 自动刷新显示Worker在线状态、工具可用性和负载分布
- 可视化展示活跃实例数、心跳状态和实例映射关系
- 支持通过浏览器直观查看整个分布式系统的运行情况

#### 3.4.4 命令行参数

| 参数名称 | 类型 | 必需 | 默认值 | 说明 |
|---------|------|------|--------|------|
| `--mode` | str | ✓ | - | 运行模式：master/worker/unified |
| `--tools_yaml_path` | str | ✗ | - | 工具配置YAML文件路径 |
| `--bootcamp_registry` | str | ✗ | - | Bootcamp注册表文件路径，可批量加载所有工具 |
| `--port` | int | ✗ | 8000 | 服务器端口号或起始端口号 |
| `--host` | str | ✗ | 0.0.0.0 | 服务器主机地址 |
| `--master_url` | str | ✗ | - | Master服务器URL（Worker模式必需） |
| `--worker_id` | str | ✗ | 自动生成 | Worker ID标识 |
| `--num_workers` | int | ✗ | 3 | Worker服务器数量 |
| `--output_dir` | str | ✗ | 输入配置文件目录 | 配置文件输出目录 |
| `--updated_tool_class` | str | ✗ | BaseMCPTool | 更新后的工具类（不建议使用） |
| `--timeout_per_query` | int | ✗ | 60 | 单个查询超时时间（秒），影响训练、评测过程，建议设置得更宽容 |
| `--keep_running` | flag | ✗ | False | 保持服务器运行（unified模式） |
| `--log_dir` | str | ✗ | - | 日志目录路径（unified模式） |
| `--test_servers` | flag | ✗ | False | 启动后自动测试服务器（unified模式） |
| `--test_timeout` | int | ✗ | 10 | 测试超时时间（秒）（unified模式） |
| `--connectivity_only` | flag | ✗ | False | 仅测试连通性 （unified模式）|

#### 3.4.5 配置文件转换

**原始工具配置**：
```yaml
tools:
  - class_name: "internbootcamp.bootcamps.example_bootcamp.example_tools.ArithmeticTool"
    config: 
      type: "native"
    tool_schema:
      # ... tool schema定义
```

**自动生成的工具配置**：  
使用该配置进行后续的训练与评测
```yaml
tools:
  - class_name: "internbootcamp.src.base_mcp_tool.BaseMCPTool"
    config:
      type: "native"
      mcp_server_url: "http://IP:PORT/ArithmeticTool"
      timeout_per_query: 60
    tool_schema:
      # ... 保持原有tool schema
```

#### 3.4.6 监控与管理

**可视化仪表板**：

Master服务器提供了一个实时监控仪表板，可通过浏览器访问Master服务器的根路径查看：

```
http://<master_ip>:<master_port>/
```

**仪表板功能特性**：

1. **实时自动刷新**：每3秒自动刷新一次，无需手动刷新页面

2. **关键指标展示**：
   - **Worker节点数**：显示在线Worker数量/总Worker数量
   - **可用工具数**：显示当前系统中注册的工具总数
   - **活跃实例数**：实时显示正在处理的实例数量（高亮显示）
   - **服务状态**：Master服务器的运行状态

3. **Worker详细信息**：
   - Worker ID和在线状态（🟢在线 / 🔴离线）
   - Worker URL地址
   - 支持的工具列表
   - 当前活跃实例数（突出显示）
   - 最后心跳时间
   - 主机信息（主机名、IP地址）
   - 注册时间

4. **工具可用性监控**：
   - 显示所有已注册工具
   - 每个工具的可用Worker数量
   - 工具状态标识（可用/不可用）

5. **负载均衡可视化**：
   - 实时查看各Worker的实例分布
   - 帮助识别负载不均的情况
   - 监控Worker健康状态

**健康检查API**：

除了可视化仪表板，还提供JSON格式的健康检查接口，便于程序化监控：

```bash
curl http://<master_ip>:<master_port>/health
```

返回信息包括：
- 服务状态
- 已注册的Worker列表及其详细信息
- 工具列表
- 实例映射关系
- Worker心跳状态

**使用示例**：

```bash
# 启动Master服务器后，在浏览器中访问
http://MASTER_IP:MASTER_PROT/

# 或通过API查询状态
curl http://MASTER_IP:MASTER_PROT/health | jq
```


### 3.5 奖励计算器开发

**示例**: [example_reward_calculator.py](/internbootcamp/bootcamps/example_bootcamp/example_reward_calculator.py)

#### 3.5.1 功能职责
- 从模型输出中提取关键信息
- 验证任务完成度并计算奖励分数
- 支持容错和部分奖励机制

#### 3.5.2 开发指南

继承`BaseRewardCalculator`基类并实现两个核心抽象方法：

```python
from internbootcamp.src.base_reward_calculator import BaseRewardCalculator

class CustomRewardCalculator(BaseRewardCalculator):
    """自定义奖励计算器"""
    
    @staticmethod
    def extract_output(output_str: str):
        """
        从模型输出中提取关键信息用于奖励计算
        
        Args:
            output_str: 模型的最终响应字符串
        
        Returns:
            提取的信息，作为_verify_correction()的extract_solution参数
        """
        # 实现输出解析逻辑，如正则表达式、JSON解析等
        pass
    
    @classmethod
    def _verify_correction(cls, extract_solution, identity: dict, **kwargs) -> float:
        """
        验证提取的解决方案并计算正确性分数
        
        Args:
            extract_solution: 从extract_output()提取的信息
            identity: 任务标准答案信息（来自InstructionGenerator.case_generator()）
            kwargs: 额外关键字参数，可在评测和训练时传递，可用于控制reward计算逻辑
        
        Returns:
            float: 正确性分数（0-1之间）
        """
        # 实现验证逻辑，对比提取结果与标准答案
        pass
```

## 4. 模型评估

### 4.1 评估配置

**示例**: [example_multiturn_w_tool_grpo.sh](/internbootcamp/bootcamps/example_bootcamp/examples/example_multiturn_w_tool_grpo.sh)

#### 4.1.1 基础评估配置

使用统一的评估脚本对自定义Bootcamp进行API call形式的模型性能评估：

```bash
python -m internbootcamp.utils.run_evaluation \
    --dataset-path "数据集路径(jsonl/parquet格式)" \
    --output-dir "评估结果输出目录" \
    --api-key "API密钥" \
    --api-url "API地址" \
    --api-model "模型名称" \
    --api-extra-headers "Authorization:Bearer sk-xxx,Custom-Header:Value" \
    --api-extra-params '{"temperature":0.7, "max_completion_tokens":65536, "extra_body": {"enable_thinking": true}}' \
    --verify-correction-kwargs '{"strict":true}' \
    --evaluator-class "评估器类路径,默认使用BaseEvaluator" \
    --reward-calculator-class "奖励计算器类路径" \
    --tool-config "工具配置文件路径" \
    --interaction-config "交互配置文件路径" \
    --max-assistant-turns "最大assistant消息数" \
    --max-user-turns "最大user消息数" \
    --max-concurrent "最大并发数量" \
    --verbose \ # 详细输出
    --dry-run \ # 干运行模式
    --tokenizer-path "tokenizer路径" \
    --bootcamp-registry "bootcamp注册表路径" \
    --resume-from-result-path "断点重试文件路径"
```

#### 4.1.2 核心功能

**1. 断点重试机制**：
```bash
# 从已有结果文件恢复评测
python -m internbootcamp.utils.run_evaluation \
    --dataset-path data/test.jsonl \
    --output-dir results/ \
    --api-key sk-xxx \
    --resume-from-result-path results/gpt-3.5-turbo/eval_results_20240315_143022.jsonl
```
- 自动检测已完成样本，跳过重复评测
- 支持评测中断后无缝续接
- 智能文件路径管理

**2. 多格式数据集支持**：

- **JSONL**: 支持标准jsonlines格式
- **Parquet**: 支持Apache Parquet格式，自动处理数据类型转换
- **JSON**: 支持单个或数组格式的JSON文件

**3. 批量评测支持**：
```bash
# 使用bootcamp注册表进行批量评测
python -m internbootcamp.utils.run_evaluation \
    --bootcamp-registry configs/bootcamp_registry.jsonl \
    --output-dir results/batch_evaluation/
```

#### 4.1.3 参数列表

| 参数名称 | 类型 | 必需 | 默认值 | 说明 |
|---------|------|------|--------|------|
| `--dataset-path` | str | ✓ | - | 数据集文件路径，支持.json、.jsonl、.parquet格式 |
| `--output-dir` | str | ✓ | - | 评测结果输出目录 |
| `--api-key` | str | ✓ | - | API密钥，用于模型调用认证 |
| `--api-url` | str | ✗ | - | API服务地址，不指定则使用默认OpenAI API |
| `--api-model` | str | ✗ | gpt-3.5-turbo | 模型名称 |
| `--api-extra-headers` | str | ✗ | - | 额外API头部，格式："key1:value1,key2:value2" |
| `--api-extra-params` | str | ✗ | - | 额外模型参数；支持 JSON 字符串或 `@文件`（JSON）。示例：`'{"temperature":0.7, "max_completion_tokens":65536, "extra_body": {"enable_thinking": true}}'` 或 `@/path/to/params.json`；兼容旧格式 `"temperature:0.7,max_tokens:2048"` 作为回退解析 |
| `--verify-correction-kwargs` | str | ✗ | - | 传递给奖励计算器 `verify_correction` 的额外参数；支持格式同 `--api-extra-params` |
| `--evaluator-class` | str | ✗ | BaseEvaluator | 评测器类路径，用于自定义评测逻辑 |
| `--reward-calculator-class` | str | ✗ | - | 奖励计算器类路径，用于计算任务完成度 |
| `--tool-config` | str | ✗ | - | 工具配置YAML文件路径，支持原生工具和MCP工具 |
| `--interaction-config` | str | ✗ | - | 交互配置YAML文件路径，启用多轮交互功能 |
| `--max-assistant-turns` | int | ✗ | 5 | assistant 响应的最大轮次 |
| `--max-user-turns` | int | ✗ | 20 | user 输入的最大轮次（包括 tool response, interaction response） |
| `--max-concurrent` | int | ✗ | 1 | 最大并发数，控制并行评测样本数量 |
| `--verbose` | flag | ✗ | False | 输出详细信息，用于调试和监控 |
| `--dry-run` | flag | ✗ | False | 只验证配置不实际运行，用于配置测试 |
| `--tokenizer-path` | str | ✗ | - | tokenizer路径，用于apply template时的文本处理 |
| `--bootcamp-registry` | str | ✗ | - | bootcamp注册表路径，用于批量评测 |
| `--resume-from-result-path` | str | ✗ | - | 断点重试文件路径，从中断点恢复评测 |

### 4.2 评估输出

评估完成后将在输出目录生成：
- 详细的对话记录和工具调用结果
- 奖励分数统计和分析报告
- 模型性能分析报告
- 断点重试支持的结果文件

### 4.3 评估数据后处理

**示例**: [data_postprocess.py](/internbootcamp/utils/data_postprocess.py)

#### 4.3.1 功能概述

数据后处理工具用于对evaluator输出的jsonl文件进行灵活的过滤和转换，支持将评估结果转换为可用于训练的数据格式。

**核心特性**：
- **可插拔过滤器**: 支持添加多个过滤函数，筛选符合条件的样本
- **灵活转换**: 支持一对一或一对多的数据转换
- **预定义函数**: 提供常用的过滤和转换函数
- **统计信息**: 自动统计处理过程中的数据变化

**主要应用场景**：
- 从评估结果中筛选高质量样本用于训练
- 提取特定分数范围的数据
- 将多轮对话展开为前缀集以支持多样化训练
- 转换数据格式以适配训练流程

#### 4.3.2 使用方式

**命令行方式**（推荐快速使用）：

```bash
# 方式1: 自动生成输出路径（添加_processed后缀）
python internbootcamp/utils/data_postprocess.py \
    path/to/eval_results.jsonl \
    --expand-messages-prefixes \
    --extract-training \
    --min-score 0.9 \
    --max-score 1.0

# 方式2: 指定输出路径
python internbootcamp/utils/data_postprocess.py \
    path/to/eval_results.jsonl \
    path/to/output.jsonl \
    --filter-success \
    --min-score 0.9 \
    --data-source bootcamp/example
```

**预定义过滤和转换功能**：
- `--filter-success`: 只保留成功的样本
- `--min-score`: 设置最小分数阈值（默认0.9）
- `--max-score`: 设置最大分数阈值（默认1.0）
- `--data-source`: 按数据源过滤
- `--expand-messages-prefixes`: 将多轮对话展开为前缀集（重要：用于生成符合常见reasoning模型chat template策略的训练数据）
- `--extract-training`: 提取训练数据格式
- `--extract-messages`: 只提取消息内容

#### 4.3.3 命令行参数

| 参数名称 | 类型 | 必需 | 默认值 | 说明 |
|---------|------|------|--------|------|
| `input` | str | ✓ | - | 输入jsonl文件路径 |
| `output` | str | ✗ | 自动生成 | 输出jsonl文件路径，不提供则自动生成为 `{input}_processed.jsonl` |
| `--filter-success` | flag | ✗ | False | 只保留success字段为True的样本 |
| `--min-score` | float | ✗ | 0.9 | 最小分数阈值（包含） |
| `--max-score` | float | ✗ | 1.0 | 最大分数阈值（包含） |
| `--data-source` | str | ✗ | - | 按指定数据源过滤 |
| `--extract-training` | flag | ✗ | False | 提取训练数据格式（包含data_source、prompt、messages、tools等字段） |
| `--extract-messages` | flag | ✗ | False | 只提取消息内容（messages、score、success字段） |
| `--expand-messages-prefixes` | flag | ✗ | False | 将多轮对话展开为前缀集，每个assistant消息生成一条样本 |

#### 4.3.4 编程接口

对于更复杂的数据处理需求，可以使用编程接口自定义过滤和转换逻辑：

```python
from internbootcamp.utils.data_postprocess import (
    DataPostProcessor,
    filter_by_success,
    filter_by_score,
    extract_for_training,
    expand_messages_prefixes
)

# 创建处理器
processor = DataPostProcessor()

# 添加过滤器
processor.add_filter(filter_by_success, name="success")
processor.add_filter(filter_by_score(min_score=0.9, max_score=1.0), name="score")

# 添加自定义过滤器
processor.add_filter(
    lambda x: len(x.get("messages", [])) > 2,
    name="min_messages"
)

# 添加转换器
processor.add_transformer(expand_messages_prefixes, name="expand")
processor.add_transformer(extract_for_training, name="training_format")

# 自定义转换器（一对一）
def add_metadata(data):
    data["processed_at"] = "2024-01-01"
    return data

processor.add_transformer(add_metadata, name="metadata")

# 自定义转换器（一对多）
def split_by_turns(data):
    messages = data.get("messages", [])
    return [{"turn": i, "msg": msg, "original_score": data.get("score")} 
            for i, msg in enumerate(messages)]

processor.add_transformer(split_by_turns, name="split")

# 执行处理
stats = processor.process(
    input_path="eval_results.jsonl",
    output_path="processed_results.jsonl",
    verbose=True
)

print(f"处理完成: {stats}")
```

**高级功能**：

```python
# 创建字段提取器
from internbootcamp.utils.data_postprocess import create_field_extractor

extractor = create_field_extractor(
    "input.data_source",
    "score",
    "messages",
    "input.extra_info.generator_name"
)
processor.add_transformer(extractor)

# 创建自定义转换器（支持字段重命名和默认值）
from internbootcamp.utils.data_postprocess import create_custom_transformer

transformer = create_custom_transformer({
    "text": "messages[-1].content",  # 提取最后一条消息
    "source": "input.data_source",
    "score": ("score", 0.0),  # 带默认值
    "custom_field": lambda x: x.get("score", 0) * 100  # 自定义函数
})
processor.add_transformer(transformer)
```

**处理流程示例**：

```bash
# 完整的数据处理流程：评估 → 后处理 → 训练
# 1. 运行评估
python -m internbootcamp.utils.run_evaluation \
    --dataset-path data/test.jsonl \
    --output-dir outputs/eval/

# 2. 后处理评估结果
python internbootcamp/utils/data_postprocess.py \
    outputs/eval/gpt-3.5-turbo/eval_results_20240315.jsonl \
    outputs/train_data/processed.jsonl \
    --filter-success \
    --min-score 0.9 \
    --expand-messages-prefixes \
    --extract-training

# 3. 使用处理后的数据进行训练
# (训练命令...)
```

## 5. 总结

本手册提供了创建自定义Bootcamp的完整指导。通过遵循这些步骤和最佳实践，您将能够成功构建一个高质量的专业领域Bootcamp系统。

如有任何问题或建议，请参考项目文档或联系开发团队。
