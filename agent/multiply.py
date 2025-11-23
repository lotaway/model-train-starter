

# @TODO session checkpoint for shared memory to multiply agents
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_swarm import create_handoff_tool, create_swarm

checkpointer = MemorySaver()
long_term_store = InMemoryStore()
swarm_workflow = create_swarm(
    agents=[
        general_qa_agent_with_handoff,
        nl2sql_agent_with_handoff,
        invoice_information_agent_with_handoff
    ],
    default_active_agent="general_qa_agent_with_handoff"  # 默认从通用QA开始
)
# 编译并添加持久化
swarm_agents = swarm_workflow.compile(
    checkpointer=checkpointer  # 保存会话状态
    store=long_term_store  # 客户偏好等信息长期保存
)

# 给每个Agent配备切换工具
general_qa_agent_with_handoff = create_react_agent(
    model=model,
    tools=[
        *general_qa_tool,  # 原有的RAG工具
        create_handoff_tool(
            agent_name="invoice_information_agent_with_handoff",
            description="切换到发票专家处理账单问题"
        ),
        create_handoff_tool(
            agent_name="nl2sql_agent_with_handoff",
            description="切换到数据库专家查询数据"
        ),
    ],
    name="general_qa_agent_with_handoff",
    prompt=general_qa_subagent_prompt
)

# NL2SQL Agent也能切换（双向的）
nl2sql_agent_with_handoff = create_react_agent(
    model=model,
    tools=[
        *nl2sql_tool,
        create_handoff_tool(
            agent_name="general_qa_agent_with_handoff",
            description="切换回通用QA处理知识问题"
        ),
        create_handoff_tool(
            agent_name="invoice_information_agent_with_handoff",
            description="切换到发票专家"
        ),
    ],
    name="nl2sql_agent_with_handoff",
    prompt=nl2sql_subagent_prompt
)

# Invoice Agent同样可以切换
invoice_information_agent_with_handoff = create_react_agent(
    model=model,
    tools=[
        *invoice_tools,
        create_handoff_tool(
            agent_name="general_qa_agent_with_handoff",
            description="切换到通用QA"
        ),
        create_handoff_tool(
            agent_name="nl2sql_agent_with_handoff", 
            description="切换到数据库专家"
        ),
    ],
    prompt=invoice_subagent_prompt,
    name="invoice_information_agent_with_handoff"
)