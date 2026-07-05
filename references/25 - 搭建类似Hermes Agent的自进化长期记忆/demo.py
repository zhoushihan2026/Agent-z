import os
from agently import Agently


def configure_model() -> None:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "需要 DEEPSEEK_API_KEY（放在课程根目录 .env 或 shell 导出）。"
        )
    Agently.set_settings(
        "OpenAICompatible",
        {
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key": api_key,
            "model": os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
        },
    )


configure_model()
requester = Agently.create_agent()

# result = requester.input("帮我记一下，我今天要去超市买三个鸡蛋").start()

# print("轮次1：", result)

rules = ["我不喜欢长篇大论，但我喜欢你在回复里用emoji如😄🎇等进行回复"]
extra_info = "Todo list: [ ]今天要去超市买三个鸡蛋"
memory = ""
instruction = "暂无"
query = "刚才我们说了什么？"
chat_history = [
    {
        "role": "system",
        "content": rules,
    },
    {"role": "user", "content": "帮我记一下，我今天要去超市买三个鸡蛋"},
    {
        "role": "assistant",
        "content": "好的，已经记下了！随时告诉我哦～ 🥚",
    },
    # {
    #     "role": "user",
    #     "content": "刚才我们说了什么？",
    # },
]

prompt_length = (
    len(str(chat_history))
    + len(extra_info)
    + len(memory)
    + len(instruction)
    + len(query)
)

if prompt_length > 100000:
    ...


result = (
    requester.set_chat_history(chat_history)  # type: ignore
    .input(
        f"信息补充：{ extra_info }\n"
        f"重要记忆：{ memory }\n"
        f"本次处理时应该注意：{ instruction }\n"
        f"用户问题：{ query }\n"
    )
    .start()
)

print(result)
