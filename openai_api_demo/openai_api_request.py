from openai import OpenAI


base_url = "http://0.0.0.0:7776/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)


def simple_chat(use_stream=False):
    messages = [
        {
            "role": "user",
            "content": "四大名著分别是什么？"
        },
        {
            "role": "assistant",
            "content": "中国四大名著是指《红楼梦》、《西游记》、《水浒传》和《三国演义》。这四部小说在中国文学史上具有极高的地位，被广泛传阅和研究。"
        },
        {
            "role": "user",
            "content": "这四部小说的作者分别是谁？"
        }
    ]

    response = client.chat.completions.create(
        model="bluelm-7b",
        messages=messages,
        stream=use_stream,
        max_tokens=1024,
        presence_penalty=1.1,
        temperature=0.,
    )

    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content)
        else:
            content = response.choices[0].message.content
            print(content)
    else:
        print("Error:", response.status_code)


def function_chat():
    """
        messages example:
        [
            {
                "role": "user",
                "content": "深圳今天天气怎么样？"
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "current-weather",
                        "arguments": {"city": "Shenzhen"},
                    }
                ]
            },
            {
                "role": "function",
                "content": json.dumps({"result": 20})
            }
        ]
    """
    messages = [
        {
            "role": "user",
            "content": "深圳今天天气怎么样？"
        }
    ]
    tools = [
        {
            "name": "text-translation-en2zh",
            "description": "将输入的英文文本翻译成中文",
            "parameters": [{
                "name": "text",
                "description": "用户输入的英文文本",
                "required": 'True'
            }]
        },
        {
            "name": "text-address",
            "description": "针对中文的地址信息，识别出里面的元素",
            "parameters": [{
                "name": "text",
                "description": "用户输入的地址信息",
                "required": 'True'
            }]
        },
        {
            "name": "current-weather",
            "description": "根据给出的城市，查询即时天气信息",
            "parameters": [{
                "name": "city",
                "description": "用户输入的城市信息",
                "required": 'True'
            }]
        }
    ]
    response = client.chat.completions.create(
        model="bluelm-7b",
        messages=messages,
        max_tokens=256,
        presence_penalty=1.1,
        temperature=0.,
        tools=tools,
    )
    if response:
        content = response.choices[0].message.content
        print(content)
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    simple_chat(use_stream=False)
    simple_chat(use_stream=True)
    function_chat()
