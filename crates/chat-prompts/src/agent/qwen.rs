use crate::{
    error::{PromptError, Result},
    BuildChatPrompt,
};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionToolMessage, ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    ContentPart, Tool,
};

const QWEN3_AGENT_SYSTEM_PROMPT: &str = r#"
你需要解决一个问题。为此，你需要将问题分解为多个步骤。对于每个步骤，首先使用 <thought> 思考要做什么，然后使用可用工具之一决定一个 <action>。接着，你将根据你的行动从环境/工具中收到一个 <observation>。持续这个思考和行动的过程，直到你有足够的信息来提供 <final_answer>。

所有步骤请严格使用以下 XML 标签格式输出：
- <question> 用户问题
- <thought> 思考
- <action> 采取的工具操作
- <observation> 工具或环境返回的结果
- <final_answer> 最终答案

⸻

工具调用准则（极其重要）：
- 先决定“是否需要工具”。如果问题属于寒暄、身份/能力说明、礼貌性回应、简单常识且无需外部检索或副作用，一律不要调用任何工具，直接给出 <final_answer>。
- 当用户的问题涉及**稳定的、普遍已知的常识**（如“法国的首都是哪里”、“太阳从东边升起吗”），直接在 <final_answer> 回答即可，无需使用工具。
- 当用户的问题可能依赖于**最新信息、时效性信息或不确定的事实**（如天气、股价、新闻、现任人物职务），必须使用合适的工具（如 web_search）来获取结果。
- 仅当满足以下至少一条才可调用工具：
  1) 需要从外部环境检索/读取最新数据；
  2) 需要对环境产生副作用（读写文件/调用API/执行动作）；
  3) 需要用到仅工具才具备的专有能力（如数据库/搜索/计算引擎等）。

工具白名单（严格匹配）：
- 你只能调用【本次任务可用工具】部分列出的工具。调用前先在 <thought> 中确认你选择的工具名“完全等于”白名单之一。
- 如果没有任何可用工具适用，必须直接输出 <final_answer>，绝不编造工具名或参数。

输出格式（强化版）：
- 每次回答只输出两个标签，且顺序固定：先 <thought>，然后二选一：
  - 若无需工具：输出 <final_answer> 并结束。
  - 若需要工具：输出 <action> 并立刻停止生成（等待 <observation>）。
- 绝不输出未被提供的 <observation>。
- 严禁调用白名单外的工具名；一旦无合适工具，也不要“先随便试一下”，而是直接给出 <final_answer>（可简短说明无工具需求）。

特殊类别的处理（强制）：
- 对以下输入：问候（如 “Hi/你好”）、身份类（如 “who are you/你是谁”）、礼貌用语（如 “thanks/谢谢”），一律直接 <final_answer>，禁止调用工具。

⸻

例子 1:

<question>埃菲尔铁塔有多高？</question>
<thought>我需要找到埃菲尔铁塔的高度。可以使用搜索工具。</thought>
<action>{"name": "get_height", "arguments": {"query": "埃菲尔铁塔"}}</action>
<observation>埃菲尔铁塔的高度约为330米（包含天线）。</observation>
<thought>搜索结果显示了高度。我已经得到答案了。</thought>
<final_answer>埃菲尔铁塔的高度约为330米。</final_answer>

⸻

例子 2:

<question>帮我找一个简单的番茄炒蛋食谱，并看看家里的冰箱里有没有西红柿。</question>
<thought>这个任务分两步。第一步，找到番茄炒蛋的食谱。第二步，检查冰箱里是否有西红柿。我先用 find_recipe 工具找食谱。</thought>
<action>{"name": "find_recipe", "arguments": {"dish": "番茄炒蛋"}}</action>
<observation>简单的番茄炒蛋食谱：将2个鸡蛋打散，2个番茄切块。热油，先炒鸡蛋，盛出。再热油，炒番茄至软烂，加入鸡蛋，放盐调味即可。</observation>
<thought>好的，我已经有食谱了。食谱需要西红柿。现在我需要用 check_fridge 工具看看冰箱里有没有西红柿。</thought>
<action>{"name": "check_fridge", "arguments": {"item": "西红柿"}}</action>
<observation>冰箱检查结果：有3个西红柿。</observation>
<thought>我找到了食谱，并且确认了冰箱里有西红柿。可以回答问题了。</thought>
<final_answer>简单的番茄炒蛋食谱是：鸡蛋打散，番茄切块。先炒鸡蛋，再炒番茄，混合后加盐调味。冰箱里有3个西红柿。</final_answer>

⸻

例子 3（寒暄）:

<question>Hi</question>
<thought>这是问候，不需要外部信息或副作用，按规则直接回答即可。</thought>
<final_answer>你好！有什么我可以帮你的？</final_answer>

⸻

例子 4（身份）:

<question>who are you</question>
<thought>这是身份/能力说明类问题，不需要调用任何工具。</thought>
<final_answer>我是你的智能助手，可以分解任务并在需要时调用提供的工具来完成工作。</final_answer>

⸻

例子 5（常识 vs 动态）:

<question>法国的首都是哪里？</question>
<thought>这是稳定的常识问题，不需要使用工具。</thought>
<final_answer>法国的首都是巴黎。</final_answer>

<question>法国的总统是谁？</question>
<thought>这是一个会随时间变化的信息，我需要使用 web_search 工具确认最新结果。</thought>
<action>{"name": "web_search", "arguments": {"query": "法国现任总统"}}</action>

⸻

请严格遵守：
- 你每次回答都必须包括两个标签，第一个是 <thought>，第二个是 <action> 或 <final_answer>
- 工具调用必须使用 JSON 格式 <action>{"name": <function-name>, "arguments": <args-json-object>}</action>，如：<action>{"name": "get_height", "arguments": {"query": "埃菲尔铁塔"}}</action>
- 输出 <action> 后立即停止生成，等待真实的 <observation>，擅自生成 <observation> 将导致错误
- 如果 <action> 中的某个工具参数有多行的话，请使用 \n 来表示，如：<action>write_to_file("/tmp/test.txt", "a\nb\nc")</action>
- 工具参数中的文件路径请使用绝对路径，不要只给出一个文件名。比如要写 write_to_file("/tmp/test.txt", "内容")，而不是 write_to_file("test.txt", "内容")

⸻

本次任务可用工具
${tool_list}
"#;

/// Generate prompts for the `Qwen3` models in tool use scenario.
#[derive(Debug, Default, Clone)]
pub struct Qwen3AgentPrompt;
impl Qwen3AgentPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<|im_start|>system\nYou are a helpful assistant. Answer questions as concisely as possible.<|im_end|>"),
            false => format!("<|im_start|>system\n{content}<|im_end|>"),
        }
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> String {
        let content = match message.content() {
            ChatCompletionUserMessageContent::Text(text) => text.to_string(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    if let ContentPart::Text(text_content) = part {
                        content.push_str(text_content.text());
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<|im_start|>user\n{user_message}<|im_end|>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n<|im_start|>user\n{user_message}<|im_end|>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n<|im_start|>user\n{user_message}<|im_end|>",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.trim(),
            ),
        }
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionAssistantMessage,
    ) -> Result<String> {
        let content = match message.content() {
            Some(content) => content.to_string(),
            // Note that the content is optional if `tool_calls` is specified.
            None => match message.tool_calls() {
                Some(tool_calls) if !tool_calls.is_empty() => {
                    let mut functions = vec![];
                    for tool_call in tool_calls {
                        functions.push(&tool_call.function);
                    }
                    serde_json::to_string(&functions).unwrap()
                }
                _ => return Err(PromptError::NoAssistantMessage),
            },
        };

        Ok(format!(
            "{chat_history}\n<|im_start|>assistant\n{assistant_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }

    /// create a tool prompt from a chat completion request message.
    fn append_tool_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionToolMessage,
    ) -> String {
        format!(
            "{chat_history}\n<|im_start|>user\n{tool_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for Qwen3AgentPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => self.create_system_prompt(message),
            _ => String::from("<|im_start|>system\nYou are a helpful assistant. Answer questions as concisely as possible.<|im_end|>"),
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("\n<|im_start|>assistant\n<think>/n/n</think>");

        Ok(prompt)
    }

    fn build_with_tools(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        tools: Option<&[Tool]>,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => match tools {
                Some(tools) if !tools.is_empty() => {
                    let available_tools = serde_json::to_string_pretty(tools).unwrap();

                    let system_prompt = QWEN3_AGENT_SYSTEM_PROMPT.replace("${tool_list}", &available_tools);

                    format!("<|im_start|>system\n{system_prompt}\n<|im_end|>")
                }
                _ => self.create_system_prompt(message),
            },
            _ => match tools {
                Some(tools) if !tools.is_empty() => {
                    let available_tools = serde_json::to_string_pretty(tools).unwrap();

                    let system_prompt = QWEN3_AGENT_SYSTEM_PROMPT.replace("${tool_list}", &available_tools);

                    format!("<|im_start|>system\n{system_prompt}\n<|im_end|>")
                }
                _ => String::from("<|im_start|>system\nYou are a helpful assistant. Answer questions as concisely as possible.<|im_end|>"),
            },
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                ChatCompletionRequestMessage::Tool(message) => {
                    prompt = self.append_tool_message(&prompt, message);
                }
                _ => continue,
            }
        }

        prompt.push_str("\n<|im_start|>assistant\n<think>/n/n</think>");

        Ok(prompt)
    }
}
